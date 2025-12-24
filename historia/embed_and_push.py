#!/usr/bin/env python3
"""
embed_and_push_pinecone.py

- Lee JSONs en ./data/out o ./data/output
- Genera embeddings con OpenAI (services.openai._get_client)
- Upserta en Pinecone via REST API: POST {PINECONE_URL}/vectors/upsert
  - PINECONE_URL debe ser la base URL del index endpoint, por ejemplo:
    https://your-index-xxxx.svc.YOUR_ENVIRONMENT.pinecone.io
  - PINECONE_API_KEY en headers: Api-Key
  - Opcional: PINECONE_NAMESPACE para agrupar vectores

Config (en .env o env):
  OPENAI_API_KEY=sk-...
  PINECONE_URL=https://<your-index>.<region>.svc.pinecone.io
  PINECONE_API_KEY=<pinecone-api-key>
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small
  PINECONE_NAMESPACE=optional_namespace

Instalá dependencias:
  pip install python-dotenv requests
"""
from pathlib import Path
import os
import json
import time
import math
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE = Path.cwd()
CANDIDATES = [BASE / "data" / "out", BASE / "data" / "output", BASE / "data/out", BASE / "data/output"]
IN_DIR = next((p for p in CANDIDATES if p.exists()), None)
if IN_DIR is None:
    raise SystemExit("No se encontró carpeta de input. Crea ./data/out o ./data/output y pon los .json ahí.")

EMBED_DIR = BASE / "data" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = EMBED_DIR / "index.jsonl"

# Pinecone config (from env)
PINECONE_URL = os.getenv("PINECONE_URL")  # e.g. https://<index>-xxxx.svc.<region>.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # optional

# OpenAI embedding model
EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Import wrapper to get OpenAI client
try:
    from services.openai import _get_client
except Exception as e:
    raise SystemExit(f"No pude importar _get_client desde services.openai: {e}")

import requests

# Pinecone upsert batch size (safe default)
UPSERT_BATCH_SIZE = 100

def make_embedding(text: str):
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    try:
        return resp.data[0].embedding
    except Exception:
        # fallback parsing
        try:
            d = json.loads(json.dumps(resp, default=str))
            return d["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"No pude extraer embedding: {e}; raw: {resp}")

def pinecone_upsert(vectors_batch):
    """
    vectors_batch: list of dicts like {"id": str, "values": [...], "metadata": {...}}
    Calls POST {PINECONE_URL}/vectors/upsert with body:
    {"vectors": [...], "namespace": "optional"}
    """
    if not PINECONE_URL or not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_URL and PINECONE_API_KEY must be set in env")

    url = PINECONE_URL.rstrip("/") + "/vectors/upsert"
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY
    }
    body = {"vectors": vectors_batch}
    if PINECONE_NAMESPACE:
        body["namespace"] = PINECONE_NAMESPACE

    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    return r.json()

def process_file(path: Path, idx_offset=0):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"[WARN] skip {path} — expected array")
        return 0

    to_upsert = []
    written = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        bullet_id = obj.get("bullet_id") or obj.get("bulletId") or None
        commentId = obj.get("commentId")
        createdAt = obj.get("createdAt")
        formatted = obj.get("formatted") or []
        if not formatted or not isinstance(formatted, list):
            continue
        text = formatted[0]
        if not text or not isinstance(text, str):
            continue

        try:
            emb = make_embedding(text)
        except Exception as e:
            print(f"[ERROR] embedding failed for {bullet_id or commentId}: {e}")
            continue

        metadata = {
            "bullet_id": bullet_id,
            "commentId": commentId,
            "createdAt": createdAt,
            "team": obj.get("Team", []),
            "sub_title": obj.get("Sub-Title", []),
            "sub_sub_title": obj.get("Sub-Sub-Title", []),
            "country": obj.get("country", []),
            "source_file": path.name
        }

        vid = bullet_id or f"{commentId}_{int(time.time())}_{idx_offset+written}"

        vector_obj = {
            "id": str(vid),
            "values": emb,
            "metadata": metadata
        }

        # local backup
        out_file = EMBED_DIR / f"{vector_obj['id']}.json"
        out_file.write_text(json.dumps({"id": vector_obj["id"], "text": text, "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
        with open(INDEX_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": vector_obj["id"], "file": str(out_file), "metadata": metadata}, ensure_ascii=False) + "\n")

        to_upsert.append(vector_obj)
        written += 1

        # If batch full, flush to Pinecone
        if len(to_upsert) >= UPSERT_BATCH_SIZE:
            try:
                resp = pinecone_upsert(to_upsert)
                print(f"Upserted batch of {len(to_upsert)} vectors. Pinecone response keys: {list(resp.keys())}")
            except Exception as e:
                print(f"[ERROR] pinecone upsert failed: {e}")
            to_upsert = []

        # small throttle
        time.sleep(0.15)

    # flush remaining
    if to_upsert:
        try:
            resp = pinecone_upsert(to_upsert)
            print(f"Upserted final batch of {len(to_upsert)} vectors for file {path.name}")
        except Exception as e:
            print(f"[ERROR] pinecone upsert final failed: {e}")

    return written

def main():
    files = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() == ".json"])
    if not files:
        print("No input files found in", IN_DIR)
        return
    total = 0
    for i, f in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {f.name}")
        w = process_file(f, idx_offset=total)
        print(f"  vectors written/upserted for {f.name}: {w}")
        total += w
    print("Done. Local embeddings dir:", EMBED_DIR)

if __name__ == "__main__":
    main()
