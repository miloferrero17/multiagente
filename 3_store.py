#!/usr/bin/env python3
"""
python3 3_store.py
Uploads vector embeddings from a JSONL file to Pinecone vector database.
All the variables are set via environment variables.
Input: ./data/embeddings/pinecone_ready.jsonl by default.
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

PINECONE_URL = os.getenv("PINECONE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
INPUT_JSONL = Path(os.getenv("INPUT_JSONL", "./data/embeddings/pinecone_ready.jsonl"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
RETRIES = int(os.getenv("RETRIES", "3"))
TIMEOUT = float(os.getenv("PINECONE_TIMEOUT", "30"))

if not PINECONE_URL or not PINECONE_API_KEY:
    raise SystemExit("Set PINECONE_URL and PINECONE_API_KEY in env/.env before running")

if not INPUT_JSONL.exists():
    raise SystemExit(f"Input file not found: {INPUT_JSONL}")

UPsert_endpoint = PINECONE_URL.rstrip("/") + "/vectors/upsert"

headers = {
    "Content-Type": "application/json",
    "Api-Key": PINECONE_API_KEY
}

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] skipping bad jsonl line: {e}")

def batch_iterable(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def upsert_batch(vectors_batch):
    body = {"vectors": vectors_batch}
    if PINECONE_NAMESPACE:
        body["namespace"] = PINECONE_NAMESPACE
    payload = json.dumps(body)
    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(UPsert_endpoint, headers=headers, data=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            backoff = 1.0 * attempt
            print(f"[WARN] upsert attempt {attempt} failed: {e} — backing off {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"Upsert failed after {RETRIES} attempts: {last_exc}")

def validate_vector_shape(vec):
    # basic checks: id, values (list), metadata (dict)
    if not isinstance(vec, dict):
        return False, "not-dict"
    if "id" not in vec or "values" not in vec:
        return False, "missing id/values"
    if not isinstance(vec["values"], list):
        return False, "values-not-list"
    return True, None

def main():
    print("Reading vectors from:", INPUT_JSONL)
    vectors_iter = read_jsonl(INPUT_JSONL)
    total = 0
    failed = 0
    for i, batch in enumerate(batch_iterable(vectors_iter, BATCH_SIZE), start=1):
        # Validate & transform: Pinecone expects {"id": "...", "values": [...], "metadata": {...}}
        to_upsert = []
        for v in batch:
            ok, reason = validate_vector_shape(v)
            if not ok:
                print(f"[SKIP] vector invalid (skipping): {reason} - {v.get('id') if isinstance(v, dict) else 'unknown'}")
                failed += 1
                continue
            # Optional: Ensure id is string
            v["id"] = str(v["id"])
            to_upsert.append(v)

        if not to_upsert:
            continue

        print(f"[{i}] Upserting batch of {len(to_upsert)} vectors...")
        try:
            resp = upsert_batch(to_upsert)
            # Pinecone typical response: {"upserted_count": N} or similar
            print(f"  -> upsert response keys: {list(resp.keys())}")
            total += len(to_upsert)
        except Exception as e:
            print(f"[ERROR] batch upsert failed: {e}")
            failed += len(to_upsert)

    print("Done.")
    print(f"Total upserted (attempted): {total}. Skipped/failed: {failed}")

if __name__ == "__main__":
    main()
