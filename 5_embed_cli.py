p
'''
#!/usr/bin/env python3
"""
5_embed_cli.py

CLI para generar embeddings (OpenAI) + opcional query directa a Pinecone REST.

Características principales:
- Genera embedding con cliente OpenAI (v1+) si está disponible o con REST si no.
- Auto-guarda el vector (solo array de floats) en last_vector.json para uso posterior.
- --format pinecone => imprime sólo la lista de floats (JSON array).
- --pinecone-query => si PINECONE_API_ENDPOINT y PINECONE_API_KEY están definidos, llama /query.
- --only-text / --csv para formatear salida de Pinecone.
- Soporta namespace por env PINECONE_NAMESPACE o por --namespace.
"""

import os
import sys
import json
import csv
import argparse
import traceback
from dotenv import load_dotenv

load_dotenv()

# Config desde .env / entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_ENDPOINT = os.getenv("PINECONE_API_ENDPOINT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")  # opcional

# Intentar cliente openai nuevo
OpenAIClient = None
USE_OPENAI_CLIENT = False
try:
    from openai import OpenAI as OpenAIClient
    USE_OPENAI_CLIENT = True
except Exception:
    OpenAIClient = None
    USE_OPENAI_CLIENT = False

import requests

def parse_args():
    p = argparse.ArgumentParser(description="Generar embedding (OpenAI) y opcionalmente query a Pinecone.")
    p.add_argument("-t", "--text", help="Texto a vectorizar. Si no se pasa, pide input por consola o stdin.")
    p.add_argument("--out", help="Archivo JSON donde guardar la salida completa (model,input_length,embedding).")
    p.add_argument("--pretty", action="store_true", help="Impresión JSON legible (pretty).")
    p.add_argument("--model", help="Forzar modelo distinto al de .env", default=None)
    p.add_argument("--format", choices=["default", "pinecone"], default="default",
                   help="'pinecone' imprime sólo la lista de floats (JSON array).")
    p.add_argument("--pinecone-query", action="store_true", help="Si están vars de Pinecone, realiza query.")
    p.add_argument("--topk", type=int, default=10, help="Número de resultados top-K a recuperar desde Pinecone (default 10).")
    p.add_argument("--only-text", action="store_true", help="Con --pinecone-query: imprime solo 'rank. score — text'.")
    p.add_argument("--csv", metavar="FILE", help="Guardar top-K results en CSV (rank,id,score,text).")
    p.add_argument("--namespace", help="Namespace para Pinecone (si no, usa PINECONE_NAMESPACE env).")
    p.add_argument("--show-id", action="store_true", help="Con --pinecone-query muestra id junto al texto.")
    return p.parse_args()

def read_input(args):
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data.strip()
    try:
        return input("Texto a vectorizar: ").strip()
    except EOFError:
        return ""

# ---------------- OpenAI embedding (cliente o REST) ----------------
def embed_with_client(text, model, api_key):
    client = OpenAIClient(api_key=api_key)
    resp = client.embeddings.create(model=model, input=text)
    data0 = resp.data[0]
    if hasattr(data0, "embedding"):
        return data0.embedding
    try:
        return data0["embedding"]
    except Exception:
        return resp["data"][0]["embedding"]

def embed_with_rest(text, model, api_key):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for REST call.")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI REST error {r.status_code}: {r.text}")
    j = r.json()
    return j["data"][0]["embedding"]

def get_embedding(text, model, api_key):
    if USE_OPENAI_CLIENT and OpenAIClient is not None:
        return embed_with_client(text, model, api_key)
    return embed_with_rest(text, model, api_key)

# ---------------- Pinecone REST query ----------------
def pinecone_query(endpoint_base, api_key, vector, top_k=10, filter_obj=None, namespace=None):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone query.")
    url = endpoint_base.rstrip("/") + "/query"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "topK": top_k, "includeMetadata": True}
    if filter_obj:
        payload["filter"] = filter_obj
    if namespace:
        payload["namespace"] = namespace
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Pinecone REST error {r.status_code}: {r.text}")
    return r.json()

# ---------------- Helpers ----------------
def pick_text_from_metadata(md):
    if not md:
        return ""
    for k in ("formatted","text","texto","content","body","description","desc","note"):
        v = md.get(k) if isinstance(md, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(md, dict):
        for v in md.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

# ---------------- main ----------------
def main():
    args = parse_args()
    model = args.model or OPENAI_EMBEDDING_MODEL
    api_key = OPENAI_API_KEY

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in env or .env. Set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(2)

    text = read_input(args)
    if not text:
        print("No text provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        emb = get_embedding(text, model, api_key)
    except Exception as e:
        print("ERROR generating embedding:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    # Auto-save del vector en last_vector.json (solo array)
    try:
        with open("last_vector.json", "w", encoding="utf-8") as _f:
            json.dump(emb, _f, ensure_ascii=False)
        print("Saved embedding vector to last_vector.json", file=sys.stderr)
    except Exception as _e:
        print("WARNING: could not save last_vector.json:", _e, file=sys.stderr)

    out = {"model": model, "input_length": len(text), "dim": len(emb) if hasattr(emb, "__len__") else None, "embedding": emb}

    # formato pinecone (solo array)
    if args.format == "pinecone":
        if args.pretty:
            print(json.dumps(emb, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(emb, ensure_ascii=False))
        # también guardar full out si piden --out
        if args.out:
            try:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2 if args.pretty else None)
                print(f"Saved full output to {args.out}", file=sys.stderr)
            except Exception as e:
                print("ERROR saving file:", e, file=sys.stderr)
        return

    # salida por defecto (json con embedding)
    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))

    # guardar si se pidió --out
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2 if args.pretty else None)
            print(f"Saved output to {args.out}", file=sys.stderr)
        except Exception as e:
            print("ERROR saving file:", e, file=sys.stderr)

    # Pinecone query si lo solicitan
    if args.pinecone_query:
        namespace = args.namespace or PINECONE_NAMESPACE
        if not PINECONE_API_ENDPOINT or not PINECONE_API_KEY:
            print("PINECONE_API_ENDPOINT and PINECONE_API_KEY must be set in env to use --pinecone-query", file=sys.stderr)
            sys.exit(4)
        try:
            res = pinecone_query(PINECONE_API_ENDPOINT, PINECONE_API_KEY, emb, top_k=args.topk, filter_obj=None, namespace=namespace)
            matches = res.get("matches", []) or []
            matches_sorted = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)[:args.topk]

            if args.only_text:
                for i, m in enumerate(matches_sorted, start=1):
                    md = m.get("metadata", {}) or {}
                    text_field = pick_text_from_metadata(md)
                    score = m.get("score", "")
                    if args.show_id:
                        print(f"{i}. {score} — {text_field} (id={m.get('id')})")
                    else:
                        print(f"{i}. {score} — {text_field}")
                return

            if args.csv:
                try:
                    with open(args.csv, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["rank", "id", "score", "text"])
                        for i, m in enumerate(matches_sorted, start=1):
                            mid = m.get("id", "")
                            score = m.get("score", "")
                            md = m.get("metadata", {}) or {}
                            text_field = pick_text_from_metadata(md)
                            writer.writerow([i, mid, score, text_field])
                    print(f"Saved top-{args.topk} results to CSV {args.csv}", file=sys.stderr)
                except Exception as e:
                    print("ERROR writing CSV:", e, file=sys.stderr)
                    traceback.print_exc()
                    sys.exit(6)
                return

            # impresión completa ordenada (por defecto)
            for i, m in enumerate(matches_sorted, start=1):
                mid = m.get("id", "")
                score = m.get("score", "")
                md = m.get("metadata", {}) or {}
                text_field = pick_text_from_metadata(md)
                if args.show_id:
                    print(f"{i}. {score} — {text_field} (id={mid})")
                else:
                    print(f"{i}. {score} — {text_field}")

        except Exception as e:
            print("ERROR calling Pinecone:", e, file=sys.stderr)
            traceback.print_exc()
            sys.exit(5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
services/embed_cli.py

CLI para generar embeddings (OpenAI) + opcional query directa a Pinecone REST.

Características:
- Usa cliente openai v1+ si está disponible.
- Si no, usa requests para llamar a OpenAI REST (/v1/embeddings).
- --format pinecone => imprime sólo la lista de floats (JSON array).
- --pinecone-query => si PINECONE_API_ENDPOINT y PINECONE_API_KEY están definidos, llama /query con filter preset.
"""

import os, sys, json, argparse, traceback
from dotenv import load_dotenv

load_dotenv()  # carga .env si existe

# Config desde .env / entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_ENDPOINT = os.getenv("PINECONE_API_ENDPOINT")  # ej: https://trello-xxx.svc.region.pinecone.io
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Intentar cliente openai nuevo
OpenAIClient = None
USE_OPENAI_CLIENT = False
try:
    from openai import OpenAI as OpenAIClient
    USE_OPENAI_CLIENT = True
except Exception:
    OpenAIClient = None
    USE_OPENAI_CLIENT = False

# requests fallback para OpenAI REST y Pinecone REST
import requests

def parse_args():
    p = argparse.ArgumentParser(description="CLI para generar embeddings (OpenAI) y opcionalmente query a Pinecone.")
    p.add_argument("-t", "--text", help="Texto a vectorizar. Si no se pasa, pide input por consola o stdin.")
    p.add_argument("--out", help="Archivo JSON donde guardar la salida (opcional).")
    p.add_argument("--pretty", action="store_true", help="Impresión JSON legible (pretty).")
    p.add_argument("--model", help="Forzar modelo distinto al de .env", default=None)
    p.add_argument("--format", choices=["default", "pinecone"], default="default",
                   help="Formato de salida. 'pinecone' imprime sólo la lista de floats (JSON array).")
    p.add_argument("--pinecone-query", action="store_true",
                   help="Si están PINECONE_API_ENDPOINT y PINECONE_API_KEY en env, llama /query con filter preset.")
    return p.parse_args()

def read_input(args):
    if args.text:
        return args.text
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data.strip()
    try:
        return input("Texto a vectorizar: ").strip()
    except EOFError:
        return ""

# ---------------- OpenAI embedding (cliente o REST) ----------------
def embed_with_client(text, model, api_key):
    client = OpenAIClient(api_key=api_key)
    resp = client.embeddings.create(model=model, input=text)
    data0 = resp.data[0]
    # devuelve la embedding, cubriendo objetos o dicts
    if hasattr(data0, "embedding"):
        return data0.embedding
    try:
        return data0["embedding"]
    except Exception:
        return resp["data"][0]["embedding"]

def embed_with_rest(text, model, api_key):
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for REST call.")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI REST error {r.status_code}: {r.text}")
    j = r.json()
    return j["data"][0]["embedding"]

def get_embedding(text, model, api_key):
    # preferir cliente si está
    if USE_OPENAI_CLIENT and OpenAIClient is not None:
        return embed_with_client(text, model, api_key)
    # fallback REST
    return embed_with_rest(text, model, api_key)

# ---------------- Pinecone REST query ----------------
DEFAULT_FILTER = {"team": "bari", "fecha": "2025-Q4", "bandera": "mastercard"}

def pinecone_query(endpoint_base, api_key, vector, top_k=10, filter_obj=None):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone query.")
    url = endpoint_base.rstrip("/") + "/query"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "topK": top_k, "includeMetadata": True}
    if filter_obj:
        payload["filter"] = filter_obj
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Pinecone REST error {r.status_code}: {r.text}")
    return r.json()

# ---------------- main ----------------
def main():
    args = parse_args()
    model = args.model or OPENAI_EMBEDDING_MODEL
    api_key = OPENAI_API_KEY

    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in env or .env. Set OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(2)

    text = read_input(args)
    if not text:
        print("No text provided. Exiting.", file=sys.stderr)
        sys.exit(1)

    try:
        emb = get_embedding(text, model, api_key)
    except Exception as e:
        print("ERROR generating embedding:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(3)

    out = {"model": model, "input_length": len(text), "dim": len(emb) if hasattr(emb, "__len__") else None, "embedding": emb}

    # Si piden formato pinecone -> imprimir solo la lista de floats
    if args.format == "pinecone":
        # pretty print array
        if args.pretty:
            print(json.dumps(emb, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(emb, ensure_ascii=False))
        # también guardar si se pidió --out
        if args.out:
            try:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2 if args.pretty else None)
                print(f"Saved full output to {args.out}", file=sys.stderr)
            except Exception as e:
                print("ERROR saving file:", e, file=sys.stderr)
        return

    # salida por defecto (json con embedding)
    if args.pretty:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out, ensure_ascii=False))

    # guardar si se pidió
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2 if args.pretty else None)
            print(f"Saved output to {args.out}", file=sys.stderr)
        except Exception as e:
            print("ERROR saving file:", e, file=sys.stderr)

    # si piden query a Pinecone, hacerlo
    if args.pinecone_query:
        if not PINECONE_API_ENDPOINT or not PINECONE_API_KEY:
            print("PINECONE_API_ENDPOINT and PINECONE_API_KEY must be set in env to use --pinecone-query", file=sys.stderr)
            sys.exit(4)
        try:
            print("Calling Pinecone /query with filter:", DEFAULT_FILTER, file=sys.stderr)
            res = pinecone_query(PINECONE_API_ENDPOINT, PINECONE_API_KEY, emb, top_k=10, filter_obj=DEFAULT_FILTER)
            print(json.dumps(res, ensure_ascii=False, indent=2))
        except Exception as e:
            print("ERROR calling Pinecone:", e, file=sys.stderr)
            traceback.print_exc()
            sys.exit(5)

if __name__ == "__main__":
    main()
'''