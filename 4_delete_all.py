
#!/usr/bin/env python3
"""
4_delete_all.py

Borra vectores de un índice Pinecone en serverless. Si no se pasa --namespace,
lista namespaces y borra cada uno.

Config (env or .env):
  PINECONE_URL
  PINECONE_API_KEY

Usage:
  # borrar un namespace específico
  python3 4_delete_all.py --namespace trello

  # borrar todos los namespaces (iterará y borrará cada uno)
  python3 4_delete_all.py

  # dry-run (no borra)
  python3 4_delete_all.py --dry-run
"""
import os
import sys
import time
import json
import argparse
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

import requests  # requieres requests instalado

PINECONE_URL = os.getenv("PINECONE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_URL or not PINECONE_API_KEY:
    print("Set PINECONE_URL and PINECONE_API_KEY in env/.env before running", file=sys.stderr)
    sys.exit(2)

HEADERS = {
    "Api-Key": PINECONE_API_KEY,
    "Content-Type": "application/json",
}

RETRIES = 3
TIMEOUT = 30.0


def _get_namespaces(pinecone_url: str) -> List[str]:
    """Intenta listar namespaces del índice. Devuelve lista (puede estar vacía)."""
    url = pinecone_url.rstrip("/") + "/namespaces"
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            # si endpoint no existe, devolvemos []
            if r.status_code == 404:
                return []
            r.raise_for_status()
            data = r.json()
            # data puede ser dict con claves 'namespaces' o ser dict de namespaces
            if isinstance(data, dict):
                # try common shapes
                if "namespaces" in data and isinstance(data["namespaces"], dict):
                    return list(data["namespaces"].keys())
                # sometimes the API returns the namespaces map directly
                return list(data.keys())
            # fallback: si es lista
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            wait = 0.5 * attempt
            print(f"[WARN] list namespaces attempt {attempt} failed: {e} (backoff {wait}s)")
            time.sleep(wait)
    raise RuntimeError("Failed to list namespaces after retries")


def _delete_namespace(pinecone_url: str, namespace: str) -> None:
    """Borra todos los vectores de un namespace usando /vectors/delete?namespace=..."""
    endpoint = pinecone_url.rstrip("/") + f"/vectors/delete?namespace={namespace}"
    payload = {"deleteAll": True}
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(endpoint, headers=HEADERS, json=payload, timeout=TIMEOUT)
            if r.status_code == 404:
                raise RuntimeError(f"404 Not Found for url: {endpoint}")
            r.raise_for_status()
            # normalmente servidor devuelve 200 con body vacío o {}
            print(f"[OK] Deleted namespace '{namespace}' (attempt {attempt}).")
            return
        except Exception as e:
            wait = 1.0 * attempt
            print(f"[WARN] delete namespace '{namespace}' attempt {attempt} failed: {e} (backoff {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to delete namespace '{namespace}' after {RETRIES} attempts")


def main():
    parser = argparse.ArgumentParser(description="Delete all vectors in Pinecone (per-namespace).")
    parser.add_argument("--namespace", "-n", type=str, default=None, help="Namespace to delete (if omitted, all namespaces will be deleted)")
    parser.add_argument("--dry-run", action="store_true", help="Do not perform deletes, just show what would be done")
    args = parser.parse_args()

    # Confirm target
    print("=== Pinecone: DELETE ALL ===")
    print(f"Target: {PINECONE_URL} " + (f"(namespace={args.namespace})" if args.namespace else "(ALL namespaces)"))
    print("IMPORTANT: This will permanently delete vectors from the index above.")

    # Dry-run: list namespaces then exit
    if args.dry_run:
        if args.namespace:
            print(f"[DRY RUN] Would delete namespace: {args.namespace}")
        else:
            namespaces = _get_namespaces(PINECONE_URL)
            print(f"[DRY RUN] Found namespaces: {namespaces}")
        print("[DRY RUN] Exiting.")
        return

    # Ask for confirmation
    confirm = input("Type 'DELETE' to confirm and continue, anything else to abort: ").strip()
    if confirm != "DELETE":
        print("Aborted by user.")
        return

    # Build list of namespaces to delete
    if args.namespace:
        namespaces = [args.namespace]
    else:
        namespaces = _get_namespaces(PINECONE_URL)
        if not namespaces:
            print("[INFO] No namespaces returned by Pinecone. Nothing to delete (or the index is serverless and API didn't expose namespaces).")
            # fallback: try deleting default namespace explicitly (dangerous) or abort
            print("[INFO] Aborting because no namespaces to iterate. If you expect a namespace, re-run with --namespace <name>.")
            return

    # Delete each namespace
    failed = []
    for ns in namespaces:
        print(f"[INFO] Deleting namespace: {ns}")
        try:
            _delete_namespace(PINECONE_URL, ns)
        except Exception as e:
            print(f"[ERROR] deletion failed for namespace {ns}: {e}")
            failed.append((ns, str(e)))

    print("Done.")
    if failed:
        print("Some namespaces failed to delete:")
        for ns, err in failed:
            print(f" - {ns}: {err}")
        sys.exit(1)
    else:
        print("All namespaces deleted successfully.")


if __name__ == "__main__":
    main()
# -----------------------