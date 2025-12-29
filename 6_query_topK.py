
#!/usr/bin/env python3
"""
python3 6_query_topk.py
Query Pinecone and print top similar vectors (score - formatted).
Usage examples:
  python3 6_query_topk.py --topk 100 --return 20 clear--team bari --show-id --debug
  python3 6_query_topk.py --fetch-id 69492de0a72a3fe09a23546a_4 --debug
  python3 6_query_topk.py --vector last_vector.json --topk 50 --return 20 --csv out.csv
All the variables are set via environment variables.
Output: printed to console, optional CSV file.
"""
from typing import Any, Optional, Tuple, List
import os
import sys
import json
import time
import argparse
import csv
import requests
from pathlib import Path

DEFAULT_VECTOR_FILE = "last_vector.json"
DEFAULT_NAMESPACE_ENV = "PINECONE_NAMESPACE"
TEAM_SAVE_PATH = Path.home() / ".pinecone_team"  # archivo opcional para guardar team

# ---------------- helpers ----------------
def _truncate(s: str, n: int) -> str:
    if not n or n <= 0:
        return s
    if len(s) <= n:
        return s
    return s[:n] + "…"

def _as_str_from_value(v: Any) -> Optional[str]:
    """Normalize a value that might be str, list, tuple; return first non-empty string or None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    return None

def _find_formatted_recursive_with_path(obj: Any, path: str = "") -> Optional[Tuple[str, str]]:
    """
    Search recursively for keys 'formatted' or 'formatted_text'.
    Returns (path, value) or None.
    """
    if obj is None:
        return None

    # dict: check for keys first
    if isinstance(obj, dict):
        for key in ("formatted", "formatted_text"):
            if key in obj:
                v = obj.get(key)
                s = _as_str_from_value(v)
                if s:
                    p = f"{path}.{key}" if path else key
                    return (p, s)
                if isinstance(v, (dict, list, tuple)):
                    res = _find_formatted_recursive_with_path(v, f"{path}.{key}" if path else key)
                    if res:
                        return res
        # then traverse other keys
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            res = _find_formatted_recursive_with_path(v, new_path)
            if res:
                return res
        return None

    # list/tuple: iterate elements
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            if isinstance(item, str) and item.strip():
                return (new_path, item.strip())
            res = _find_formatted_recursive_with_path(item, new_path)
            if res:
                return res
        return None

    return None

def pick_formatted_from_metadata_with_path(md: dict, truncate_chars: int = 1000) -> Tuple[str, str]:
    """
    Returns (path, formatted_text) if found, or ("MISSING", "").
    """
    if not md or not isinstance(md, dict):
        return ("MISSING", "")

    # root check
    for key in ("formatted", "formatted_text"):
        if key in md:
            v = md.get(key)
            s = _as_str_from_value(v)
            if s:
                return (key, _truncate(s, truncate_chars))
            if isinstance(v, (dict, list, tuple)):
                res = _find_formatted_recursive_with_path(v, key)
                if res:
                    p, val = res
                    return (p, _truncate(val, truncate_chars))

    # full recursive search
    res = _find_formatted_recursive_with_path(md, "")
    if res:
        p, val = res
        return (p, _truncate(val, truncate_chars))
    return ("MISSING", "")

# ---------------- Pinecone REST helpers ----------------
def pinecone_query(endpoint_base: str, api_key: str, vector, top_k: int = 100, namespace: Optional[str] = None, timeout: int = 60, filter_obj: Optional[dict] = None):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone query.")
    url = endpoint_base.rstrip("/") + "/query"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "topK": top_k, "includeMetadata": True}
    if namespace:
        payload["namespace"] = namespace
    if filter_obj:
        payload["filter"] = filter_obj
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def pinecone_fetch_by_id(endpoint_base: str, api_key: str, ids: List[str], namespace: Optional[str] = None, timeout: int = 30):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone fetch.")
    url = endpoint_base.rstrip("/") + "/vectors/fetch"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"ids": ids}
    if namespace:
        payload["namespace"] = namespace
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------- CLI & main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Query Pinecone and print top similar vectors (score - formatted).")
    p.add_argument("--vector", default=DEFAULT_VECTOR_FILE, help="Archivo JSON con el vector (array floats).")
    p.add_argument("--topk", type=int, default=50, help="topK pedido al servidor (default 50).")
    p.add_argument("--return", dest="ret", type=int, default=20, help="Cuántos mostrar (default 20).")
    p.add_argument("--csv", help="Guardar resultados en CSV (rank,id,score,path,text).")
    p.add_argument("--timeout", type=int, default=60, help="Timeout por request (default 60s).")
    p.add_argument("--tries", type=int, default=3, help="Reintentos ante timeout (default 3).")
    p.add_argument("--namespace", help="Namespace de Pinecone (si no, usa env PINECONE_NAMESPACE).")
    p.add_argument("--truncate", type=int, default=1000, help="Truncar texto a N chars (0 = no truncar).")
    p.add_argument("--show-id", action="store_true", help="Mostrar id entre paréntesis junto al texto.")
    p.add_argument("--debug", action="store_true", help="Imprime metadata cruda y path de cada match.")
    p.add_argument("--skip-empty", action="store_true", help="Saltar matches sin formatted (para asegurar N filas con texto).")
    p.add_argument("--fetch-id", nargs='+', help="No hace query: fetch por id(s) y muestra metadata (útil para inspeccionar un id específico).")
    p.add_argument("--team", help="Filtrar por metadata.team/Team (escritura libre). Si no se pasa, se preguntará en consola si estás en TTY.")
    return p.parse_args()

def build_team_filter(team_value: Optional[str]) -> Optional[dict]:
    """
    Build a filter that tries both 'team' and 'Team' keys (case-sensitive).
    Uses $in so it matches scalar or list metadata.
    """
    if not team_value:
        return None
    tv = team_value.strip()
    if not tv:
        return None
    # Use $or to match either key name
    return {
        "$or": [
            {"team": {"$in": [tv]}},
            {"Team": {"$in": [tv]}}
        ]
    }

def _maybe_load_saved_team() -> Optional[str]:
    """Load saved team from HOME file if exists."""
    try:
        if TEAM_SAVE_PATH.exists():
            txt = TEAM_SAVE_PATH.read_text(encoding="utf-8").strip()
            return txt if txt else None
    except Exception:
        pass
    return None

def _maybe_save_team(team_value: str):
    """Save the provided team value to ~/.pinecone_team (best-effort)."""
    try:
        TEAM_SAVE_PATH.write_text(team_value.strip(), encoding="utf-8")
        print(f"(Saved team to {TEAM_SAVE_PATH})", file=sys.stderr)
    except Exception as e:
        print(f"(Could not save team to {TEAM_SAVE_PATH}: {e})", file=sys.stderr)

def main():
    args = parse_args()

    endpoint = os.getenv("PINECONE_API_ENDPOINT")
    api_key = os.getenv("PINECONE_API_KEY")
    env_ns = os.getenv(DEFAULT_NAMESPACE_ENV)
    namespace = args.namespace or env_ns

    if not endpoint or not api_key:
        print("ERROR: exportá PINECONE_API_ENDPOINT y PINECONE_API_KEY en tu shell", file=sys.stderr)
        sys.exit(2)

    # TEAM handling: priority order
    # 1) --team CLI argument (if provided) -> use it (no interactive prompt)
    # 2) if not provided and TTY: try to load saved value and ask to confirm/edit; otherwise ask
    # 3) if not TTY and no --team, no filter applied
    team_value = args.team

    if not team_value and sys.stdin.isatty():
        # try load saved team
        saved = _maybe_load_saved_team()
        if saved:
            try:
                prompt = input(f"Team [{saved}] (enter para aceptar / escribir nuevo / vacio = no filter): ").strip()
            except EOFError:
                prompt = ""
            if prompt == "":
                team_value = saved
            else:
                team_value = prompt
        else:
            try:
                team_prompt = input("Filter by Team (leave empty = no filter): ").strip()
                team_value = team_prompt if team_prompt else None
            except EOFError:
                team_value = None

        # si hay un valor no-empty, preguntar si guardar
        if team_value:
            try:
                save_q = input(f"Guardar '{team_value}' en {TEAM_SAVE_PATH} para uso futuro? [y/N]: ").strip().lower()
            except EOFError:
                save_q = "n"
            if save_q in ("y", "yes"):
                _maybe_save_team(team_value)

    # if not TTY and no --team, we do not prompt and team_value remains None

    filter_obj = build_team_filter(team_value)

    # fetch by id mode
    if args.fetch_id:
        try:
            resp = pinecone_fetch_by_id(endpoint, api_key, ids=args.fetch_id, namespace=namespace, timeout=args.timeout)
        except Exception as e:
            print("ERROR fetching ids:", e, file=sys.stderr)
            sys.exit(3)
        vectors = resp.get("vectors", {}) or {}
        for vid, vobj in vectors.items():
            md = vobj.get("metadata", {}) or {}
            path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
            print(f"id={vid} -> path={path} -> text='{text}'")
            if args.debug:
                print("METADATA:", json.dumps(md, ensure_ascii=False, indent=2))
        return

    # read vector
    if not os.path.exists(args.vector):
        print(f"ERROR: vector file not found: {args.vector}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(args.vector, "r", encoding="utf-8") as f:
            vector = json.load(f)
    except Exception as e:
        print("ERROR reading vector file:", e, file=sys.stderr)
        sys.exit(1)

    # query with retries
    attempt = 0
    last_exc = None
    resp = None
    while attempt < args.tries:
        attempt += 1
        try:
            resp = pinecone_query(endpoint, api_key, vector, top_k=args.topk, namespace=namespace, timeout=args.timeout, filter_obj=filter_obj)
            break
        except requests.exceptions.ReadTimeout as e:
            print(f"[Attempt {attempt}] ReadTimeout (timeout={args.timeout}s). Retrying...", file=sys.stderr)
            last_exc = e
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print("Request error when calling Pinecone:", e, file=sys.stderr)
            last_exc = e
            break
    else:
        print("All attempts failed. Last exception:", last_exc, file=sys.stderr)
        sys.exit(3)

    matches = resp.get("matches", []) or []
    if not matches:
        print("No matches returned by Pinecone (0).", file=sys.stderr)
        if filter_obj:
            print(f"(filter applied: {json.dumps(filter_obj, ensure_ascii=False)})", file=sys.stderr)
        sys.exit(0)

    # sort and collect
    candidates = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

    results = []
    processed = 0
    # if skip-empty requested, we will keep scanning up to topk to collect args.ret non-empty formatted rows
    for m in candidates:
        processed += 1
        md = m.get("metadata", {}) or {}
        path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
        results.append({
            "id": m.get("id", ""),
            "score": m.get("score", 0) or 0,
            "path": path,
            "text": text,
            "raw_metadata": md
        })

        # break conditions
        if not args.skip_empty and len(results) >= args.ret:
            break

        if args.skip_empty:
            nonempty = [r for r in results if r["text"]]
            if len(nonempty) >= args.ret:
                results = nonempty[:args.ret]
                break

        if processed >= args.topk:
            break

    if not args.skip_empty:
        results = results[:args.ret]

    # print (modificado: omitimos path cuando solo es 'formatted' / 'formatted_text' / 'MISSING')
    for idx, r in enumerate(results, start=1):
        show_path = r["path"] and r["path"] not in ("formatted", "formatted_text", "MISSING")
        if args.show_id:
            if show_path:
                print(f"{idx}. {r['score']:.4f} - {r['text']} (id={r['id']})  path={r['path']}")
            else:
                print(f"{idx}. {r['score']:.4f} - {r['text']} (id={r['id']})")
        else:
            if show_path:
                print(f"{idx}. {r['score']:.4f} - {r['text']}  path={r['path']}")
            else:
                print(f"{idx}. {r['score']:.4f} - {r['text']}")
        if args.debug:
            print("   RAW_METADATA:", json.dumps(r["raw_metadata"], ensure_ascii=False))
            print()

    # csv (si pediste csv, guardamos path vacío si es uno de los valores 'por defecto')
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["rank", "id", "score", "path", "text"])
                for i, r in enumerate(results, start=1):
                    path_for_csv = r["path"] if r["path"] not in ("formatted", "formatted_text", "MISSING") else ""
                    writer.writerow([i, r["id"], f"{r['score']:.6f}", path_for_csv, r["text"]])
            print(f"Saved results to CSV: {args.csv}", file=sys.stderr)
        except Exception as e:
            print("ERROR writing CSV:", e, file=sys.stderr)

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
6_query_topk.py

Query Pinecone and print top similar vectors (score - formatted).

Usage examples:
  python3 topk.py --topk 100 --return 20 --team bari --show-id --debug
  python3 topk.py --fetch-id 69492de0a72a3fe09a23546a_4 --debug
  python3 topk.py --vector last_vector.json --topk 50 --return 20 --csv out.csv

Environment:
  PINECONE_API_ENDPOINT  (required for queries/fetch)
  PINECONE_API_KEY       (required)
  PINECONE_NAMESPACE     (optional default namespace)
"""
from typing import Any, Optional, Tuple, List
import os
import sys
import json
import time
import argparse
import csv
import requests
from pathlib import Path

DEFAULT_VECTOR_FILE = "last_vector.json"
DEFAULT_NAMESPACE_ENV = "PINECONE_NAMESPACE"
TEAM_SAVE_PATH = Path.home() / ".pinecone_team"  # archivo opcional para guardar team

# ---------------- helpers ----------------
def _truncate(s: str, n: int) -> str:
    if not n or n <= 0:
        return s
    if len(s) <= n:
        return s
    return s[:n] + "…"

def _as_str_from_value(v: Any) -> Optional[str]:
    """Normalize a value that might be str, list, tuple; return first non-empty string or None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    return None

def _find_formatted_recursive_with_path(obj: Any, path: str = "") -> Optional[Tuple[str, str]]:
    """
    Search recursively for keys 'formatted' or 'formatted_text'.
    Returns (path, value) or None.
    """
    if obj is None:
        return None

    # dict: check for keys first
    if isinstance(obj, dict):
        for key in ("formatted", "formatted_text"):
            if key in obj:
                v = obj.get(key)
                s = _as_str_from_value(v)
                if s:
                    p = f"{path}.{key}" if path else key
                    return (p, s)
                if isinstance(v, (dict, list, tuple)):
                    res = _find_formatted_recursive_with_path(v, f"{path}.{key}" if path else key)
                    if res:
                        return res
        # then traverse other keys
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            res = _find_formatted_recursive_with_path(v, new_path)
            if res:
                return res
        return None

    # list/tuple: iterate elements
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            if isinstance(item, str) and item.strip():
                return (new_path, item.strip())
            res = _find_formatted_recursive_with_path(item, new_path)
            if res:
                return res
        return None

    return None

def pick_formatted_from_metadata_with_path(md: dict, truncate_chars: int = 1000) -> Tuple[str, str]:
    """
    Returns (path, formatted_text) if found, or ("MISSING", "").
    """
    if not md or not isinstance(md, dict):
        return ("MISSING", "")

    # root check
    for key in ("formatted", "formatted_text"):
        if key in md:
            v = md.get(key)
            s = _as_str_from_value(v)
            if s:
                return (key, _truncate(s, truncate_chars))
            if isinstance(v, (dict, list, tuple)):
                res = _find_formatted_recursive_with_path(v, key)
                if res:
                    p, val = res
                    return (p, _truncate(val, truncate_chars))

    # full recursive search
    res = _find_formatted_recursive_with_path(md, "")
    if res:
        p, val = res
        return (p, _truncate(val, truncate_chars))
    return ("MISSING", "")

# ---------------- Pinecone REST helpers ----------------
def pinecone_query(endpoint_base: str, api_key: str, vector, top_k: int = 100, namespace: Optional[str] = None, timeout: int = 60, filter_obj: Optional[dict] = None):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone query.")
    url = endpoint_base.rstrip("/") + "/query"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "topK": top_k, "includeMetadata": True}
    if namespace:
        payload["namespace"] = namespace
    if filter_obj:
        payload["filter"] = filter_obj
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def pinecone_fetch_by_id(endpoint_base: str, api_key: str, ids: List[str], namespace: Optional[str] = None, timeout: int = 30):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone fetch.")
    url = endpoint_base.rstrip("/") + "/vectors/fetch"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"ids": ids}
    if namespace:
        payload["namespace"] = namespace
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------- CLI & main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Query Pinecone and print top similar vectors (score - formatted).")
    p.add_argument("--vector", default=DEFAULT_VECTOR_FILE, help="Archivo JSON con el vector (array floats).")
    p.add_argument("--topk", type=int, default=50, help="topK pedido al servidor (default 50).")
    p.add_argument("--return", dest="ret", type=int, default=20, help="Cuántos mostrar (default 20).")
    p.add_argument("--csv", help="Guardar resultados en CSV (rank,id,score,path,text).")
    p.add_argument("--timeout", type=int, default=60, help="Timeout por request (default 60s).")
    p.add_argument("--tries", type=int, default=3, help="Reintentos ante timeout (default 3).")
    p.add_argument("--namespace", help="Namespace de Pinecone (si no, usa env PINECONE_NAMESPACE).")
    p.add_argument("--truncate", type=int, default=1000, help="Truncar texto a N chars (0 = no truncar).")
    p.add_argument("--show-id", action="store_true", help="Mostrar id entre paréntesis junto al texto.")
    p.add_argument("--debug", action="store_true", help="Imprime metadata cruda y path de cada match.")
    p.add_argument("--skip-empty", action="store_true", help="Saltar matches sin formatted (para asegurar N filas con texto).")
    p.add_argument("--fetch-id", nargs='+', help="No hace query: fetch por id(s) y muestra metadata (útil para inspeccionar un id específico).")
    p.add_argument("--team", help="Filtrar por metadata.team/Team (escritura libre). Si no se pasa, se preguntará en consola si estás en TTY.")
    return p.parse_args()

def build_team_filter(team_value: Optional[str]) -> Optional[dict]:
    """
    Build a filter that tries both 'team' and 'Team' keys (case-sensitive).
    Uses $in so it matches scalar or list metadata.
    """
    if not team_value:
        return None
    tv = team_value.strip()
    if not tv:
        return None
    # Use $or to match either key name
    return {
        "$or": [
            {"team": {"$in": [tv]}},
            {"Team": {"$in": [tv]}}
        ]
    }

def _maybe_load_saved_team() -> Optional[str]:
    """Load saved team from HOME file if exists."""
    try:
        if TEAM_SAVE_PATH.exists():
            txt = TEAM_SAVE_PATH.read_text(encoding="utf-8").strip()
            return txt if txt else None
    except Exception:
        pass
    return None

def _maybe_save_team(team_value: str):
    """Save the provided team value to ~/.pinecone_team (best-effort)."""
    try:
        TEAM_SAVE_PATH.write_text(team_value.strip(), encoding="utf-8")
        print(f"(Saved team to {TEAM_SAVE_PATH})", file=sys.stderr)
    except Exception as e:
        print(f"(Could not save team to {TEAM_SAVE_PATH}: {e})", file=sys.stderr)

def main():
    args = parse_args()

    endpoint = os.getenv("PINECONE_API_ENDPOINT")
    api_key = os.getenv("PINECONE_API_KEY")
    env_ns = os.getenv(DEFAULT_NAMESPACE_ENV)
    namespace = args.namespace or env_ns

    if not endpoint or not api_key:
        print("ERROR: exportá PINECONE_API_ENDPOINT y PINECONE_API_KEY en tu shell", file=sys.stderr)
        sys.exit(2)

    # TEAM handling: priority order
    # 1) --team CLI argument (if provided) -> use it (no interactive prompt)
    # 2) if not provided and TTY: try to load saved value and ask to confirm/edit; otherwise ask
    # 3) if not TTY and no --team, no filter applied
    team_value = args.team

    if not team_value and sys.stdin.isatty():
        # try load saved team
        saved = _maybe_load_saved_team()
        if saved:
            try:
                prompt = input(f"Team [{saved}] (enter para aceptar / escribir nuevo / vacio = no filter): ").strip()
            except EOFError:
                prompt = ""
            if prompt == "":
                team_value = saved
            else:
                team_value = prompt
        else:
            try:
                team_prompt = input("Filter by Team (leave empty = no filter): ").strip()
                team_value = team_prompt if team_prompt else None
            except EOFError:
                team_value = None

        # si hay un valor no vacío, preguntar si guardar
        if team_value:
            try:
                save_q = input(f"Guardar '{team_value}' en {TEAM_SAVE_PATH} para uso futuro? [y/N]: ").strip().lower()
            except EOFError:
                save_q = "n"
            if save_q in ("y", "yes"):
                _maybe_save_team(team_value)

    # if not TTY and no --team, we do not prompt and team_value remains None

    filter_obj = build_team_filter(team_value)

    # fetch by id mode
    if args.fetch_id:
        try:
            resp = pinecone_fetch_by_id(endpoint, api_key, ids=args.fetch_id, namespace=namespace, timeout=args.timeout)
        except Exception as e:
            print("ERROR fetching ids:", e, file=sys.stderr)
            sys.exit(3)
        vectors = resp.get("vectors", {}) or {}
        for vid, vobj in vectors.items():
            md = vobj.get("metadata", {}) or {}
            path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
            print(f"id={vid} -> path={path} -> text='{text}'")
            if args.debug:
                print("METADATA:", json.dumps(md, ensure_ascii=False, indent=2))
        return

    # read vector
    if not os.path.exists(args.vector):
        print(f"ERROR: vector file not found: {args.vector}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(args.vector, "r", encoding="utf-8") as f:
            vector = json.load(f)
    except Exception as e:
        print("ERROR reading vector file:", e, file=sys.stderr)
        sys.exit(1)

    # query with retries
    attempt = 0
    last_exc = None
    resp = None
    while attempt < args.tries:
        attempt += 1
        try:
            resp = pinecone_query(endpoint, api_key, vector, top_k=args.topk, namespace=namespace, timeout=args.timeout, filter_obj=filter_obj)
            break
        except requests.exceptions.ReadTimeout as e:
            print(f"[Attempt {attempt}] ReadTimeout (timeout={args.timeout}s). Retrying...", file=sys.stderr)
            last_exc = e
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print("Request error when calling Pinecone:", e, file=sys.stderr)
            last_exc = e
            break
    else:
        print("All attempts failed. Last exception:", last_exc, file=sys.stderr)
        sys.exit(3)

    matches = resp.get("matches", []) or []
    if not matches:
        print("No matches returned by Pinecone (0).", file=sys.stderr)
        if filter_obj:
            print(f"(filter applied: {json.dumps(filter_obj, ensure_ascii=False)})", file=sys.stderr)
        sys.exit(0)

    # sort and collect
    candidates = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

    results = []
    processed = 0
    # if skip-empty requested, we will keep scanning up to topk to collect args.ret non-empty formatted rows
    for m in candidates:
        processed += 1
        md = m.get("metadata", {}) or {}
        path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
        results.append({
            "id": m.get("id", ""),
            "score": m.get("score", 0) or 0,
            "path": path,
            "text": text,
            "raw_metadata": md
        })

        # break conditions
        if not args.skip_empty and len(results) >= args.ret:
            break

        if args.skip_empty:
            nonempty = [r for r in results if r["text"]]
            if len(nonempty) >= args.ret:
                results = nonempty[:args.ret]
                break

        if processed >= args.topk:
            break

    if not args.skip_empty:
        results = results[:args.ret]

    # print
    for idx, r in enumerate(results, start=1):
        if args.show_id:
            print(f"{idx}. {r['score']:.4f} - {r['text']} (id={r['id']})  path={r['path']}")
        else:
            print(f"{idx}. {r['score']:.4f} - {r['text']}  path={r['path']}")
        if args.debug:
            print("   RAW_METADATA:", json.dumps(r["raw_metadata"], ensure_ascii=False))
            print()

    # csv
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["rank", "id", "score", "path", "text"])
                for i, r in enumerate(results, start=1):
                    writer.writerow([i, r["id"], f"{r['score']:.6f}", r["path"], r["text"]])
            print(f"Saved results to CSV: {args.csv}", file=sys.stderr)
        except Exception as e:
            print("ERROR writing CSV:", e, file=sys.stderr)

if __name__ == "__main__":
    main()


'''
#!/usr/bin/env python3
"""
6_query_topK.py

Query Pinecone and print top similar vectors (score - formatted).

Características principales:
- Busca 'formatted' o 'formatted_text' en metadata en la raíz.
- Si no está en la raíz, busca recursivamente en dicts y listas dentro de metadata.
- Soporta casos donde formatted es una lista de strings (usa el primer string no vacío).
- Opciones:
    --vector    FILE   : archivo JSON con el vector (default last_vector.json)
    --topk      N      : topK pedido al servidor (default 50)
    --return    M      : cuántos mostrar (default 20)
    --csv       FILE   : guardar resultados CSV (rank,id,score,path,text)
    --namespace NS     : namespace de Pinecone (o usa PINECONE_NAMESPACE env)
    --timeout   S      : timeout por request (default 60s)
    --tries     N      : reintentos ante timeout (default 3)
    --truncate  N      : truncar texto a N chars (0 = no truncar, default 1000)
    --show-id            : mostrar id junto al texto
    --debug              : imprime metadata cruda + path para depuración
    --skip-empty         : saltear matches sin formatted y asegurar --return filas con texto (pedir topk>return recomendado)
    --fetch-id ID [ID2 ...] : no hace query, hace fetch por id(s) y muestra metadata (útil para inspeccionar un id específico)
"""
from typing import Any, Optional, Tuple, List
import os
import sys
import json
import time
import argparse
import csv
import requests

DEFAULT_VECTOR_FILE = "last_vector.json"
DEFAULT_NAMESPACE_ENV = "PINECONE_NAMESPACE"

# ---------------- helpers ----------------
def _truncate(s: str, n: int) -> str:
    if not n or n <= 0:
        return s
    if len(s) <= n:
        return s
    return s[:n] + "…"

def _as_str_from_value(v: Any) -> Optional[str]:
    """Normaliza un value que puede ser str, list, tuple; devuelve primera string no-vacía o None."""
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, str) and item.strip():
                return item.strip()
            # si el item es dict/list puede contener nested formatted (lo manejamos recursivamente en caller)
        return None
    return None

def _find_formatted_recursive_with_path(obj: Any, path: str = "") -> Optional[Tuple[str, str]]:
    """
    Busca recursivamente claves 'formatted' o 'formatted_text'.
    Soporta strings, listas/tuplas de strings o estructuras anidadas.
    Devuelve (ruta, valor) si encuentra, o None si no.
    Ruta ejemplo: "content[0].blocks[2].formatted"
    """
    if obj is None:
        return None

    # dict: prioridad por claves exactas en el dict actual
    if isinstance(obj, dict):
        for key in ("formatted", "formatted_text"):
            if key in obj:
                v = obj.get(key)
                # si es str o lista/tuple con strings, extraer primera string
                s = _as_str_from_value(v)
                if s:
                    p = f"{path}.{key}" if path else key
                    return (p, s)
                # si el valor es dict/list, buscar dentro
                if isinstance(v, (dict, list, tuple)):
                    res = _find_formatted_recursive_with_path(v, f"{path}.{key}" if path else key)
                    if res:
                        return res
        # si no, recorrer valores
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            res = _find_formatted_recursive_with_path(v, new_path)
            if res:
                return res
        return None

    # list/tuple: iterar elementos
    if isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]"
            # si el item es str directamente, devolverlo (caso raro)
            if isinstance(item, str) and item.strip():
                return (new_path, item.strip())
            res = _find_formatted_recursive_with_path(item, new_path)
            if res:
                return res
        return None

    # primitivo: nothing to do
    return None

def pick_formatted_from_metadata_with_path(md: dict, truncate_chars: int = 1000) -> Tuple[str, str]:
    """
    Devuelve (path, formatted_text) si encuentra formatted/ formatted_text, o ("MISSING", "") si no.
    - Primero chequea en la raíz metadata['formatted'] / ['formatted_text'] y normaliza listas.
    - Luego búsqueda recursiva en toda la metadata.
    """
    if not md or not isinstance(md, dict):
        return ("MISSING", "")

    # Chequeo rápido en raíz
    for key in ("formatted", "formatted_text"):
        if key in md:
            v = md.get(key)
            s = _as_str_from_value(v)
            if s:
                return (key, _truncate(s, truncate_chars))
            # si es anidado, buscar recursivamente
            if isinstance(v, (dict, list, tuple)):
                res = _find_formatted_recursive_with_path(v, key)
                if res:
                    p, val = res
                    return (p, _truncate(val, truncate_chars))

    # búsqueda recursiva completa
    res = _find_formatted_recursive_with_path(md, "")
    if res:
        p, val = res
        return (p, _truncate(val, truncate_chars))
    return ("MISSING", "")

# ---------------- Pinecone REST helpers ----------------
def pinecone_query(endpoint_base: str, api_key: str, vector, top_k: int = 100, namespace: Optional[str] = None, timeout: int = 60):
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone query.")
    url = endpoint_base.rstrip("/") + "/query"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"vector": vector, "topK": top_k, "includeMetadata": True}
    if namespace:
        payload["namespace"] = namespace
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def pinecone_fetch_by_id(endpoint_base: str, api_key: str, ids: List[str], namespace: Optional[str] = None, timeout: int = 30):
    """Fetch por id(s) usando /vectors/fetch"""
    if not endpoint_base or not api_key:
        raise RuntimeError("PINECONE_API_ENDPOINT and PINECONE_API_KEY are required for pinecone fetch.")
    url = endpoint_base.rstrip("/") + "/vectors/fetch"
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}
    payload = {"ids": ids}
    if namespace:
        payload["namespace"] = namespace
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------------- CLI & main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Query Pinecone and print top similar vectors (score - formatted).")
    p.add_argument("--vector", default=DEFAULT_VECTOR_FILE, help="Archivo JSON con el vector (array floats).")
    p.add_argument("--topk", type=int, default=50, help="topK pedido al servidor (default 50).")
    p.add_argument("--return", dest="ret", type=int, default=20, help="Cuántos mostrar (default 20).")
    p.add_argument("--csv", help="Guardar resultados en CSV (rank,id,score,path,text).")
    p.add_argument("--timeout", type=int, default=60, help="Timeout por request (default 60s).")
    p.add_argument("--tries", type=int, default=3, help="Reintentos ante timeout (default 3).")
    p.add_argument("--namespace", help="Namespace de Pinecone (si no, usa env PINECONE_NAMESPACE).")
    p.add_argument("--truncate", type=int, default=1000, help="Truncar texto a N chars (0 = no truncar).")
    p.add_argument("--show-id", action="store_true", help="Mostrar id entre paréntesis junto al texto.")
    p.add_argument("--debug", action="store_true", help="Imprime metadata cruda y path de cada match.")
    p.add_argument("--skip-empty", action="store_true", help="Saltar matches sin formatted (para asegurar N filas con texto).")
    p.add_argument("--fetch-id", nargs='+', help="No hace query: fetch por id(s) y muestra metadata (útil para inspeccionar un id específico).")
    return p.parse_args()

def main():
    args = parse_args()

    endpoint = os.getenv("PINECONE_API_ENDPOINT")
    api_key = os.getenv("PINECONE_API_KEY")
    env_ns = os.getenv(DEFAULT_NAMESPACE_ENV)
    namespace = args.namespace or env_ns

    if not endpoint or not api_key:
        print("ERROR: exportá PINECONE_API_ENDPOINT y PINECONE_API_KEY en tu shell", file=sys.stderr)
        sys.exit(2)

    # Si piden fetch por id(s), usar fetch y devolver metadata
    if args.fetch_id:
        try:
            resp = pinecone_fetch_by_id(endpoint, api_key, ids=args.fetch_id, namespace=namespace, timeout=args.timeout)
        except Exception as e:
            print("ERROR fetching ids:", e, file=sys.stderr)
            sys.exit(3)
        vectors = resp.get("vectors", {}) or {}
        for vid, vobj in vectors.items():
            md = vobj.get("metadata", {}) or {}
            path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
            print(f"id={vid} -> path={path} -> text='{text}'")
            if args.debug:
                print("METADATA:", json.dumps(md, ensure_ascii=False, indent=2))
        return

    # Leer vector de archivo
    if not os.path.exists(args.vector):
        print(f"ERROR: vector file not found: {args.vector}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(args.vector, "r", encoding="utf-8") as f:
            vector = json.load(f)
    except Exception as e:
        print("ERROR reading vector file:", e, file=sys.stderr)
        sys.exit(1)

    # Query con reintentos ante timeout
    attempt = 0
    last_exc = None
    resp = None
    while attempt < args.tries:
        attempt += 1
        try:
            resp = pinecone_query(endpoint, api_key, vector, top_k=args.topk, namespace=namespace, timeout=args.timeout)
            break
        except requests.exceptions.ReadTimeout as e:
            print(f"[Attempt {attempt}] ReadTimeout (timeout={args.timeout}s). Retrying...", file=sys.stderr)
            last_exc = e
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print("Request error when calling Pinecone:", e, file=sys.stderr)
            last_exc = e
            break
    else:
        print("All attempts failed. Last exception:", last_exc, file=sys.stderr)
        sys.exit(3)

    matches = resp.get("matches", []) or []
    if not matches:
        print("No matches returned by Pinecone (0).", file=sys.stderr)
        sys.exit(0)

    # ordenar candidatos por score desc
    candidates = sorted(matches, key=lambda m: m.get("score", 0), reverse=True)

    # recolectar resultados; si --skip-empty True, continuar hasta obtener args.ret con texto o agotar candidates/topk
    results = []
    processed = 0
    for m in candidates:
        processed += 1
        md = m.get("metadata", {}) or {}
        path, text = pick_formatted_from_metadata_with_path(md, truncate_chars=args.truncate)
        results.append({
            "id": m.get("id", ""),
            "score": m.get("score", 0) or 0,
            "path": path,
            "text": text,
            "raw_metadata": md
        })

        if not args.skip_empty and len(results) >= args.ret:
            break

        if args.skip_empty:
            nonempty = [r for r in results if r["text"]]
            if len(nonempty) >= args.ret:
                results = nonempty[:args.ret]
                break

        # límite práctico: no procesar más candidatos que topk pedidos
        if processed >= args.topk:
            break

    if not args.skip_empty:
        results = results[:args.ret]

    # imprimir
    for idx, r in enumerate(results, start=1):
        if args.show_id:
            print(f"{idx}. {r['score']:.4f} - {r['text']} (id={r['id']})  path={r['path']}")
        else:
            print(f"{idx}. {r['score']:.4f} - {r['text']}  path={r['path']}")
        if args.debug:
            print("   RAW_METADATA:", json.dumps(r["raw_metadata"], ensure_ascii=False))
            print()

    # CSV
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["rank", "id", "score", "path", "text"])
                for i, r in enumerate(results, start=1):
                    writer.writerow([i, r["id"], f"{r['score']:.6f}", r["path"], r["text"]])
            print(f"Saved results to CSV: {args.csv}", file=sys.stderr)
        except Exception as e:
            print("ERROR writing CSV:", e, file=sys.stderr)

if __name__ == "__main__":
    main()
'''
