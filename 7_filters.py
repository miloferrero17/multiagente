#!/usr/bin/env python3
"""
python3 7_filters.py

Scan all vectors in a Pinecone namespace and get DISTINCT values of one or more.
Examples:
  python3 7_filters.py --namespace trello --field Team --field team
  python3 7_filters.py --namespace trello --field fecha
  python3 7_filters.py --namespace trello --field Team --csv out.csv
  python3 7_filters.py --env-file /ruta/a/.env --namespace trello --field Team
Output: printed to console, optional CSV file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Iterable
from collections import Counter
import os
import sys
import csv
import argparse
import requests
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore


DEFAULT_NAMESPACE = "trello"
NAMESPACE_ENV_VAR = "PINECONE_NAMESPACE"
ENDPOINT_ENV_VAR = "PINECONE_API_ENDPOINT"
API_KEY_ENV_VAR = "PINECONE_API_KEY"


# ----------------------- helpers -----------------------

def _raise_bad_response(r: requests.Response, label: str):
    ct = (r.headers.get("content-type") or "").lower()
    snippet = (r.text or "")[:1000]
    raise RuntimeError(
        f"[{label}] Bad / non-JSON response\n"
        f"URL: {r.url}\n"
        f"Status: {r.status_code}\n"
        f"Content-Type: {ct}\n"
        f"Body (first 1000 chars):\n{snippet}\n"
    )


def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _load_env_file(env_file: Optional[str]) -> None:
    if load_dotenv is None:
        print(
            "ERROR: falta dependencia 'python-dotenv'. Instalá con:\n"
            "  pip install python-dotenv\n",
            file=sys.stderr,
        )
        return

    if env_file:
        path = Path(env_file).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent / ".env"

    if path.exists():
        load_dotenv(dotenv_path=path, override=False)
    else:
        if env_file:
            print(f"WARNING: no existe el .env indicado: {path}", file=sys.stderr)


def _get_nested(d: Dict[str, Any], dotted_key: str) -> Any:
    """
    Permite pedir keys anidadas tipo "a.b.c".
    Si en algún punto no existe, devuelve None.
    """
    cur: Any = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        if part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _normalize_value_to_strings(v: Any) -> List[str]:
    """
    Convierte un valor de metadata a lista de strings "contables".
    - str -> [str]
    - list/tuple/set -> flatten a strings (si items son scalars) o JSON (si no)
    - int/float/bool -> [str(v)]
    - dict/otros -> [json.dumps(v)]
    - None -> []
    """
    if v is None:
        return []

    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []

    if isinstance(v, (int, float, bool)):
        return [str(v)]

    if isinstance(v, (list, tuple, set)):
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, (int, float, bool)):
                out.append(str(item))
            else:
                out.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return out

    if isinstance(v, dict):
        return [json.dumps(v, ensure_ascii=False, sort_keys=True)]

    # fallback
    return [str(v)]


def _flatten_fields(fields: Optional[List[str]]) -> List[str]:
    """
    Permite:
      --field Team --field team
    o:
      --field Team team fecha
    """
    if not fields:
        return []
    out: List[str] = []
    for f in fields:
        if not f:
            continue
        # argparse con nargs="+" ya arma listas, pero por si pasan "Team,team"
        parts = [p.strip() for p in f.split(",") if p.strip()]
        out.extend(parts)
    # dedupe manteniendo orden
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# ----------------------- Pinecone REST helpers -----------------------

def pinecone_list_ids(
    endpoint_base: str,
    api_key: str,
    namespace: Optional[str],
    limit: int = 100,
    pagination_token: Optional[str] = None,
    prefix: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    if not endpoint_base or not api_key:
        raise RuntimeError(f"{ENDPOINT_ENV_VAR} and {API_KEY_ENV_VAR} are required.")

    url = endpoint_base.rstrip("/") + "/vectors/list"
    headers = {"Api-Key": api_key, "Accept": "application/json"}

    params: Dict[str, Any] = {"limit": limit}
    if namespace:
        params["namespace"] = namespace
    if pagination_token:
        params["paginationToken"] = pagination_token
    if prefix:
        params["prefix"] = prefix

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if not r.ok:
        _raise_bad_response(r, "LIST (HTTP error)")
    try:
        return r.json()
    except ValueError:
        _raise_bad_response(r, "LIST (JSON decode)")


def pinecone_fetch_by_id_get(
    endpoint_base: str,
    api_key: str,
    ids: List[str],
    namespace: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    if not endpoint_base or not api_key:
        raise RuntimeError(f"{ENDPOINT_ENV_VAR} and {API_KEY_ENV_VAR} are required.")

    url = endpoint_base.rstrip("/") + "/vectors/fetch"
    headers = {"Api-Key": api_key, "Accept": "application/json"}

    params: List[Tuple[str, str]] = [("ids", vid) for vid in ids]
    if namespace:
        params.append(("namespace", namespace))

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if not r.ok:
        _raise_bad_response(r, "FETCH-GET (HTTP error)")
    try:
        return r.json()
    except ValueError:
        _raise_bad_response(r, "FETCH-GET (JSON decode)")


# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Distinct values from Pinecone metadata fields.")
    p.add_argument("--env-file", help="Ruta a .env (default: .env al lado del script)")
    p.add_argument("--namespace", help=f"Namespace (default: env {NAMESPACE_ENV_VAR} o '{DEFAULT_NAMESPACE}')")
    p.add_argument("--field", nargs="+", action="append",
                   help="Campo(s) de metadata a contar (ej: Team team fecha source o nested: a.b.c). "
                        "Podés repetir --field o pasar varios juntos.")
    p.add_argument("--limit", type=int, default=100, help="IDs por página en /vectors/list (default 100)")
    p.add_argument("--prefix", help="Opcional: filtrar IDs por prefix en /vectors/list")
    p.add_argument("--max-ids", type=int, default=0, help="0 = todos. Si >0 corta al llegar a N IDs (debug).")
    p.add_argument("--fetch-batch", type=int, default=25, help="IDs por batch en /vectors/fetch (default 25)")
    p.add_argument("--timeout", type=int, default=60, help="Timeout por request (default 60)")
    p.add_argument("--csv", help="Guardar resultados en CSV (columns: field,value,count)")
    p.add_argument("--show-samples", type=int, default=0, help="Muestra N IDs de ejemplo (debug)")
    return p.parse_args()


# ----------------------- main -----------------------

def main():
    args = parse_args()

    # 0) cargar .env
    _load_env_file(args.env_file)

    endpoint = os.getenv(ENDPOINT_ENV_VAR)
    api_key = os.getenv(API_KEY_ENV_VAR)
    namespace = args.namespace or os.getenv(NAMESPACE_ENV_VAR) or DEFAULT_NAMESPACE

    # fields vienen como list[list[str]] por action="append" + nargs="+"
    raw_fields: List[str] = []
    if args.field:
        for group in args.field:
            raw_fields.extend(group)
    fields = _flatten_fields(raw_fields)

    if not fields:
        print(
            "ERROR: tenés que indicar al menos un campo de metadata con --field.\n"
            "Ejemplos:\n"
            "  python3 7_filters.py --namespace trello --field Team\n"
            "  python3 7_filters.py --namespace trello --field Team team\n"
            "  python3 7_filters.py --namespace trello --field fecha\n",
            file=sys.stderr,
        )
        sys.exit(2)

    if not endpoint or not api_key:
        print(
            "ERROR: faltan credenciales.\n"
            f"- Definí {ENDPOINT_ENV_VAR} y {API_KEY_ENV_VAR} en tu .env (o env vars).\n"
            "Ejemplo .env:\n"
            "  PINECONE_API_ENDPOINT=https://TU_HOST_DEL_INDEX\n"
            "  PINECONE_API_KEY=TU_API_KEY\n"
            "  PINECONE_NAMESPACE=trello\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # 1) listar ids (paginado)
    all_ids: List[str] = []
    token: Optional[str] = None
    page = 0

    while True:
        page += 1
        resp = pinecone_list_ids(
            endpoint_base=endpoint,
            api_key=api_key,
            namespace=namespace,
            limit=args.limit,
            pagination_token=token,
            prefix=args.prefix,
            timeout=args.timeout,
        )

        vectors = resp.get("vectors", []) or []
        page_ids = [
            v.get("id")
            for v in vectors
            if isinstance(v, dict) and v.get("id")
        ]
        all_ids.extend(page_ids)

        if args.show_samples and page == 1:
            print(f"Sample IDs (first page, up to {args.show_samples}):")
            for s in page_ids[:args.show_samples]:
                print("  ", s)
            print()

        if args.max_ids and len(all_ids) >= args.max_ids:
            all_ids = all_ids[:args.max_ids]
            break

        token = (resp.get("pagination") or {}).get("next")
        if not token:
            break

    if not all_ids:
        print("No se encontraron IDs. Posibles causas:", file=sys.stderr)
        print("- namespace vacío o incorrecto", file=sys.stderr)
        print("- /vectors/list no está habilitado en este índice", file=sys.stderr)
        sys.exit(0)

    print(f"Total IDs listados: {len(all_ids)}", file=sys.stderr)
    print(f"Fields a contar: {', '.join(fields)}", file=sys.stderr)

    # 2) fetch metadata y contar
    # Guardamos counters por field, así podés pedir múltiples campos
    counters: Dict[str, Counter] = {f: Counter() for f in fields}

    for i, batch in enumerate(_chunks(all_ids, args.fetch_batch), start=1):
        fresp = pinecone_fetch_by_id_get(
            endpoint_base=endpoint,
            api_key=api_key,
            ids=batch,
            namespace=namespace,
            timeout=args.timeout,
        )

        vectors = fresp.get("vectors", {}) or {}
        for _vid, vobj in vectors.items():
            md = (vobj or {}).get("metadata", {}) or {}
            if not isinstance(md, dict):
                continue

            for f in fields:
                val = _get_nested(md, f)
                for sval in _normalize_value_to_strings(val):
                    counters[f][sval] += 1

        if i % 10 == 0:
            print(f"Fetched {min(i*args.fetch_batch, len(all_ids))} ids...", file=sys.stderr)

    # 3) imprimir resultados
    any_found = False
    for f in fields:
        c = counters[f]
        if not c:
            print(f"\nField '{f}': NO encontré valores en metadata.")
            continue

        any_found = True
        print(f"\nField '{f}' - Distinct values: {len(c)}")
        for value, cnt in c.most_common():
            print(f"{cnt:>8}  {value}")

    if not any_found:
        print("\nNo encontré ninguno de los campos pedidos en la metadata de los vectores.", file=sys.stderr)
        sys.exit(0)

    # 4) guardar CSV si corresponde
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["field", "value", "count"])
                for f in fields:
                    for value, cnt in counters[f].most_common():
                        w.writerow([f, value, cnt])
            print(f"\nSaved CSV: {args.csv}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR writing CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


'''
#!/usr/bin/env python3
"""
7_filters.py

Objetivo:
- Recorrer TODOS los vectores de un namespace y obtener los valores DISTINCT del metadata Team (y team),
  con conteos.
- Funciona aunque el endpoint NO soporte POST /vectors/fetch en JSON (en tu caso devuelve application/grpc).
- Usa:
  1) GET /vectors/list   (paginado)  -> obtiene IDs
  2) GET /vectors/fetch  (ids repetidos en querystring) -> obtiene metadata
     (NO usa coma-separado, porque Pinecone lo interpreta como un solo ID y falla)

Requiere en .env o env:
  PINECONE_API_ENDPOINT
  PINECONE_API_KEY
Opcional:
  PINECONE_NAMESPACE

Uso:
  python3 7_filters.py --namespace trello
  python3 7_filters.py --namespace trello --csv teams.csv
  python3 7_filters.py --namespace trello --fetch-batch 25 --limit 100
  python3 7_filters.py --namespace trello --max-ids 1000   # debug
  python3 7_filters.py --env-file /ruta/a/.env --namespace trello

Notas:
- /vectors/list suele estar disponible en índices serverless. Si tu índice no lo soporta,
  el script te lo va a mostrar claramente (HTTP error + body).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import os
import sys
import csv
import argparse
import requests
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore


DEFAULT_NAMESPACE = "trello"
NAMESPACE_ENV_VAR = "PINECONE_NAMESPACE"
ENDPOINT_ENV_VAR = "PINECONE_API_ENDPOINT"
API_KEY_ENV_VAR = "PINECONE_API_KEY"


# ----------------------- helpers -----------------------

def _as_str_list(v: Any) -> List[str]:
    """Normaliza Team/team: puede ser str o list[str]. Devuelve lista de strings no vacíos."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, (list, tuple)):
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    return []


def _raise_bad_response(r: requests.Response, label: str):
    ct = (r.headers.get("content-type") or "").lower()
    snippet = (r.text or "")[:1000]
    raise RuntimeError(
        f"[{label}] Bad / non-JSON response\n"
        f"URL: {r.url}\n"
        f"Status: {r.status_code}\n"
        f"Content-Type: {ct}\n"
        f"Body (first 1000 chars):\n{snippet}\n"
    )


def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _load_env_file(env_file: Optional[str]) -> None:
    """
    Carga variables desde un .env.
    - Si env_file es None, intenta usar ".env" al lado de este script.
    - Si python-dotenv no está instalado, muestra un mensaje claro.
    """
    if load_dotenv is None:
        print(
            "ERROR: falta dependencia 'python-dotenv'. Instalá con:\n"
            "  pip install python-dotenv\n",
            file=sys.stderr,
        )
        return

    if env_file:
        path = Path(env_file).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent / ".env"

    if path.exists():
        load_dotenv(dotenv_path=path, override=False)
    else:
        # No es fatal: puede que el usuario use env vars exportadas.
        # Igual avisamos para que sea obvio.
        if env_file:
            print(f"WARNING: no existe el .env indicado: {path}", file=sys.stderr)
        else:
            # Silencioso si no existe, pero podés cambiarlo a WARNING si querés.
            pass


# ----------------------- Pinecone REST helpers -----------------------

def pinecone_list_ids(
    endpoint_base: str,
    api_key: str,
    namespace: Optional[str],
    limit: int = 100,
    pagination_token: Optional[str] = None,
    prefix: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    GET /vectors/list
    Devuelve algo como:
      {
        "vectors":[{"id":"..."}, ...],
        "pagination":{"next":"..."}
      }
    """
    if not endpoint_base or not api_key:
        raise RuntimeError(f"{ENDPOINT_ENV_VAR} and {API_KEY_ENV_VAR} are required.")

    url = endpoint_base.rstrip("/") + "/vectors/list"
    headers = {"Api-Key": api_key, "Accept": "application/json"}

    params: Dict[str, Any] = {"limit": limit}
    if namespace:
        params["namespace"] = namespace
    if pagination_token:
        params["paginationToken"] = pagination_token
    if prefix:
        params["prefix"] = prefix

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if not r.ok:
        _raise_bad_response(r, "LIST (HTTP error)")
    try:
        return r.json()
    except ValueError:
        _raise_bad_response(r, "LIST (JSON decode)")


def pinecone_fetch_by_id_get(
    endpoint_base: str,
    api_key: str,
    ids: List[str],
    namespace: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    GET /vectors/fetch?ids=<id>&ids=<id>...
    IMPORTANTÍSIMO: NO usar coma-separado (ids=id1,id2,...) porque Pinecone lo interpreta como 1 solo ID.
    """
    if not endpoint_base or not api_key:
        raise RuntimeError(f"{ENDPOINT_ENV_VAR} and {API_KEY_ENV_VAR} are required.")

    url = endpoint_base.rstrip("/") + "/vectors/fetch"
    headers = {"Api-Key": api_key, "Accept": "application/json"}

    params: List[Tuple[str, str]] = [("ids", vid) for vid in ids]
    if namespace:
        params.append(("namespace", namespace))

    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if not r.ok:
        _raise_bad_response(r, "FETCH-GET (HTTP error)")
    try:
        return r.json()
    except ValueError:
        _raise_bad_response(r, "FETCH-GET (JSON decode)")


# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Distinct Team values from Pinecone metadata.")
    p.add_argument("--env-file", help="Ruta a .env (default: .env al lado del script)")
    p.add_argument("--namespace", help=f"Namespace (default: env {NAMESPACE_ENV_VAR} o '{DEFAULT_NAMESPACE}')")
    p.add_argument("--limit", type=int, default=100, help="IDs por página en /vectors/list (default 100)")
    p.add_argument("--prefix", help="Opcional: filtrar IDs por prefix en /vectors/list")
    p.add_argument("--max-ids", type=int, default=0, help="0 = todos. Si >0 corta al llegar a N IDs (debug).")
    p.add_argument("--fetch-batch", type=int, default=25, help="IDs por batch en /vectors/fetch (default 25)")
    p.add_argument("--timeout", type=int, default=60, help="Timeout por request (default 60)")
    p.add_argument("--csv", help="Guardar resultados (team,count) en CSV")
    p.add_argument("--show-samples", type=int, default=0, help="Muestra N IDs de ejemplo (debug)")
    return p.parse_args()


# ----------------------- main -----------------------

def main():
    args = parse_args()

    # 0) cargar .env (sin depender de export en la terminal)
    _load_env_file(args.env_file)

    endpoint = os.getenv(ENDPOINT_ENV_VAR)
    api_key = os.getenv(API_KEY_ENV_VAR)
    namespace = args.namespace or os.getenv(NAMESPACE_ENV_VAR) or DEFAULT_NAMESPACE

    if not endpoint or not api_key:
        print(
            "ERROR: faltan credenciales.\n"
            f"- Definí {ENDPOINT_ENV_VAR} y {API_KEY_ENV_VAR} en tu .env (o env vars).\n"
            "Ejemplo .env:\n"
            "  PINECONE_API_ENDPOINT=https://TU_HOST_DEL_INDEX\n"
            "  PINECONE_API_KEY=TU_API_KEY\n"
            "  PINECONE_NAMESPACE=trello\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # 1) listar todos los ids (paginado)
    all_ids: List[str] = []
    token: Optional[str] = None
    page = 0

    while True:
        page += 1
        resp = pinecone_list_ids(
            endpoint_base=endpoint,
            api_key=api_key,
            namespace=namespace,
            limit=args.limit,
            pagination_token=token,
            prefix=args.prefix,
            timeout=args.timeout,
        )

        vectors = resp.get("vectors", []) or []
        page_ids = [
            v.get("id")
            for v in vectors
            if isinstance(v, dict) and v.get("id")
        ]
        all_ids.extend(page_ids)

        if args.show_samples and page == 1:
            print(f"Sample IDs (first page, up to {args.show_samples}):")
            for s in page_ids[:args.show_samples]:
                print("  ", s)
            print()

        if args.max_ids and len(all_ids) >= args.max_ids:
            all_ids = all_ids[:args.max_ids]
            break

        token = (resp.get("pagination") or {}).get("next")
        if not token:
            break

    if not all_ids:
        print("No se encontraron IDs. Posibles causas:", file=sys.stderr)
        print("- namespace vacío o incorrecto", file=sys.stderr)
        print("- /vectors/list no está habilitado en este índice", file=sys.stderr)
        sys.exit(0)

    print(f"Total IDs listados: {len(all_ids)}", file=sys.stderr)

    # 2) fetch metadata por batches y contar Team/team
    counter = Counter()

    for i, batch in enumerate(_chunks(all_ids, args.fetch_batch), start=1):
        fresp = pinecone_fetch_by_id_get(
            endpoint_base=endpoint,
            api_key=api_key,
            ids=batch,
            namespace=namespace,
            timeout=args.timeout,
        )

        vectors = fresp.get("vectors", {}) or {}
        # vectors suele ser dict: {id: {"id":..., "metadata":{...}, ...}, ...}
        for _vid, vobj in vectors.items():
            md = (vobj or {}).get("metadata", {}) or {}
            teams = _as_str_list(md.get("Team")) + _as_str_list(md.get("team"))
            for t in teams:
                counter[t] += 1

        if i % 10 == 0:
            print(f"Fetched {i*args.fetch_batch} ids...", file=sys.stderr)

    # 3) imprimir resultados
    if not counter:
        print("No encontré metadata Team/team en los vectores fetchados.", file=sys.stderr)
        sys.exit(0)

    print(f"\nDistinct Team values: {len(counter)}")
    for team, cnt in counter.most_common():
        print(f"{cnt:>8}  {team}")

    # 4) guardar CSV si corresponde
    if args.csv:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["team", "count"])
                for team, cnt in counter.most_common():
                    w.writerow([team, cnt])
            print(f"\nSaved CSV: {args.csv}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR writing CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
'''