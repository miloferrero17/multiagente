"""
python 8_filter_to_json.py
Filter Pinecone vectors by metadata field and export to JSON using pagination.

Example:
python3 8_filter_to_json.py \
  --namespace trello \
  --filter_field Team \
  --filter_value "Relacionamiento con las banderas"

All the variables are set via environment variables.
Output: JSON file with filtered metadata (excluding some keys).
"""

import os
import sys
import re
import json
import argparse
import requests
from pathlib import Path
from typing import List, Optional, Any, Dict

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PINECONE_API_ENDPOINT = os.getenv("PINECONE_API_ENDPOINT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Campos a excluir del output final
EXCLUDE_FIELDS = {"bullet_id", "commentId", "id", "source_file"}


def norm(s) -> str:
    """
    Normaliza texto para comparación case-insensitive:
    - trim
    - casefold (mejor que lower)
    - colapsa espacios internos
    """
    return " ".join(str(s).strip().casefold().split())


def slugify(s: str) -> str:
    """
    Convierte a un nombre seguro para carpeta/archivo:
    - trim + casefold
    - reemplaza espacios por '_'
    - remueve chars raros
    """
    s = norm(s).replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s or "all"


def pinecone_list_ids_page(
    endpoint: str,
    api_key: str,
    namespace: str,
    token: Optional[str] = None,
    limit: int = 100,
):
    """Obtiene una página de IDs usando GET /vectors/list"""
    url = f"{endpoint.rstrip('/')}/vectors/list"
    headers = {"Api-Key": api_key, "Accept": "application/json"}
    params = {"namespace": namespace, "limit": limit}
    if token:
        params["paginationToken"] = token

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    if not resp.ok:
        print(f"[ERROR] Response: {resp.text}")
        resp.raise_for_status()
    return resp.json()


def pinecone_fetch(endpoint: str, api_key: str, ids: List[str], namespace: str):
    """Fetch metadata usando GET con IDs repetidos en query string"""
    url = f"{endpoint.rstrip('/')}/vectors/fetch"
    headers = {"Api-Key": api_key, "Accept": "application/json"}
    params = [("ids", vid) for vid in ids]
    if namespace:
        params.append(("namespace", namespace))

    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("vectors", {})


def metadata_matches(value: Any, wanted_norm: str) -> bool:
    """
    Compara metadata vs wanted de forma case-insensitive y robusta.
    - value puede ser string, number, bool, None, list, etc.
    - wanted_norm ya viene normalizado con norm()
    """
    if value is None:
        return False

    if isinstance(value, list):
        return wanted_norm in {norm(v) for v in value if v is not None}

    return norm(value) == wanted_norm


def clean_meta_for_output(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Elimina campos no deseados del output JSON."""
    return {k: v for k, v in meta.items() if k not in EXCLUDE_FIELDS}


def main():
    parser = argparse.ArgumentParser(
        description="Exportar metadata de Pinecone a JSON con paginación y filtro por cualquier campo."
    )
    parser.add_argument("--namespace", default="trello", help="Namespace de Pinecone.")
    parser.add_argument(
        "--output",
        default="",
        help="Nombre del JSON. Si se omite, se usa <filter_value>.json.",
    )
    parser.add_argument(
        "--filter_field",
        required=True,
        help="Nombre del campo de metadata por el que filtrar (ej: Team, Board, List, etc.).",
    )
    parser.add_argument(
        "--filter_value",
        required=True,
        help="Valor a matchear contra el campo (case-insensitive).",
    )
    parser.add_argument(
        "--limit_total",
        type=int,
        default=0,
        help="0 = todos. Límite máximo de IDs a procesar.",
    )
    args = parser.parse_args()

    if not PINECONE_API_ENDPOINT or not PINECONE_API_KEY:
        print("ERROR: Falta PINECONE_API_ENDPOINT o PINECONE_API_KEY en el .env")
        sys.exit(1)

    filter_field = args.filter_field.strip()
    wanted_norm = norm(args.filter_value)

    print("--- Iniciando exportación completa ---")
    print(f"Namespace: {args.namespace}")
    print(f"Filtro: {filter_field} == {args.filter_value} (case-insensitive)")

    # Directorio: data/output/vectors/<filter_field>
    out_dir = Path("data") / "output" / "vectors" / slugify(filter_field)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Archivo: si no se pasa --output => <filter_value>.json
    default_name = f"{slugify(args.filter_value)}.json"
    output_name = Path(args.output).name if args.output else default_name
    out_file = out_dir / output_name

    # 1) Listar IDs paginando
    all_ids: List[str] = []
    next_token: Optional[str] = None
    print("Listando IDs desde Pinecone...")

    while True:
        resp = pinecone_list_ids_page(
            PINECONE_API_ENDPOINT,
            PINECONE_API_KEY,
            args.namespace,
            token=next_token,
        )
        vectors = resp.get("vectors", [])
        page_ids = [v["id"] for v in vectors]
        all_ids.extend(page_ids)

        print(f"  > IDs encontrados hasta ahora: {len(all_ids)}")

        next_token = resp.get("pagination", {}).get("next")
        if not next_token or (
            args.limit_total > 0 and len(all_ids) >= args.limit_total
        ):
            break

    if args.limit_total > 0:
        all_ids = all_ids[: args.limit_total]

    if not all_ids:
        print("No se encontraron vectores.")
        return

    # 2) Fetch + filtrar metadata
    print("Fetching metadata y aplicando filtro...")
    rows: List[Dict[str, Any]] = []
    batch_size = 100

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        vectors = pinecone_fetch(
            PINECONE_API_ENDPOINT,
            PINECONE_API_KEY,
            batch_ids,
            args.namespace,
        )

        for vid, vdata in vectors.items():
            meta = vdata.get("metadata", {})
            if filter_field not in meta:
                continue

            if metadata_matches(meta.get(filter_field), wanted_norm):
                # Importante: NO agregamos "id" del vector, y además limpiamos campos del metadata
                row = clean_meta_for_output(meta)
                rows.append(row)

        print(f"  > Procesados {min(i + batch_size, len(all_ids))}/{len(all_ids)}")

    if not rows:
        print("No hubo coincidencias con el filtro.")
        return

    # 3) Escribir JSON
    print(f"Escribiendo JSON en: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

    print(f"✔ Exportación finalizada. Objetos escritos: {len(rows)}")


if __name__ == "__main__":
    main()
