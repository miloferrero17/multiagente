#!/usr/bin/env python3
"""
python 8_filter_to_csv.py
Filter Pinecone vectors by metadata field and export to CSV using pagination.
Example usages:
python3 8_filter_to_csv.py \
  --namespace trello \
  --team "mejora continua y scheme enablers" \
  --output 1.csv

All the variables are set via environment variables.
Output: CSV file with filtered metadata.
"""

import os
import csv
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Cargar variables de entorno desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PINECONE_API_ENDPOINT = os.getenv("PINECONE_API_ENDPOINT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def pinecone_list_ids_page(endpoint: str, api_key: str, namespace: str, token: Optional[str] = None, limit: int = 100):
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

def main():
    parser = argparse.ArgumentParser(description="Exportar metadata de Pinecone a CSV con paginación.")
    parser.add_argument("--namespace", default="trello", help="Namespace de Pinecone.")
    parser.add_argument("--output", default="metadata_export.csv", help="Archivo de salida.")
    parser.add_argument("--team", help="Filtrar por el campo 'Team' (case-insensitive).")
    parser.add_argument("--limit_total", type=int, default=0, help="0 = todos. Límite máximo de IDs a procesar.")
    args = parser.parse_args()

    if not PINECONE_API_ENDPOINT or not PINECONE_API_KEY:
        print("ERROR: Falta PINECONE_API_ENDPOINT o PINECONE_API_KEY en el .env")
        sys.exit(1)

    print(f"--- Iniciando exportación completa ---")
    print(f"Namespace: {args.namespace}")
    if args.team:
        print(f"Filtro: Team == {args.team}")

    # 1. Listar TODOS los IDs usando paginación
    all_ids = []
    next_token = None
    print("Listando IDs desde Pinecone...")
    
    while True:
        resp = pinecone_list_ids_page(PINECONE_API_ENDPOINT, PINECONE_API_KEY, args.namespace, token=next_token)
        vectors = resp.get("vectors", [])
        page_ids = [v['id'] for v in vectors]
        all_ids.extend(page_ids)
        
        print(f"  > IDs encontrados hasta ahora: {len(all_ids)}")
        
        next_token = resp.get("pagination", {}).get("next")
        if not next_token or (args.limit_total > 0 and len(all_ids) >= args.limit_total):
            break

    if not all_ids:
        print("No se encontraron vectores.")
        return

    # 2. Fetch y Filtrado por batches
    all_rows = []
    headers_found = set(["id"])
    batch_size = 25 
    
    print(f"Procesando metadata y aplicando filtros...")
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        try:
            vectors_data = pinecone_fetch(PINECONE_API_ENDPOINT, PINECONE_API_KEY, batch_ids, args.namespace)
            
            for vid, vobj in vectors_data.items():
                metadata = vobj.get("metadata", {}) or {}
                
                # Filtrado por el campo 'Team'
                if args.team:
                    actual_team = metadata.get("Team")
                    if isinstance(actual_team, list):
                        if args.team.lower() not in [str(t).lower() for t in actual_team]:
                            continue
                    elif str(actual_team).lower() != args.team.lower():
                        continue

                row = {"id": vid}
                for k, v in metadata.items():
                    headers_found.add(k)
                    row[k] = ", ".join(v) if isinstance(v, list) else v
                all_rows.append(row)
                
        except Exception as e:
            print(f"Error en bloque {i}: {e}")

    # 3. Guardar CSV
    if not all_rows:
        print("No se encontraron coincidencias para el filtro aplicado.")
        return

    columns = sorted(list(headers_found))
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Éxito: Se filtraron {len(all_rows)} de {len(all_ids)} vectores.")
    print(f"Resultados guardados en '{args.output}'")

if __name__ == "__main__":
    main()