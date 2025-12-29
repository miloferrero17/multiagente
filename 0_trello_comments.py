#!/usr/bin/env python3
"""
python3 0_trello_comments.py
Use Trello API to fetch comments from a specified list within a date range.
All the variables are set via environment variables.
Output is saved to data/comments.json by default, or to a specified file if provided.
"""
from __future__ import annotations
import os
import sys
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

TRELLO_API_BASE = "https://api.trello.com/1"
KEY = os.getenv("TRELLO_KEY")
TOKEN = os.getenv("TRELLO_TOKEN")
ENV_LIST_ID = os.getenv("TRELLO_LIST_ID")
ENV_DATE_FROM = os.getenv("TRELLO_DATE_FROM")
ENV_DATE_TO = os.getenv("TRELLO_DATE_TO")

if not KEY or not TOKEN:
    raise SystemExit("Faltan las variables de entorno TRELLO_KEY y/o TRELLO_TOKEN. Cargalas en .env")

MAX_RETRIES = 3
BACKOFF_BASE = 0.5  # segundos


def parse_dt(s: Optional[str], is_start: bool) -> Optional[datetime]:
    """
    Parse a date string to a timezone-aware UTC datetime.
    - accepts YYYY-MM-DD (interpreted as start or end of day depending on is_start)
    - accepts ISO-8601 (with or without 'Z')
    is_start: if True and only date provided, returns YYYY-MM-DDT00:00:00Z
              if False and only date provided, returns YYYY-MM-DDT23:59:59.999999Z
    """
    if not s:
        return None
    s = s.strip()
    try:
        # handle trailing Z (UTC)
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s2)
            return dt.astimezone(timezone.utc)
        # try full ISO (may include offset)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # naive datetime (has date and time but no tz) -> assume UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        # try date-only YYYY-MM-DD
        try:
            dt_date = datetime.strptime(s, "%Y-%m-%d").date()
            if is_start:
                return datetime(dt_date.year, dt_date.month, dt_date.day, 0, 0, 0, tzinfo=timezone.utc)
            else:
                # end of day
                return datetime(dt_date.year, dt_date.month, dt_date.day, 23, 59, 59, 999999, tzinfo=timezone.utc)
        except Exception:
            raise ValueError(f"Formato de fecha no reconocido: {s}. Usa YYYY-MM-DD o ISO-8601.")


def _trello_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    """Wrapper GET con reintento y manejo básico de rate limit (429)."""
    if params is None:
        params = {}
    params.update({"key": KEY, "token": TOKEN})
    url = f"{TRELLO_API_BASE}{path}"
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(wait)
            continue
        if not r.ok:
            raise RuntimeError(f"Trello API error {r.status_code} for {url}\n{r.text[:1000]}")
        return r.json()
    raise RuntimeError("Máximo reintentos alcanzado para " + url)


def fetch_cards_from_list(list_id: str) -> List[Dict[str, Any]]:
    """Trae las cards de una lista. Solicitamos id, shortLink, name y dateLastActivity."""
    params = {"fields": "id,shortLink,name,shortUrl,dateLastActivity"}
    return _trello_get(f"/lists/{list_id}/cards", params=params)


def fetch_comments_for_card(card_id: str) -> List[Dict[str, Any]]:
    """Trae acciones del tipo commentCard de una card (comentarios)."""
    params = {
        "filter": "commentCard",
        "limit": 1000,
    }
    return _trello_get(f"/cards/{card_id}/actions", params=params)


def comments_from_list(list_id: str, date_from: Optional[datetime], date_to: Optional[datetime]) -> List[Dict[str, Any]]:
    """
    Dado list_id y optional date_from/date_to devuelve la lista de comentarios filtrados.
    """
    cards = fetch_cards_from_list(list_id)
    result: List[Dict[str, Any]] = []

    for idx, c in enumerate(cards, start=1):
        card_id = c.get("id")
        short_link = c.get("shortLink") or c.get("id")
        date_last_activity_raw = c.get("dateLastActivity")
        # if card has dateLastActivity and we have date_from, skip cards that were inactive before date_from
        if date_from and date_last_activity_raw:
            try:
                dlast = parse_dt(date_last_activity_raw, is_start=True)
                if dlast and dlast < date_from:
                    # skip this card entirely (no relevant comments)
                    continue
            except Exception:
                # if parse fails, don't skip; fallthrough to fetch comments
                pass

        if not card_id:
            continue

        actions = fetch_comments_for_card(card_id)

        for a in actions:
            created_raw = a.get("date")
            try:
                created_dt = parse_dt(created_raw, is_start=True)
            except Exception:
                # si no se puede parsear, incluimos (más seguro) — o podríamos saltarlo
                created_dt = None

            # filtro por rango de fechas
            if date_from and created_dt and created_dt < date_from:
                continue
            if date_to and created_dt and created_dt > date_to:
                continue

            data = a.get("data", {}) or {}
            member_creator = a.get("memberCreator") or {}
            member_id = member_creator.get("id") or a.get("idMemberCreator")
            member_fullname = member_creator.get("fullName") or member_creator.get("username") or ""

            result.append({
                "commentId": a.get("id"),
                "cardId": card_id,
                "shortLink": short_link,
                "text": data.get("text"),
                "createdAt": a.get("date"),
                "memberId": member_id,
                "memberFullName": member_fullname,
            })

        # pequeña pausa para no saturar
        if idx % 20 == 0:
            time.sleep(0.3)

    return result


def parse_cli_args(argv: List[str]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    CLI args:
       argv[1] = [LIST_ID] (optional)
       argv[2] = [OUT_JSON] (optional)
       argv[3] = [DATE_FROM] (optional)
       argv[4] = [DATE_TO] (optional)
    Returns tuple (list_id, out_path, date_from_str, date_to_str)
    """
    list_id = argv[1] if len(argv) >= 2 else ENV_LIST_ID
    out_path = argv[2] if len(argv) >= 3 else None
    date_from_str = argv[3] if len(argv) >= 4 else ENV_DATE_FROM
    date_to_str = argv[4] if len(argv) >= 5 else ENV_DATE_TO
    return list_id, out_path, date_from_str, date_to_str


def main(argv: List[str]) -> int:
    list_id, out_path, date_from_str, date_to_str = parse_cli_args(argv)
    if not list_id:
        print("Uso: python3 services/trello_comments.py <LIST_ID> [OUT_JSON] [DATE_FROM] [DATE_TO]  OR configure TRELLO_LIST_ID in .env")
        return 2

    try:
        date_from = parse_dt(date_from_str, is_start=True) if date_from_str else None
        date_to = parse_dt(date_to_str, is_start=False) if date_to_str else None
    except Exception as e:
        print("Error al parsear fechas:", str(e), file=sys.stderr)
        return 2

    try:
        comments = comments_from_list(list_id, date_from, date_to)
    except Exception as e:
        print("Error al obtener comentarios:", str(e), file=sys.stderr)
        return 1

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"Guardado {len(comments)} comentarios en {out_path}")
    else:
        print(json.dumps(comments, ensure_ascii=False, indent=2))

    return 0

def main(argv: List[str]) -> int:
    """
    CLI: python3 0_trello_comments.py [LIST_ID] [OUT_JSON] [DATE_FROM] [DATE_TO]
    If LIST_ID is not provided, uses TRELLO_LIST_ID from .env.
    If OUT_JSON is not provided, defaults to data/comments.json.
    """
    list_id, out_path, date_from_str, date_to_str = parse_cli_args(argv)

    # If list_id not provided via CLI, try ENV_LIST_ID (already handled in parse_cli_args).
    if not list_id:
        print("No LIST_ID provided and TRELLO_LIST_ID not set in .env. Aborting.")
        return 2

    # Default output file if none provided
    if not out_path:
        out_path = "data/comments.json"
        from pathlib import Path
        Path("data").mkdir(parents=True, exist_ok=True)

    # Parse date range (if provided)
    try:
        date_from = parse_dt(date_from_str, is_start=True) if date_from_str else None
        date_to = parse_dt(date_to_str, is_start=False) if date_to_str else None
    except Exception as e:
        print("Error al parsear fechas:", str(e), file=sys.stderr)
        return 2

    try:
        comments = comments_from_list(list_id, date_from, date_to)
    except Exception as e:
        print("Error al obtener comentarios:", str(e), file=sys.stderr)
        return 1

    # Save to out_path (always a path at this point)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        print(f"Guardado {len(comments)} comentarios en {out_path}")
    except Exception as e:
        print("Error al guardar archivo:", str(e), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))



'''
import os
import requests
from dotenv import load_dotenv

load_dotenv()

r = requests.get(
    "https://api.trello.com/1/members/me",
    params={
        "key": os.getenv("TRELLO_KEY"),
        "token": os.getenv("TRELLO_TOKEN"),
    },
    timeout=10,
)

print("status:", r.status_code)
print(r.text[:1000])
'''