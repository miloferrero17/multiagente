#!/usr/bin/env python3
"""
python3 1_comments_processor.py [INPUT_JSON] [OUTPUT_JSON]

Lee data/comments.json (salida de 0_trello_comments.py),
procesa cada comentario con Claude y el system prompt de aux_1_system_message.txt,
y guarda los bullets estructurados en data/processed_comments.json.

Variables de entorno requeridas:
  ANTHROPIC_API_KEY
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SYSTEM_PROMPT_FILE = Path(__file__).parent / "aux_1_system_message.txt"
DEFAULT_INPUT = Path("data/comments.json")
DEFAULT_OUTPUT = Path("data/processed_comments.json")

# Reintentos ante rate limit
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # segundos


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_FILE.exists():
        raise FileNotFoundError(f"No se encontró {SYSTEM_PROMPT_FILE}")
    return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")


def process_comment(
    client: OpenAI,
    system_prompt: str,
    comment: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Envía un comentario a OpenAI y devuelve el arreglo de bullets.
    """
    user_message = json.dumps(comment, ensure_ascii=False, indent=2)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()

            # Extraer JSON del response (puede venir con markdown)
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

            bullets = json.loads(raw)
            if not isinstance(bullets, list):
                bullets = []
            return bullets

        except RateLimitError as e:
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  [rate limit] esperando {wait:.1f}s (intento {attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"  [warn] no se pudo parsear respuesta: {e}")
            return []

    print(f"  [error] máximo de reintentos alcanzado para commentId={comment.get('commentId')}")
    return []


def main(argv: List[str]) -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: falta OPENAI_API_KEY en .env", file=sys.stderr)
        return 2

    input_path = Path(argv[1]) if len(argv) >= 2 else DEFAULT_INPUT
    output_path = Path(argv[2]) if len(argv) >= 3 else DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"ERROR: no se encontró {input_path}", file=sys.stderr)
        return 1

    try:
        system_prompt = load_system_prompt()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    comments: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"Comentarios a procesar: {len(comments)}")

    client = OpenAI(api_key=api_key)
    all_bullets: List[Dict[str, Any]] = []

    for idx, comment in enumerate(comments, start=1):
        comment_id = comment.get("commentId", "?")
        print(f"[{idx}/{len(comments)}] commentId={comment_id} ...", end=" ", flush=True)

        bullets = process_comment(client, system_prompt, comment)
        all_bullets.extend(bullets)
        print(f"{len(bullets)} bullets")

        # Pausa leve para no saturar la API
        if idx % 10 == 0:
            time.sleep(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_bullets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nTotal bullets: {len(all_bullets)}")
    print(f"Guardado en: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
