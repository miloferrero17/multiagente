#!/usr/bin/env python3
'''
python3 1_comments_procesor.py
Transforms ./data/comments.json using OpenAI in parallel using aux_1_system_message as system message and each comment as a prompt.
All the variables are set via environment variables.
Output: ./data/output/<commentId>.json for each comment.
'''


from pathlib import Path
import json
import re
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

load_dotenv()

from services.openai import ask_openai

INPUT_PATH = Path("./data/comments.json")
OUT_DIR = Path("./data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_FILE = Path("./aux_1_system_message.txt")
if not SYSTEM_FILE.exists():
    raise FileNotFoundError(f"System message file not found: {SYSTEM_FILE}")
SYSTEM_MESSAGE = SYSTEM_FILE.read_text(encoding="utf-8")

def extract_json_array_from_text(txt: str) -> str:
    if not txt:
        return ""
    s = txt.strip()
    if s.startswith("```"):
        s = re.sub(r'^```[^\n]*\n', '', s)
        s = re.sub(r'\n```$', '', s).strip()
    first = s.find('[')
    last = s.rfind(']')
    if first != -1 and last != -1 and last > first:
        return s[first:last+1]
    return s

def call_ai_for_comment(comment, max_retries=3, wait_seconds=1.0):
    user_msg = (
        "Procesa este comentario y RESPONDE SOLO con el arreglo JSON solicitado por la especificación.\n\n"
        "Comentario JSON:\n" + json.dumps(comment, ensure_ascii=False) + "\n\n"
        "Insisto: devuelve ÚNICAMENTE el arreglo JSON (por ejemplo: [] o [{...}, ...]) sin texto adicional."
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_msg}
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp_text = ask_openai(messages, temperature=0.0, model="gpt-4.1")
            candidate_txt = extract_json_array_from_text(resp_text)
            parsed = json.loads(candidate_txt)

            if not isinstance(parsed, list):
                raise ValueError("AI returned JSON but not a list")

            # mínima validación
            for obj in parsed:
                if not isinstance(obj, dict):
                    raise ValueError("List contains non-dict item")
                if "bullet_id" not in obj or "commentId" not in obj:
                    raise ValueError("Missing bullet_id/commentId in one or more items")

            return parsed

        except Exception as e:
            last_err = e
            time.sleep(wait_seconds * attempt)

    print(f"[AI] Failed for comment {comment.get('commentId')}: {last_err}")
    return None

def process_one_comment(comment, idx, total, skip_if_exists=True):
    cid = comment.get("commentId") or f"no-id-{idx}"
    out_path = OUT_DIR / f"{cid}.json"

    if skip_if_exists and out_path.exists():
        return cid, "skipped", out_path, None

    print(f"[AI] Processing comment {cid} ({idx+1}/{total})...")
    ai_result = call_ai_for_comment(comment)
    if ai_result is None:
        ai_result = []

    out_path.write_text(json.dumps(ai_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return cid, "written", out_path, len(ai_result)

def process_all_parallel_ai_only(max_workers=None, skip_if_exists=True):
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Esperaba un array JSON de comentarios en ./data/comments.json")

    total = len(raw)
    if max_workers is None:
        # default conservador (podés setear con env MAX_WORKERS)
        max_workers = int(os.getenv("MAX_WORKERS", "6"))

    written = 0
    skipped = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_one_comment, comment, i, total, skip_if_exists)
            for i, comment in enumerate(raw)
        ]

        for fut in as_completed(futures):
            try:
                cid, status, out_path, n_bullets = fut.result()
                if status == "skipped":
                    skipped += 1
                    print(f"[AI] Skipped {cid} (already exists): {out_path}")
                else:
                    written += 1
                    print(f"[AI] Wrote {n_bullets} bullets -> {out_path}")
            except Exception as e:
                failures += 1
                print(f"[AI] Worker crashed: {e}")

    print(
        f"[AI] Done. total={total}, written={written}, skipped={skipped}, failures={failures}. "
        f"Output dir: {OUT_DIR}"
    )

if __name__ == "__main__":
    process_all_parallel_ai_only()

'''
#!/usr/bin/env python3
# process_comments_serial_ai_only.py
"""
Procesa ./data/comments.json en serie *usando IA para cada comentario*.
- Para cada comment: llama a OpenAI con system+user messages (tu spec completa).
- Espera que el modelo devuelva SOLO el arreglo JSON (por ejemplo: [] o [{...}, ...]).
- Guarda por comment: ./data/output/<commentId>.json
"""

from pathlib import Path
import json
import re
import time
from dotenv import load_dotenv
from datetime import datetime

# Carga .env para que services.openai pueda leer OPENAI_API_KEY
load_dotenv()

# Importa la función que tenés en services/openai.py
from services.openai import ask_openai

INPUT_PATH = Path("./data/comments.json")
OUT_DIR = Path("./data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# System message (tu especificación EXACTA). Puedes editar aquí si querés.
# -----------------------
# -----------------------
# System message: leído desde archivo externo (aux_1_system_message.txt)
# -----------------------
SYSTEM_FILE = Path("./aux_1_system_message.txt")
if not SYSTEM_FILE.exists():
    raise FileNotFoundError(f"System message file not found: {SYSTEM_FILE}")
SYSTEM_MESSAGE = SYSTEM_FILE.read_text(encoding="utf-8")

# -----------------------
# Helpers para llamar la IA y validar respuesta
# -----------------------
def extract_json_array_from_text(txt: str) -> str:
    """
    Intenta extraer el primer/último bloque JSON array válido del texto del modelo.
    También quita bloques de código triple backticks si existen.
    """
    if not txt:
        return ""
    s = txt.strip()
    # quitar wrapper ```json ... ``` o ```
    if s.startswith("```"):
        # quitar la primera línea que puede ser ```json
        s = re.sub(r'^```[^\n]*\n', '', s)
        s = re.sub(r'\n```$', '', s).strip()
    # buscar primer '[' y último ']'
    first = s.find('[')
    last = s.rfind(']')
    if first != -1 and last != -1 and last > first:
        return s[first:last+1]
    return s

def call_ai_for_comment(comment, max_retries=3, wait_seconds=1.0):
    """
    Llama a ask_openai con system + user messages y parsea la respuesta.
    Retorna: lista (el arreglo JSON) o None si falla.
    """
    user_msg = (
        "Procesa este comentario y RESPONDE SOLO con el arreglo JSON solicitado por la especificación.\n\n"
        "Comentario JSON:\n" + json.dumps(comment, ensure_ascii=False) + "\n\n"
        "Insisto: devuelve ÚNICAMENTE el arreglo JSON (por ejemplo: [] o [{...}, ...]) sin texto adicional."
    )
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_msg}
    ]

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp_text = ask_openai(messages, temperature=0.0, model="gpt-4.1")
            candidate_txt = extract_json_array_from_text(resp_text)
            parsed = json.loads(candidate_txt)
            if isinstance(parsed, list):
                # mínima validación estructural opcional: cada objeto debe tener 'bullet_id' y 'commentId'
                valid = True
                for obj in parsed:
                    if not isinstance(obj, dict):
                        valid = False
                        break
                    if 'bullet_id' not in obj or 'commentId' not in obj:
                        valid = False
                        break
                if not valid:
                    raise ValueError("Parsed JSON list but objects fail minimal schema check")
                return parsed
            else:
                raise ValueError("AI returned JSON but not a list")
        except Exception as e:
            last_err = e
            # backoff simple
            time.sleep(wait_seconds * attempt)
            continue
    # if it fails after retries, log and return None
    print(f"[AI] Failed for comment {comment.get('commentId')}: {last_err}")
    return None

# -----------------------
# Pipeline principal (IA siempre)
# -----------------------
def process_all_serial_ai_only():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Esperaba un array JSON de comentarios en ./data/comments.json")

    processed = 0
    for comment in raw:
        cid = comment.get("commentId") or f"no-id-{processed}"
        print(f"[AI] Processing comment {cid} ({processed+1}/{len(raw)})...")
        ai_result = call_ai_for_comment(comment)
        if ai_result is None:
            # si la IA falla, escribimos [] para ese comment para no bloquear el pipeline
            ai_result = []
        out_path = OUT_DIR / f"{cid}.json"
        out_path.write_text(json.dumps(ai_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[AI] Wrote {len(ai_result)} bullets -> {out_path}")
        processed += 1

    print(f"[AI] Processed {processed} comments. Output dir: {OUT_DIR}")

if __name__ == "__main__":
    process_all_serial_ai_only()
'''
