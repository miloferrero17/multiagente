#!/usr/bin/env python3
"""
services/process_comments_ai.py

Procesa comentarios UNO A UNO con la IA usando tu wrapper ask_openai.
Genera out/<commentId>.json con la respuesta (arreglo JSON).

Uso:
  python3 services/process_comments_ai.py --input comments.json --out-dir out
  cat single.json | python3 services/process_comments_ai.py --stdin -

Requisitos:
 - Tener OPENAI_API_KEY en .env o en el entorno.
 - Tener una función ask_openai(messages, temperature=0, model="gpt-4.1")
   exportada desde service/openai.py o services/openai.py
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import argparse
from typing import Any, Dict, List, Optional


from dotenv import load_dotenv
load_dotenv()

# ## IMPORT FALLBACK: allow loading services.openai regardless of sys.path quirks
# Try normal imports first (existing code will attempt them). If ask_openai is not found,
# this fallback will try to load services/openai.py by path using importlib.
try:
    import importlib
    import importlib.util
    import os
    import sys
    # ensure repo root is on sys.path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
except Exception:
    pass

# End IMPORT FALLBACK


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Try import ask_openai from common local modules
ask_openai = None
_import_errors = []
try:
    from service.openai import ask_openai  # type: ignore
    logger.debug("Imported ask_openai from service.openai")
except Exception as e:
    _import_errors.append(("service.openai", str(e)))
    try:
        from services.openai import ask_openai  # type: ignore
        logger.debug("Imported ask_openai from services.openai")
    except Exception as e2:
        _import_errors.append(("services.openai", str(e2)))
        ask_openai = None  # will be checked at call time

# System prompt (exact as provided by you)
SYSTEM_MESSAGE = '''Eres un formateador de texto.

En cada llamada vas a recibir como input UN SOLO objeto JSON llamado comentario. Ese objeto incluye al menos estos campos:

commentId: identificador único del comentario.

createdAt: fecha/hora en ISO 8601 (ej: "2025-12-12T15:44:29.278Z")

text: contenido del comentario.

Ejemplo:

{
  "commentId": "693c385dcf57f598a5335964",
  "createdAt": "2025-12-12T15:44:29.278Z",
  "text": "Genova\n\n- Ajuste de pricing en COL\n- Lanzamiento ELO"
}

Objetivo

Transformar comentario.text en una lista de bullets normalizados e incluir la fecha del comentario en el string final.

Reglas

Ignora todos los campos excepto commentId, createdAt y text.

Define fecha como:

fecha = comentario.createdAt convertido a YYYY-MM-DD (tomado de la parte de fecha del ISO 8601).

Ejemplo: "2025-12-12T15:44:29.278Z" → "2025-12-12".

Analiza text para identificar:

Título, subtítulo y sub-subtítulo (si existen), siguiendo prácticas comunes (por ejemplo: líneas sueltas, encabezados, separadores, etc.). Del team, titulo, sub-titulo y sub-sub-titulo transformamos a minúsculas y elimina todo los caracteres especiales. El resto del contenido debe convertirse en bullets (ítems) y los bullets tienen un bullet_id = <commentID>_Orden de aparición. Si ya hay bullets (por ejemplo -, *, •, numeración), úsalos.Si no hay bullets explícitos, intenta separar en ítems por semántica (frases/cortes naturales). Si aun así no es claro, considera que no hay bullets. Debes generar UN objeto por cada bullet detectado. Para cada bullet, construye un string formatted con este formato: Siempre debe incluir como mínimo fecha, título e ítem: "<texto del ítem>" Si existen subtítulo y/o sub-subtítulo, se agregan en el medio, en este orden: "[<subtítulo>] - <texto del ítem>",  "[<subtítulo>] - [<sub-subtítulo>] - <texto del ítem>" El output debe ser SIEMPRE un arreglo JSON con tantos objetos como bullets se hayan encontrado. 
El pais del texto: 
Argentina -> MLA; 
Brasil -> MLB; 
Mexico -> MLM; 
Uruguay -> MLU; 
Chile -> MLC; 
Colombia -> MLO; 
en caso de no poder detectar el pais responder “CORP”.

Cada objeto del arreglo debe tener exactamente esta estructura:

{ 
 "bullet_id": <commentID>_Orden de aparición,
  "commentId": "<comentario.commentId>",
   "createdAt": "2025-12-12T15:44:29.278Z",
  "Team": [<título>],
 "Sub-Title":  [<subtítulo>] ,
 "Sub-Sub-Title":  [<sub-subtítulo>],
  "formatted": [<texto del ítem>],
"country":  [<texto del ítem>]

}

El commentId se repite en todos los objetos generados.

Si no hay bullets, devuelve un arreglo vacío:

[]

Restricción de salida

NO devuelvas explicaciones.

Devuelve ÚNICAMENTE el arreglo JSON.
'''


def make_messages_for_comment(comment: Dict[str, Any]) -> List[Dict[str, str]]:
    user_content = json.dumps({
        "commentId": comment.get("commentId") or comment.get("id") or comment.get("comment_id"),
        "createdAt": comment.get("createdAt"),
        "text": comment.get("text")
    }, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def call_model_with_retries(messages: List[Dict[str, str]],
                            model: str = "gpt-4.1",
                            temperature: float = 0.0,
                            max_retries: int = 3,
                            pause: float = 1.0) -> str:
    if not ask_openai:
        raise RuntimeError(
            "ask_openai no está disponible. Intentos de import previos:\n" +
            "\n".join(f"{m}: {err}" for m, err in _import_errors)
        )
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling model (attempt %d) model=%s temp=%.2f", attempt, model, temperature)
            resp_text = ask_openai(messages, temperature=temperature, model=model)
            if not isinstance(resp_text, str):
                resp_text = str(resp_text)
            return resp_text
        except Exception as e:
            logger.warning("Model call error: %s", e)
            if attempt < max_retries:
                wait = pause * (2 ** (attempt - 1))
                logger.info("Retrying after %.1f seconds...", wait)
                time.sleep(wait)
            else:
                logger.error("Max retries reached for model call.")
                raise
    raise RuntimeError("Retries exhausted")


def extract_json_array_from_text(resp_text: str) -> List[Any]:
    resp_text = resp_text.strip()
    try:
        parsed = json.loads(resp_text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    start = resp_text.find("[")
    end = resp_text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No se encontró un arreglo JSON en la respuesta del modelo.")
    snippet = resp_text[start:end+1]
    return json.loads(snippet)


def validate_and_save_output(comment_id: str, resp_text: str, out_dir: str) -> List[Dict[str, Any]]:
    parsed = extract_json_array_from_text(resp_text)
    if not isinstance(parsed, list):
        raise ValueError("La respuesta parseada no es un arreglo JSON.")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{comment_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)
    logger.info("Saved model output: %s (items=%d)", out_path, len(parsed))
    return parsed


def process_single_comment_job_impl(comment: Dict[str, Any],
                                    model: Optional[str] = None,
                                    temperature: Optional[float] = None,
                                    out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    comment_id = comment.get("commentId") or comment.get("id") or comment.get("comment_id")
    if not comment_id:
        raise ValueError("El comentario no tiene commentId")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1")
    temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", "0"))
    out_dir = out_dir or os.getenv("OUT_DIR", "out")
    messages = make_messages_for_comment({
        "commentId": comment_id,
        "createdAt": comment.get("createdAt"),
        "text": comment.get("text")
    })
    resp_text = call_model_with_retries(messages, model=model, temperature=temperature)
    parsed = validate_and_save_output(comment_id, resp_text, out_dir)
    return parsed


def process_comments_file(path: str, model: str, temperature: float, out_dir: str,
                          delay_between_calls: float = 0.6, dry_run: bool = False):
    with open(path, "r", encoding="utf-8") as f:
        comments = json.load(f)
    if not isinstance(comments, list):
        raise ValueError("El archivo de entrada debe contener un arreglo JSON.")
    logger.info("Found %d comments in %s", len(comments), path)
    os.makedirs(out_dir, exist_ok=True)
    for idx, comment in enumerate(comments, start=1):
        logger.info("(%d/%d) commentId=%s", idx, len(comments), comment.get("commentId"))
        if dry_run:
            logger.info("Dry run: messages would be: %s", make_messages_for_comment(comment))
        else:
            try:
                process_single_comment_job_impl(comment, model=model, temperature=temperature, out_dir=out_dir)
            except Exception as e:
                logger.exception("Error procesando comment %s: %s", comment.get("commentId"), e)
        time.sleep(delay_between_calls)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process Trello comments one-by-one with OpenAI.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="JSON file with array of comments")
    group.add_argument("--stdin", "-s", nargs="?", const="-", help="Read a single comment JSON from stdin (use '-' or provide no path)")
    p.add_argument("--out-dir", "-o", default=os.getenv("OUT_DIR", "out"), help="Output directory (default: out)")
    p.add_argument("--delay", "-d", type=float, default=float(os.getenv("DELAY_SECS", "10")), help="Delay between calls in seconds")
    p.add_argument("--dry-run", action="store_true", help="Do not call the API; show messages that would be sent.")
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1"), help="Model to use")
    p.add_argument("--temperature", type=float, default=float(os.getenv("OPENAI_TEMPERATURE", "0")), help="Temperature for model")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not ask_openai:
        logger.warning("ask_openai no encontrado. Intentos de import y errores:")
        for module_name, err in _import_errors:
            logger.warning("%s -> %s", module_name, err)
        logger.warning("Asegurate de tener service/openai.py o services/openai.py con ask_openai.")
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        if args.input:
            process_comments_file(args.input, model=args.model, temperature=args.temperature, out_dir=args.out_dir, delay_between_calls=args.delay, dry_run=args.dry_run)
        else:
            logger.info("Reading single comment JSON from stdin...")
            raw = sys.stdin.read()
            comment = json.loads(raw)
            if args.dry_run:
                logger.info("Dry run: messages would be: %s", make_messages_for_comment(comment))
            else:
                process_single_comment_job_impl(comment, model=args.model, temperature=args.temperature, out_dir=args.out_dir)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
