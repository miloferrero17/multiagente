#!/usr/bin/env python3
"""
python3 2_emb.py
Vectorizes JSON files in a folder serially using OpenAI embeddings.
All the variables are set via environment variables.
Output: .data/embeddings/pinecone_ready.jsonl and individual backups in ./data/embeddings/
"""

from pathlib import Path
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()  # load .env into environment

# --- Config (can be overriden via .env) ---
INPUT_DIR = Path(os.getenv("INPUT_DIR", "./data/output")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data/embeddings")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = Path(os.getenv("OUTPUT_JSONL", str(OUTPUT_DIR / "pinecone_ready.jsonl"))).resolve()

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.0"))  # seconds, multiplied by attempt

print("INPUT_DIR =", INPUT_DIR)
print("OUTPUT_DIR =", OUTPUT_DIR)
print("OUTPUT_JSONL =", OUTPUT_JSONL)
print("EMBED_MODEL =", EMBED_MODEL)
print("EMBED_BATCH_SIZE =", EMBED_BATCH_SIZE)

# --- OpenAI client helper from your project ---
try:
    # expects services/openai.py to define _get_client() wrapper
    from services.openai import _get_client
except Exception as e:
    raise RuntimeError("Missing local OpenAI helper 'services.openai._get_client'. Ensure services/openai.py exists.") from e

# initialize client (lazy)
_client = None
def get_client():
    global _client
    if _client is None:
        _client = _get_client()
    return _client

# --- Helpers ---
def iter_input_files(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for p in sorted(input_dir.glob("*.json")):
        yield p

def read_json(path: Path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text)

def prepare_text_for_embedding(obj: dict) -> str:
    """
    Choose which text to embed for a given object.
    Priority:
      - 'formatted' field (if string or list)
      - 'text' field
      - join of other reasonable fields
    """
    if "formatted" in obj:
        f = obj["formatted"]
        if isinstance(f, list):
            return " ".join([str(x) for x in f if x])
        if isinstance(f, str):
            return f
    if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
        return obj["text"].strip()
    # fallback: join metadata values
    meta_vals = []
    for k in ("Team", "Sub-Title", "Sub-Sub-Title", "commentId", "createdAt"):
        if k in obj and obj[k]:
            meta_vals.append(str(obj[k]))
    if meta_vals:
        return " | ".join(meta_vals)
    # last resort: dump the object
    return json.dumps(obj, ensure_ascii=False)

def call_embeddings(model: str, inputs: list, max_retries=3):
    """
    inputs: list[str]
    returns: list[list[float]] aligned with inputs
    """
    client = get_client()
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Using the newer responses-style API may differ by project.
            # The _get_client() should return an OpenAI client compatible with:
            # client.embeddings.create(model=..., input=inputs)
            resp = client.embeddings.create(model=model, input=inputs)
            # resp.data is a list of objects with .embedding
            embeddings = []
            for item in resp.data:
                # item.embedding or item['embedding']
                emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
                if emb is None:
                    # try alternate attribute names
                    emb = item["embedding"] if isinstance(item, dict) and "embedding" in item else None
                embeddings.append(list(emb))
            return embeddings
        except Exception as e:
            last_exc = e
            backoff = RETRY_BACKOFF * attempt
            print(f"[WARN] embeddings attempt {attempt} failed: {e} — backing off {backoff}s")
            time.sleep(backoff)
            continue
    raise RuntimeError(f"Embeddings failed after {max_retries} attempts: {last_exc}")

# --- Main processing ---
def main():
    files = list(iter_input_files(INPUT_DIR))
    print(f"Found {len(files)} input files in {INPUT_DIR}")
    if not files:
        print("No files to process. Exiting.")
        return

    # Open output_jsonl in append mode (create if missing)
    with OUTPUT_JSONL.open("a", encoding="utf-8") as out_f:
        total_written = 0
        total_failed = 0

        for file_idx, file_path in enumerate(files, start=1):
            try:
                data = read_json(file_path)
            except Exception as e:
                print(f"[ERROR] Failed reading {file_path}: {e}")
                continue

            # Expect data to be a list (e.g., list of bullet objects)
            if isinstance(data, dict):
                # maybe the file itself is a single object or contains an array under root key
                # try to find a list inside
                # if not, wrap as single element list
                wrapped = False
                found_list = None
                for v in data.values():
                    if isinstance(v, list):
                        found_list = v
                        break
                if found_list is not None:
                    items = found_list
                else:
                    items = [data]
                    wrapped = True
            elif isinstance(data, list):
                items = data
            else:
                print(f"[WARN] {file_path} contains unsupported JSON root type ({type(data)}). Skipping.")
                continue

            print(f"[{file_idx}/{len(files)}] Processing file {file_path.name} -> {len(items)} items")

            # Prepare batches of texts/ids/metadata
            batch_texts = []
            batch_ids = []
            batch_metadatas = []
            batch_source = file_path.name

            for item_idx, item in enumerate(items, start=1):
                # Determine id for vector
                # Prefer 'bullet_id', then 'id', then generate from filename+index
                vid = None
                if isinstance(item, dict):
                    vid = item.get("bullet_id") or item.get("id") or item.get("bulletId") or None
                if not vid:
                    vid = f"{file_path.stem}_{item_idx}"
                # metadata: include the item itself but avoid huge 'values' fields
                metadata = {}
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k == "values":  # skip if exists
                            continue
                        metadata[k] = v
                metadata["source_file"] = str(file_path.name)

                text = prepare_text_for_embedding(item)
                batch_texts.append(text)
                batch_ids.append(str(vid))
                batch_metadatas.append(metadata)

                # If batch full or last item, flush the batch
                if len(batch_texts) >= EMBED_BATCH_SIZE:
                    try:
                        embeddings = call_embeddings(EMBED_MODEL, batch_texts, max_retries=MAX_RETRIES)
                    except Exception as e:
                        print(f"[ERROR] Embeddings failed for batch in {file_path.name}: {e}")
                        total_failed += len(batch_texts)
                        # reset batch
                        batch_texts = []
                        batch_ids = []
                        batch_metadatas = []
                        continue

                    # write each vector as a JSON line
                    for vid_out, emb_vec, meta in zip(batch_ids, embeddings, batch_metadatas):
                        vec_obj = {"id": vid_out, "values": emb_vec, "metadata": meta}
                        out_f.write(json.dumps(vec_obj, ensure_ascii=False) + "\n")
                        total_written += 1

                    # flush to disk to be safe
                    out_f.flush()
                    os.fsync(out_f.fileno())

                    # reset batch
                    batch_texts = []
                    batch_ids = []
                    batch_metadatas = []

            # flush remaining batch for this file
            if batch_texts:
                try:
                    embeddings = call_embeddings(EMBED_MODEL, batch_texts, max_retries=MAX_RETRIES)
                except Exception as e:
                    print(f"[ERROR] Embeddings failed for final batch in {file_path.name}: {e}")
                    total_failed += len(batch_texts)
                    batch_texts = []
                    batch_ids = []
                    batch_metadatas = []
                else:
                    for vid_out, emb_vec, meta in zip(batch_ids, embeddings, batch_metadatas):
                        vec_obj = {"id": vid_out, "values": emb_vec, "metadata": meta}
                        out_f.write(json.dumps(vec_obj, ensure_ascii=False) + "\n")
                        total_written += 1
                    out_f.flush()
                    os.fsync(out_f.fileno())

            print(f"[OK] Finished {file_path.name}: written so far {total_written}, failed {total_failed}")

    print("Done.")
    print(f"Total vectors written: {total_written}. Failed: {total_failed}. OUTPUT_JSONL: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()


'''
#!/usr/bin/env python3
# embebed.py
"""
Procesa archivos JSON en una carpeta en serie:
 - Escanea INPUT_DIR (por defecto ./data/out)
 - Procesa cada archivo JSON (uno por uno)
 - Para cada objeto en el array del archivo genera embedding con OpenAI
 - Guarda backup por vector en ./data/embeddings/
 - Append a ./data/embeddings/pinecone_ready.jsonl (cada línea es un vector dict)

Config (env or .env):
  INPUT_DIR (default ./data/out)
  OUTPUT_DIR (default ./data/embeddings)
  OPENAI_EMBEDDING_MODEL (default text-embedding-3-small)

Requiere:
  pip install python-dotenv requests
  tener services/openai._get_client() disponible y OPENAI_API_KEY en env/.env
"""
from pathlib import Path
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
print("INPUT_DIR =", INPUT_DIR.resolve())
print("Found files:", len(list(INPUT_DIR.glob("*.json"))))

# Config
INPUT_DIR = Path(os.getenv("INPUT_DIR", "./data/output"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./data/embeddings"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = OUTPUT_DIR / "pinecone_ready.jsonl"

EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Import OpenAI client helper from your project
try:
    from services.openai import _get_client
except Exception as e:
    raise SystemExit(f"Cannot import _get_client from services.openai: {e}")

# small throttle between embeddings (seconds)
EMBED_THROTTLE = float(os.getenv("EMBED_THROTTLE", "0.12"))

def make_embedding(text: str):
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    # Try common extraction patterns
    try:
        return resp.data[0].embedding
    except Exception:
        try:
            r = json.loads(json.dumps(resp, default=str))
            return r["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"Could not extract embedding: {e}; raw: {resp}")

def iter_input_files(input_dir: Path):
    """
    Escanea el directorio ordenadamente y devuelve generator de Path.
    Si el directorio no existe, lo crea (para flujo de desarrollo).
    """
    if not input_dir.exists():
        print(f"Input directory {input_dir} not found — creating it. Add your JSON files there and re-run.")
        input_dir.mkdir(parents=True, exist_ok=True)
        return []
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])
    return files

def prepare_and_write_vector(obj, source_file):
    """
    Dado un objeto (bullet) devuelve el vector dict listo para JSONL o None si no aplica.
    También escribe backup individual.
    """
    if not isinstance(obj, dict):
        return None

    bullet_id = obj.get("bullet_id") or obj.get("bulletId") or None
    commentId = obj.get("commentId")
    createdAt = obj.get("createdAt")
    formatted = obj.get("formatted") or []
    text = None
    if isinstance(formatted, list) and formatted:
        text = formatted[0]
    if not text:
        text = obj.get("text") or obj.get("content")
    if not text or not isinstance(text, str):
        return None

    # build minimal metadata
    metadata = {
        "bullet_id": bullet_id,
        "commentId": commentId,
        "createdAt": createdAt,
        "Team": obj.get("Team", []),
        "Sub-Title": obj.get("Sub-Title", []),
        "Sub-Sub-Title": obj.get("Sub-Sub-Title", []),
        "country": obj.get("country", []),
        "source_file": source_file.name
    }

    emb = make_embedding(text)

    vec_id = bullet_id or f"{commentId}_{int(time.time()*1000)}"
    vector = {"id": str(vec_id), "values": emb, "metadata": metadata}

    # write individual backup (safe, but overwrite if exists)
    backup_file = OUTPUT_DIR / f"{vector['id']}.json"
    try:
        backup_file.write_text(json.dumps({"id": vector["id"], "text": text, "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not write backup {backup_file}: {e}")

    return vector

def process_file(path: Path, out_jsonl_handle):
    """
    Procesa un solo archivo: parsea JSON (espera array) y procesa cada objeto en serie.
    Escribe cada vector como una línea JSON en out_jsonl_handle.
    Retorna (vectors_written, vectors_failed).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Could not read/parse {path}: {e}")
        return 0, 0

    if not isinstance(data, list):
        print(f"[WARN] Skipping {path}: expected top-level array")
        return 0, 0

    written = 0
    failed = 0
    for idx, obj in enumerate(data, start=1):
        try:
            vector = prepare_and_write_vector(obj, path)
            if vector is None:
                failed += 1
                continue
            # write JSONL line
            out_jsonl_handle.write(json.dumps(vector, ensure_ascii=False) + "\n")
            out_jsonl_handle.flush()
            written += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed processing object #{idx} in {path.name}: {e}")
        # throttle to avoid rate limits
        time.sleep(EMBED_THROTTLE)
    return written, failed

def main():
    files = iter_input_files(INPUT_DIR)
    if not files:
        print("No files to process. Put your JSON files into", INPUT_DIR)
        return

    print(f"Found {len(files)} file(s) in {INPUT_DIR}. Processing serially...")

    total_written = 0
    total_failed = 0
    start_ts = time.time()
    # open JSONL in append mode so you can re-run safely
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, fpath in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] Processing file: {fpath.name}")
            w, fail = process_file(fpath, out_f)
            total_written += w
            total_failed += fail
            print(f"  -> written: {w}, failed/skipped: {fail}")
    elapsed = time.time() - start_ts
    print(f"Done. Total vectors written: {total_written}. Total failed/skipped: {total_failed}. Elapsed: {elapsed:.1f}s")
    print("Outputs:")
    print(" - individual backups in:", OUTPUT_DIR)
    print(" - pinecone-ready jsonl:", OUTPUT_JSONL)


def prepare_and_write_vector(obj, source_file):
    """
    Dado un objeto (bullet) devuelve el vector dict listo para JSONL o None si no aplica.
    También escribe backup individual.
    """
    if not isinstance(obj, dict):
        return None

    bullet_id = obj.get("bullet_id") or obj.get("bulletId") or None
    commentId = obj.get("commentId")
    createdAt = obj.get("createdAt")

    # Aceptar formatted como list o string
    formatted = obj.get("formatted")
    text = None
    if isinstance(formatted, list) and formatted:
        # lista -> tomar primer elemento
        text = formatted[0]
    elif isinstance(formatted, str) and formatted.strip():
        # string -> usar tal cual
        text = formatted.strip()

    # fallback a campos alternativos si no hay formatted usable
    if not text:
        text = obj.get("text") or obj.get("content")

    # si todavía no tenemos texto, log y skip
    if not text or not isinstance(text, str) or not text.strip():
        # imprimimos breve debug para entender por qué se saltó
        print(f"[SKIP] missing text for object in {source_file.name} — bullet_id={bullet_id} commentId={commentId}")
        return None

    # build minimal metadata
    metadata = {
        "bullet_id": bullet_id,
        "commentId": commentId,
        "createdAt": createdAt,
        "Team": obj.get("Team", []),
        "Sub-Title": obj.get("Sub-Title", []),
        "Sub-Sub-Title": obj.get("Sub-Sub-Title", []),
        "country": obj.get("country", []),
        "source_file": source_file.name
    }

    # Create embedding (capturamos excepciones)
    try:
        emb = make_embedding(text)
    except Exception as e:
        print(f"[ERROR] embedding failed for bullet_id={bullet_id} commentId={commentId}: {e}")
        return None

    vec_id = bullet_id or f"{commentId}_{int(time.time()*1000)}"
    vector = {"id": str(vec_id), "values": emb, "metadata": metadata}

    # write individual backup (overwrite if exists)
    backup_file = OUTPUT_DIR / f"{vector['id']}.json"
    try:
        backup_file.write_text(json.dumps({"id": vector["id"], "text": text, "metadata": metadata}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Could not write backup {backup_file}: {e}")

    return vector

if __name__ == "__main__":
    main()
'''