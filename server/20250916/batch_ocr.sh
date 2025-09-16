# Example: re-run OCR for remaining PDFs in a folder
python - <<'PY'
import os, json
from pathlib import Path
from ocr_unified import run_full_ocr, OCRParams, pdf_path, meta_path, doc_dir
# adjust import path if needed: from src.routers.ocr_unified import ...

ROOT = Path("data")  # where subfolders = doc_ids
todo = []
for d in ROOT.iterdir():
    if not d.is_dir(): continue
    if not (d/"original.pdf").exists(): continue
    if not (d/"meta.json").exists():
        todo.append(d.name)

print(f"Re-OCR for {len(todo)} docs...")
for did in todo:
    run_full_ocr(did, OCRParams())
    print("OK", did)
PY