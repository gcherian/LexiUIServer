# scripts/batch_ocr.py
# Recursively OCR PDFs under a root and populate data/{doc_id}/ (original.pdf, meta.json, boxes.json, pages.txt)
# Local-only: pdfium + pytesseract. Safe to run multiple times (skips done unless --force).
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

def render_pdf_pages(pdf_file: Path, dpi: int):
    pdf = pdfium.PdfDocument(str(pdf_file))
    for i in range(len(pdf)):
        page = pdf[i]
        zoom = dpi / 72
        pil = page.render(scale=zoom).to_pil()
        yield i+1, pil, pil.width, pil.height

def preprocess_pil(img: Image.Image, binarize: bool) -> Image.Image:
    gray = img.convert("L")
    if binarize:
        gray = ImageOps.autocontrast(gray)
        gray = gray.point(lambda p: 255 if p > 127 else 0, mode="1").convert("L")
    return gray

def tesseract_image_to_data(img: Image.Image, lang: str, oem: int, psm: int):
    cfg = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

def run_full_ocr_to_dir(src_pdf: Path, out_dir: Path, dpi=260, lang="eng", oem=1, psm=6, binarize=True, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy original
    dst_pdf = out_dir / "original.pdf"
    if not dst_pdf.exists() or force:
        shutil.copy2(src_pdf, dst_pdf)
    meta_p = out_dir / "meta.json"
    boxes_p = out_dir / "boxes.json"
    pages_p = out_dir / "pages.txt"
    if meta_p.exists() and boxes_p.exists() and pages_p.exists() and not force:
        return  # already done

    all_boxes = []
    pages_meta = []
    all_page_texts = []

    for page_no, pil, w, h in render_pdf_pages(dst_pdf, dpi):
        img = preprocess_pil(pil, binarize)
        d = tesseract_image_to_data(img, lang, oem, psm)
        tokens_this_page = []
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            if not txt: continue
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            all_boxes.append({"page":page_no,"x0":float(x),"y0":float(y),"x1":float(x+ww),"y1":float(y+hh),"text":txt})
            tokens_this_page.append(txt)
        pages_meta.append({"page":page_no,"width":float(w),"height":float(h)})
        all_page_texts.append(" ".join(tokens_this_page))

    meta_p.write_text(json.dumps({
        "pages": pages_meta,
        "params": {"dpi":dpi,"psm":psm,"oem":oem,"lang":lang,"binarize":binarize},
        "coord_space": {"origin":"top-left","units":"px@dpi","dpi":dpi}
    }, indent=2))
    boxes_p.write_text(json.dumps(all_boxes))
    pages_p.write_text("\n\n".join(all_page_texts))

def doc_id_for(pdf_path: Path) -> str:
    # doc_id = filename stem (alnum/underscore), else sanitized hash-like
    stem = pdf_path.stem
    m = re.sub(r"[^A-Za-z0-9_-]", "_", stem)
    return m[:32] if m else f"d_{abs(hash(pdf_path)) & 0xfffffff:07x}"

def main():
    ap = argparse.ArgumentParser(description="Batch OCR PDFs into data/{doc_id}")
    ap.add_argument("--root", required=True, help="Root folder containing PDFs (e.g., EDIP-Extraction or a SharePoint export).")
    ap.add_argument("--data", default="data", help="Output data folder (default: ./data)")
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--oem", type=int, default=1)
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--binarize", action="store_true", default=True)
    ap.add_argument("--force", action="store_true", help="Re-OCR even if outputs exist.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data = Path(args.data).resolve()
    pdfs = list(root.rglob("*.pdf"))
    print(f"[batch_ocr] PDFs found: {len(pdfs)} under {root}")
    for pdf in pdfs:
        did = doc_id_for(pdf)
        out_dir = data / did
        print(f"[batch_ocr] OCR: {pdf} -> {out_dir}")
        run_full_ocr_to_dir(pdf, out_dir, dpi=args.dpi, lang=args.lang, oem=args.oem, psm=args.psm, binarize=args.binarize, force=args.force)
    print("[batch_ocr] Done.")

if __name__ == "__main__":
    main()