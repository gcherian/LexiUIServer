from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from uuid import uuid4
import shutil, json, re, os

# Imaging / OCR
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

router = APIRouter(prefix="/lasso", tags=["lasso"])

# ---- Paths / Globals ---------------------------------------------------------
SRC_DIR: Path = Path(".")
PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")
PUBLIC_BASE: str = "http://localhost:8000"

# Local model root (read by locate router as well)
MODELS_ROOT: Path = Path(os.environ.get(
    "MODELS_ROOT",
    str((Path(__file__).resolve().parent.parent / "models").resolve())
))

def init_paths_for_src(src_dir: Path, public_base: str = "http://localhost:8000"):
    """Call once from main.py to set DATA & base url."""
    global SRC_DIR, PROJECT_ROOT, DATA, PUBLIC_BASE
    SRC_DIR = src_dir.resolve()
    PROJECT_ROOT = SRC_DIR.parent
    PUBLIC_BASE = public_base.rstrip("/")
    DATA = (PROJECT_ROOT / "data").resolve()
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"[unified] SRC_DIR={SRC_DIR}")
    print(f"[unified] DATA={DATA}")
    print(f"[unified] MODELS_ROOT={MODELS_ROOT}")

def doc_dir(doc_id: str) -> Path:
    d = DATA / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def pdf_path(doc_id: str) -> Path:       return doc_dir(doc_id) / "original.pdf"
def meta_path(doc_id: str) -> Path:      return doc_dir(doc_id) / "meta.json"
def boxes_path(doc_id: str) -> Path:     return doc_dir(doc_id) / "boxes.json"

_DOC_URL_RE = re.compile(r"/data/([^/]+)/original\.pdf$", re.IGNORECASE)
def doc_id_from_doc_url(doc_url: str) -> Optional[str]:
    if not doc_url:
        return None
    if (DATA / doc_url / "original.pdf").exists():
        return doc_url
    m = _DOC_URL_RE.search(doc_url)
    if m:
        did = m.group(1)
        if pdf_path(did).exists():
            return did
    if re.fullmatch(r"[A-Za-z0-9_-]{6,32}", doc_url):
        return doc_url
    return None

# ---- Models ------------------------------------------------------------------
class OCRParams(BaseModel):
    dpi: int = 260
    psm: int = 6
    oem: int = 1
    lang: str = "eng"
    binarize: bool = True

class Rect(BaseModel):
    x0: float; y0: float; x1: float; y1: float

class Box(Rect):
    page: int
    text: Optional[str] = None

class UploadResp(BaseModel):
    doc_id: str
    annotated_tokens_url: str
    pages: int

class MetaResp(BaseModel):
    pages: List[Dict[str, float]]

class LassoReq(Rect):
    doc_id: str; page: int

# ---- OCR impl ----------------------------------------------------------------
def _render_pages(pdf_file: Path, dpi: int):
    pdf = pdfium.PdfDocument(str(pdf_file))
    for i in range(len(pdf)):
        page = pdf[i]
        zoom = dpi / 72
        pil = page.render(scale=zoom).to_pil()
        yield i + 1, pil, pil.width, pil.height

def _pre(img: Image.Image, params: OCRParams) -> Image.Image:
    gray = img.convert("L")
    if params.binarize:
        gray = ImageOps.autocontrast(gray)
        gray = gray.point(lambda p: 255 if p > 127 else 0, mode="1").convert("L")
    return gray

def _image_to_data(img: Image.Image, params: OCRParams):
    cfg = f"--oem {params.oem} --psm {params.psm}"
    return pytesseract.image_to_data(img, lang=params.lang, config=cfg, output_type=pytesseract.Output.DICT)

def run_full_ocr(doc_id: str, params: OCRParams) -> MetaResp:
    pdf = pdf_path(doc_id)
    all_boxes: List[Dict[str, Any]] = []
    pages_meta: List[Dict[str, float]] = []

    for page_no, pil, w, h in _render_pages(pdf, params.dpi):
        img = _pre(pil, params)
        d = _image_to_data(img, params)
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            if not txt:
                continue
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            all_boxes.append({
                "page": page_no, "x0": float(x), "y0": float(y),
                "x1": float(x + ww), "y1": float(y + hh), "text": txt
            })
        pages_meta.append({"page": page_no, "width": float(w), "height": float(h)})

    meta_path(doc_id).write_text(json.dumps({
        "pages": pages_meta,
        "params": params.model_dump(),
        "coord_space": {"origin": "top-left", "units": "px@dpi", "dpi": params.dpi}
    }, indent=2))
    boxes_path(doc_id).write_text(json.dumps(all_boxes))
    print(f"[ocr] wrote {boxes_path(doc_id).name} with {len(all_boxes)} tokens")
    return MetaResp(pages=pages_meta)

# ---- Routes ------------------------------------------------------------------
@router.get("/health")
def health():
    return {"ok": True, "svc": "ocr_unified", "MODELS_ROOT": str(MODELS_ROOT)}

@router.post("/upload", response_model=UploadResp)
async def upload(request: Request, pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    doc_id = uuid4().hex[:12]
    p = pdf_path(doc_id)
    with p.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)
    params = OCRParams()
    run_full_ocr(doc_id, params)
    pages = json.loads(meta_path(doc_id).read_text())["pages"]
    base = str(request.base_url).rstrip("/")
    return UploadResp(doc_id=doc_id,
                      annotated_tokens_url=f"{base}/data/{doc_id}/original.pdf",
                      pages=len(pages))

@router.get("/doc/{doc_id}/meta", response_model=MetaResp)
async def get_meta(doc_id: str):
    mp = meta_path(doc_id)
    if not mp.exists(): raise HTTPException(404, "Meta not found")
    return MetaResp(pages=json.loads(mp.read_text())["pages"])

@router.get("/doc/{doc_id}/boxes", response_model=List[Box])
async def get_boxes(doc_id: str, page: Optional[int] = Query(None)):
    bp = boxes_path(doc_id)
    if not bp.exists(): return []
    items = [Box(**b) for b in json.loads(bp.read_text())]
    if page is not None:
        items = [b for b in items if b.page == page]
    return items

@router.post("/lasso")
async def lasso_crop(req: LassoReq):
    mp = meta_path(req.doc_id)
    if not mp.exists():
        raise HTTPException(404, "Meta missing")
    meta = json.loads(mp.read_text())
    dpi = meta.get("params", {}).get("dpi", 260)

    pdf = pdfium.PdfDocument(str(pdf_path(req.doc_id)))
    if req.page < 1 or req.page > len(pdf):
        raise HTTPException(400, f"Page out of range: {req.page}")

    pil = pdf[req.page - 1].render(scale=(dpi / 72)).to_pil()
    gray = ImageOps.autocontrast(pil.convert("L"))

    x0, y0 = float(min(req.x0, req.x1)), float(min(req.y0, req.y1))
    x1, y1 = float(max(req.x0, req.x1)), float(max(req.y0, req.y1))

    pad = 4
    X0 = max(0, int(x0 - pad)); Y0 = max(0, int(y0 - pad))
    X1 = min(gray.width - 1, int(x1 + pad)); Y1 = min(gray.height - 1, int(y1 + pad))

    crop = gray.crop((X0, Y0, X1, Y1))
    if crop.width < 140 or crop.height < 40:
        scale = 3 if max(crop.width, crop.height) < 60 else 2
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)

    def ocr(psm: int) -> str:
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(
            crop, lang=meta.get("params", {}).get("lang", "eng"), config=cfg
        ).strip()

    cand = sorted(((len(t), t) for t in (ocr(6), ocr(7), ocr(11))), reverse=True)
    best = cand[0][1] if cand else ""

    crop_path = doc_dir(req.doc_id) / "last_crop.png"
    crop_url = None
    try:
        crop.save(crop_path)
        crop_url = f"/data/{req.doc_id}/last_crop.png"
    except Exception:
        pass

    return {
        "text": best,
        "rect_used": {"page": req.page, "x0": X0, "y0": Y0, "x1": X1, "y1": Y1},
        "page_size": {"width": gray.width, "height": gray.height},
        "crop_url": crop_url,
    }