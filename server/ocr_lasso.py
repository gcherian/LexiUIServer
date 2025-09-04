from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pathlib import Path
import shutil, json

import numpy as np
import cv2
import pytesseract
import pypdfium2 as pdfium

# ---------------------- App & Static Mounts ----------------------

app = FastAPI(title="EDIP OCR Workbench", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"; DATA.mkdir(exist_ok=True)
OUT  = BASE / "out" ; OUT.mkdir(exist_ok=True)

app.mount("/data", StaticFiles(directory=str(DATA), html=False), name="data")
app.mount("/out" , StaticFiles(directory=str(OUT ), html=False), name="out")

# ---------------------- Models ----------------------

class OCRParams(BaseModel):
    dpi: int = 220               # rasterization DPI
    psm: int = 6                 # tesseract page segmentation mode
    oem: int = 1                 # tesseract engine mode
    lang: str = "eng"
    binarize: bool = True
    deskew: bool = True
    dilate: int = 0              # px kernel; 0 disables
    erode: int = 0               # px kernel; 0 disables

class Rect(BaseModel):
    x0: float; y0: float; x1: float; y1: float

class Box(Rect):
    page: int
    id: Optional[str] = None
    label: Optional[str] = None
    text: Optional[str] = None    # raw OCR token text

class UploadResp(BaseModel):
    doc_id: str
    annotated_tokens_url: str     # we serve original PDF; overlay happens on client
    pages: int

class MetaResp(BaseModel):
    pages: List[Dict[str, float]] # [{page,width,height} in OCR px @dpi]

class SearchReq(BaseModel):
    doc_id: str; query: str; topk: int = 20

class LassoReq(Rect):
    doc_id: str; page: int

class BoxesSaveReq(BaseModel):
    boxes: List[Box]

# ---------------------- Utils ----------------------

def doc_dir(doc_id: str) -> Path:
    d = DATA / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def pdf_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "original.pdf"

def meta_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "meta.json"

def boxes_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "boxes.json"

def render_pdf_to_images(pdf_file: Path, dpi: int) -> List[np.ndarray]:
    """Rasterize PDF pages to RGB numpy arrays using pypdfium2."""
    imgs = []
    pdf = pdfium.PdfDocument(str(pdf_file))
    for i in range(len(pdf)):
        page = pdf[i]
        # scale from DPI (72 is 1.0)
        zoom = dpi / 72
        bitmap = page.render(scale=zoom).to_pil()
        imgs.append(np.array(bitmap))  # RGB
    return imgs

def _deskew(img_gray: np.ndarray) -> np.ndarray:
    # estimate skew via moments; simple and robust for forms
    coords = np.column_stack(np.where(img_gray < 250))  # non-white
    if coords.size == 0:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(img_rgb: np.ndarray, params: OCRParams) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if params.deskew:
        gray = _deskew(gray)
    if params.binarize:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if params.dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (params.dilate, params.dilate))
        gray = cv2.dilate(gray, k, iterations=1)
    if params.erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (params.erode, params.erode))
        gray = cv2.erode(gray, k, iterations=1)
    return gray

def tesseract_tokens(img_gray: np.ndarray, params: OCRParams) -> List[Box]:
    cfg = f"--oem {params.oem} --psm {params.psm}"
    data = pytesseract.image_to_data(
        img_gray,
        lang=params.lang,
        config=cfg,
        output_type=pytesseract.Output.DICT
    )
    boxes = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:  # skip blanks
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        boxes.append(Box(page=1, x0=float(x), y0=float(y), x1=float(x+w), y1=float(y+h),
                         id=None, label=None, text=txt))
    return boxes

def run_full_ocr(doc_id: str, params: OCRParams) -> MetaResp:
    """Rasterize PDF @dpi and run OCR per page; persist meta + boxes (with per-page)."""
    pdf = pdf_path(doc_id)
    imgs_rgb = render_pdf_to_images(pdf, params.dpi)
    all_boxes: List[Box] = []
    pages_meta = []

    for pi, rgb in enumerate(imgs_rgb, start=1):
        gray = preprocess_for_ocr(rgb, params)
        # Page-specific OCR
        cfg = f"--oem {params.oem} --psm {params.psm}"
        data = pytesseract.image_to_data(
            gray,
            lang=params.lang,
            config=cfg,
            output_type=pytesseract.Output.DICT
        )
        n = len(data["text"])
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            all_boxes.append(Box(page=pi, x0=float(x), y0=float(y), x1=float(x+w), y1=float(y+h),
                                 id=None, label=None, text=txt))
        # meta (image size in px == OCR coordinate space)
        h, w = gray.shape[:2]
        pages_meta.append({"page": pi, "width": float(w), "height": float(h)})

    # persist
    with open(meta_path(doc_id), "w") as f:
        json.dump({"pages": pages_meta, "params": params.model_dump()}, f, indent=2)

    with open(boxes_path(doc_id), "w") as f:
        json.dump([b.model_dump() for b in all_boxes], f)

    return MetaResp(pages=pages_meta)

def load_boxes(doc_id: str) -> List[Box]:
    p = boxes_path(doc_id)
    if not p.exists(): return []
    raw = json.loads(p.read_text())
    return [Box(**b) for b in raw]

# ---------------------- Endpoints ----------------------

@app.post("/lasso/upload", response_model=UploadResp)
async def upload(pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    doc_id = uuid4().hex[:12]
    ddir = doc_dir(doc_id)
    p = pdf_path(doc_id)
    with p.open("wb") as f:
        shutil.copyfileobj(pdf.file, f)

    # initial OCR with defaults (good coverage, not too slow)
    params = OCRParams()
    run_full_ocr(doc_id, params)

    return UploadResp(
        doc_id=doc_id,
        annotated_tokens_url=f"/data/{doc_id}/original.pdf",
        pages=len(json.loads(meta_path(doc_id).read_text())["pages"])
    )

@app.get("/lasso/doc/{doc_id}/meta", response_model=MetaResp)
async def get_meta(doc_id: str):
    m = meta_path(doc_id)
    if not m.exists():
        raise HTTPException(404, "Meta not found")
    obj = json.loads(m.read_text())
    return MetaResp(pages=obj["pages"])

@app.post("/lasso/doc/{doc_id}/rebuild", response_model=MetaResp)
async def rebuild(doc_id: str, params: OCRParams):
    if not pdf_path(doc_id).exists():
        raise HTTPException(404, "Doc not found")
    return run_full_ocr(doc_id, params)

@app.get("/lasso/doc/{doc_id}/boxes", response_model=List[Box])
async def get_boxes(doc_id: str):
    return load_boxes(doc_id)

@app.put("/lasso/doc/{doc_id}/boxes")
async def put_boxes(doc_id: str, req: BoxesSaveReq):
    p = boxes_path(doc_id)
    with p.open("w") as f:
        json.dump([b.model_dump() for b in req.boxes], f, indent=2)
    return {"ok": True, "count": len(req.boxes)}

@app.post("/lasso/search")
async def search(req: SearchReq):
    import difflib
    boxes = load_boxes(req.doc_id)
    matches = []
    q = req.query.strip()
    if not q:
        return {"matches": []}
    for b in boxes:
        # simple fuzzy ratio on token text
        score = difflib.SequenceMatcher(None, q.lower(), (b.text or "").lower()).ratio()
        if score > 0.45:
            matches.append({
                "page": b.page,
                "bbox": {"x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1},
                "text": b.text,
                "score": round(score, 3),
            })
    # sort best first, cap topk
    matches.sort(key=lambda m: m["score"], reverse=True)
    return {"matches": matches[:req.topk]}

@app.post("/lasso/lasso")
async def lasso_crop(req: LassoReq):
    """OCR the selected rectangle region in OCR space."""
    m = meta_path(req.doc_id); 
    if not m.exists():
        raise HTTPException(404, "Meta missing")
    meta = json.loads(m.read_text())
    pdf = pdf_path(req.doc_id)
    dpi = meta.get("params", {}).get("dpi", 220)

    # render the one page only
    pdfdoc = pdfium.PdfDocument(str(pdf))
    page = pdfdoc[req.page-1]
    zoom = dpi / 72
    bitmap = page.render(scale=zoom).to_pil()
    rgb = np.array(bitmap)
    gray = preprocess_for_ocr(rgb, OCRParams(**meta.get("params", {})))

    # crop
    x0, y0, x1, y1 = map(int, [req.x0, req.y0, req.x1, req.y1])
    x0, y0 = max(x0,0), max(y0,0)
    x1, y1 = min(x1, gray.shape[1]-1), min(y1, gray.shape[0]-1)
    crop = gray[y0:y1, x0:x1]
    text = pytesseract.image_to_string(crop, lang=meta.get("params", {}).get("lang","eng"))

    return {"text": text.strip()}

@app.post("/lasso/audit")
async def audit(payload: Dict[str, Any]):
    (OUT / "audit.log").write_text(
        ((OUT / "audit.log").read_text() if (OUT/"audit.log").exists() else "")
        + json.dumps(payload) + "\n"
    )
    return {"ok": True}
