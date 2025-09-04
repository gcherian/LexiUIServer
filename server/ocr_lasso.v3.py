# server/lasso_router.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pathlib import Path
import shutil, json, time, difflib

import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

router = APIRouter(prefix="/lasso", tags=["lasso"])

# Set by host app
BASE: Path = Path(".")
DATA: Path
OUT: Path
PROM: Path
PUBLIC_BASE: str = "http://localhost:8000"  # used to build absolute URLs

def init_paths(base: Path, public_base: str = "http://localhost:8000"):
    global BASE, DATA, OUT, PROM, PUBLIC_BASE
    BASE = base
    PUBLIC_BASE = public_base.rstrip("/")
    DATA = BASE / "data"; DATA.mkdir(exist_ok=True, parents=True)
    OUT  = BASE / "out" ; OUT.mkdir(exist_ok=True, parents=True)
    PROM = BASE / "prom"; PROM.mkdir(exist_ok=True, parents=True)

# ---------- Models ----------
class OCRParams(BaseModel):
    dpi: int = 220
    psm: int = 6
    oem: int = 1
    lang: str = "eng"
    binarize: bool = True
    deskew: bool = False
    dilate: int = 0
    erode: int = 0

class Rect(BaseModel):
    x0: float; y0: float; x1: float; y1: float

class Box(Rect):
    page: int
    id: Optional[str] = None
    label: Optional[str] = None
    text: Optional[str] = None

class UploadResp(BaseModel):
    doc_id: str
    annotated_tokens_url: str
    pages: int

class MetaResp(BaseModel):
    pages: List[Dict[str, float]]

class SearchReq(BaseModel):
    doc_id: str; query: str; topk: int = 20

class BoxesSaveReq(BaseModel):
    boxes: List[Box]

class PromField(BaseModel):
    key: str
    label: str
    type: str = "string"
    required: bool = False
    enum: Optional[List[str]] = None

class PromCatalog(BaseModel):
    doctype: str
    version: str
    fields: List[PromField]

class FieldState(BaseModel):
    key: str
    value: Optional[Any] = None
    bbox: Optional[Dict[str, Any]] = None
    source: str = "user"
    confidence: float = 0.0

class FieldDocState(BaseModel):
    doc_id: str
    doctype: str
    fields: List[FieldState]
    audit: List[Dict[str, Any]] = []

class ECMExtractReq(BaseModel):
    doc_id: str
    doctype: str

class SetDocTypeReq(BaseModel):
    doctype: str

class SetDocTypeShort(BaseModel):  # convenience
    doc_id: str
    doctype: str

class BindReq(BaseModel):
    doc_id: str
    page: int
    rect: Rect
    key: str

# ---------- Paths ----------
def doc_dir(doc_id: str) -> Path:
    d = DATA / doc_id; d.mkdir(parents=True, exist_ok=True); return d

def pdf_path(doc_id: str) -> Path: return doc_dir(doc_id) / "original.pdf"
def meta_path(doc_id: str) -> Path: return doc_dir(doc_id) / "meta.json"
def boxes_path(doc_id: str) -> Path: return doc_dir(doc_id) / "boxes.json"
def field_state_path(doc_id: str) -> Path: return doc_dir(doc_id) / "fields.json"
def prom_path(doctype: str) -> Path: return PROM / f"{doctype}.json"

# ---------- OCR helpers ----------
def render_pdf_pages(pdf_file: Path, dpi: int):
    pdf = pdfium.PdfDocument(str(pdf_file))
    for i in range(len(pdf)):
        page = pdf[i]
        zoom = dpi / 72
        pil = page.render(scale=zoom).to_pil()
        yield i+1, pil, pil.width, pil.height

def preprocess_pil(img: Image.Image, params: OCRParams) -> Image.Image:
    gray = img.convert("L")
    if params.binarize:
        gray = ImageOps.autocontrast(gray)
        gray = gray.point(lambda p: 255 if p > 127 else 0, mode="1").convert("L")
    return gray

def tesseract_image_to_data(img: Image.Image, params: OCRParams):
    cfg = f"--oem {params.oem} --psm {params.psm}"
    return pytesseract.image_to_data(
        img, lang=params.lang, config=cfg, output_type=pytesseract.Output.DICT
    )

def run_full_ocr(doc_id: str, params: OCRParams) -> MetaResp:
    pdf = pdf_path(doc_id)
    all_boxes: List[Dict[str, Any]] = []
    pages_meta: List[Dict[str, float]] = []
    for page_no, pil, w, h in render_pdf_pages(pdf, params.dpi):
        img = preprocess_pil(pil, params)
        d = tesseract_image_to_data(img, params)
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            if not txt: continue
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            all_boxes.append({"page":page_no,"x0":float(x),"y0":float(y),"x1":float(x+ww),"y1":float(y+hh),"text":txt})
        pages_meta.append({"page":page_no,"width":float(w),"height":float(h)})
    meta_path(doc_id).write_text(json.dumps({"pages": pages_meta, "params": params.model_dump()}, indent=2))
    boxes_path(doc_id).write_text(json.dumps(all_boxes))
    return MetaResp(pages=pages_meta)

def load_boxes(doc_id: str) -> List[Box]:
    p = boxes_path(doc_id)
    if not p.exists(): return []
    return [Box(**b) for b in json.loads(p.read_text())]

# ---------- Field-state helpers ----------
def ensure_field_state(doc_id: str, doctype: str) -> FieldDocState:
    fp = field_state_path(doc_id)
    if fp.exists():
        return FieldDocState(**json.loads(fp.read_text()))
    # auto-init from PROM
    pp = prom_path(doctype)
    if not pp.exists():
        raise HTTPException(404, f"PROM catalog not found for doctype '{doctype}'")
    prom = json.loads(pp.read_text())
    fields = [FieldState(key=f["key"]) for f in prom["fields"]]
    state = FieldDocState(
        doc_id=doc_id, doctype=doctype, fields=fields,
        audit=[{"ts": int(time.time()), "event":"auto_init", "doctype": doctype}]
    )
    fp.write_text(state.model_dump_json(indent=2))
    return state

# ---------- Endpoints ----------
@router.get("/health")
def health(): return {"ok": True, "svc": "lasso-router"}

@router.post("/upload", response_model=UploadResp)
async def upload(pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    doc_id = uuid4().hex[:12]
    p = pdf_path(doc_id); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f: shutil.copyfileobj(pdf.file, f)
    params = OCRParams()
    run_full_ocr(doc_id, params)
    pages = json.loads(meta_path(doc_id).read_text())["pages"]
    # return ABSOLUTE URL so PDF.js always loads
    return UploadResp(
        doc_id=doc_id,
        annotated_tokens_url=f"{PUBLIC_BASE}/data/{doc_id}/original.pdf",
        pages=len(pages)
    )

@router.get("/doc/{doc_id}/meta", response_model=MetaResp)
async def get_meta(doc_id: str):
    m = meta_path(doc_id)
    if not m.exists(): raise HTTPException(404, "Meta not found")
    return MetaResp(pages=json.loads(m.read_text())["pages"])

@router.post("/doc/{doc_id}/rebuild", response_model=MetaResp)
async def rebuild(doc_id: str, params: OCRParams):
    if not pdf_path(doc_id).exists(): raise HTTPException(404, "Doc not found")
    return run_full_ocr(doc_id, params)

@router.get("/doc/{doc_id}/boxes", response_model=List[Box])
async def get_boxes(doc_id: str): return load_boxes(doc_id)

@router.put("/doc/{doc_id}/boxes")
async def put_boxes(doc_id: str, req: BoxesSaveReq):
    boxes_path(doc_id).write_text(json.dumps([b.model_dump() for b in req.boxes], indent=2))
    return {"ok": True, "count": len(req.boxes)}

@router.post("/search")
async def search(req: SearchReq):
    tokens = load_boxes(req.doc_id)
    q = req.query.strip().lower()
    if not q: return {"matches": []}
    matches = []
    for t in tokens:
        txt = (t.text or "").lower() if t.text else ""
        if not txt: continue
        score = difflib.SequenceMatcher(None, q, txt).ratio()
        if score > 0.45:
            matches.append({
                "page": t.page,
                "bbox": {"x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1},
                "text": t.text,
                "score": round(score, 3),
            })
    matches.sort(key=lambda m: m["score"], reverse=True)
    return {"matches": matches[:req.topk]}


# If not already defined in your file:
class LassoReq(BaseModel):
    doc_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

@router.post("/lasso")
async def lasso_crop(req: LassoReq):
    """
    Robust lasso OCR:
    - soft grayscale (no hard binarization)
    - safe padding around crop
    - upscale small regions (2–3x)
    - try multiple PSMs (6,7,11) and pick best
    - fallback: inflate region and retry if OCR is too short
    Returns: {"text": "..."}  (UI depends on this exact shape)
    """
    # ---- load page + meta ----
    m = meta_path(req.doc_id)
    if not m.exists():
        raise HTTPException(404, "Meta missing")
    meta = json.loads(m.read_text())
    dpi = meta.get("params", {}).get("dpi", 260)

    # render page at OCR dpi
    pdf = pdfium.PdfDocument(str(pdf_path(req.doc_id)))
    if req.page < 1 or req.page > len(pdf):
        raise HTTPException(400, f"Page out of range: {req.page}")
    pil = pdf[req.page - 1].render(scale=(dpi / 72)).to_pil()

    # soft grayscale (avoid hard thresholding which clips thin glyphs)
    def prep(img: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(img.convert("L"))

    img = prep(pil)

    # normalize coordinates + base padding
    base_pad = 6
    x0, y0, x1, y1 = float(req.x0), float(req.y0), float(req.x1), float(req.y1)
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    x0, y0 = max(0, int(x0 - base_pad)), max(0, int(y0 - base_pad))
    x1, y1 = min(img.width - 1, int(x1 + base_pad)), min(img.height - 1, int(y1 + base_pad))

    # crop
    crop = img.crop((x0, y0, x1, y1))

    # upscale small crops (helps Tesseract a LOT)
    cw, ch = crop.size
    if cw < 140 or ch < 40:
        scale = 3 if max(cw, ch) < 60 else 2
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)

    def ocr_try(psm: int, im: Image.Image) -> str:
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(
            im,
            lang=meta.get("params", {}).get("lang", "eng"),
            config=cfg
        ).strip()

    # score candidates: prefer longer text, reward spaces, penalize non-printables
    def score_text(s: str):
        return (len(s), s.count(" "), -sum(1 for ch in s if not ch.isprintable()))

    # first pass
    cands = [(score_text(t := ocr_try(p, crop)), t) for p in (6, 7, 11)]
    cands.sort(reverse=True)
    best = cands[0][1]

    # fallback: if too short, inflate and retry once
    if len(best) <= 2:
        inflate = 18
        X0 = max(0, x0 - inflate)
        Y0 = max(0, y0 - inflate)
        X1 = min(img.width - 1, x1 + inflate)
        Y1 = min(img.height - 1, y1 + inflate)
        bigger = img.crop((X0, Y0, X1, Y1))
        bw, bh = bigger.size
        if bw < 220 or bh < 70:
            bigger = bigger.resize((bw * 2, bh * 2), Image.BICUBIC)
        retry = [(score_text(t := ocr_try(p, bigger)), t) for p in (6, 7, 11)]
        retry.sort(reverse=True)
        if len(retry[0][1]) > len(best):
            best = retry[0][1]

    return {"text": best}

@router.post("/audit")
async def audit(payload: Dict[str, Any]):
    with (OUT / "audit.log").open("a") as f: f.write(json.dumps(payload) + "\n")
    return {"ok": True}

# ---- PROM / Fields / ECM ----
@router.get("/prom/{doctype}", response_model=PromCatalog)
async def get_prom(doctype: str):
    p = prom_path(doctype)
    if not p.exists(): raise HTTPException(404, "PROM catalog not found")
    return PromCatalog(**json.loads(p.read_text()))

@router.get("/prom")
async def list_proms():
    items = []
    for p in PROM.glob("*.json"):
        try:
            obj = json.loads(p.read_text())
            items.append({"doctype": obj.get("doctype", p.stem), "file": p.name})
        except Exception:
            pass
    return {"doctypes": items}

@router.post("/doc/{doc_id}/doctype")
async def set_doctype(doc_id: str, req: SetDocTypeReq):
    ensure_field_state(doc_id, req.doctype)  # creates if missing
    return {"ok": True}

@router.post("/doc")  # convenience: accepts {doc_id, doctype}
async def set_doctype_short(req: SetDocTypeShort):
    ensure_field_state(req.doc_id, req.doctype)
    return {"ok": True}

@router.get("/doc/{doc_id}/fields", response_model=FieldDocState)
async def get_fields(doc_id: str):
    p = field_state_path(doc_id)
    if not p.exists(): raise HTTPException(404, "Fields not initialized")
    return FieldDocState(**json.loads(p.read_text()))

@router.put("/doc/{doc_id}/fields", response_model=FieldDocState)
async def put_fields(doc_id: str, state: FieldDocState):
    field_state_path(doc_id).write_text(state.model_dump_json(indent=2))
    return state

@router.post("/ecm/extract", response_model=FieldDocState)
async def ecm_extract(req: ECMExtractReq):
    # auto-init if missing
    state = ensure_field_state(req.doc_id, req.doctype)
    ecm_fields = await _mock_ecm(req.doc_id, req.doctype)
    tokens = load_boxes(req.doc_id)

    def infer_bbox(val: str):
        if not val or not tokens: return None
        v = str(val).lower()
        best, best_s = None, 0.0
        for t in tokens:
            txt = (t.text or "").lower() if t.text else ""
            if not txt: continue
            s = difflib.SequenceMatcher(None, v, txt).ratio()
            if s > best_s: best, best_s = t, s
        if best and best_s >= 0.55:
            return {"page": best.page, "x0": best.x0, "y0": best.y0, "x1": best.x1, "y1": best.y1}
        return None

    by_key = {f.key: f for f in state.fields}
    for ef in ecm_fields:
        key = ef.get("key"); 
        if not key: continue
        fs = by_key.get(key) or FieldState(key=key); by_key[key] = fs
        fs.value = ef.get("value")
        fs.confidence = float(ef.get("confidence", 0.0))
        fs.source = "ecm"
        fs.bbox = ef.get("bbox") or infer_bbox(fs.value)

    state.fields = list(by_key.values())
    state.audit.append({"ts": int(time.time()), "event":"ecm_merge", "count": len(ecm_fields)})
    field_state_path(req.doc_id).write_text(state.model_dump_json(indent=2))
    return state

@router.post("/doc/{doc_id}/bind", response_model=FieldDocState)
async def bind_field(doc_id: str, req: BindReq):
    p = field_state_path(doc_id)
    if not p.exists(): raise HTTPException(404, "Fields not initialized")
    state = FieldDocState(**json.loads(p.read_text()))
    target = next((f for f in state.fields if f.key == req.key), None)
    if not target: raise HTTPException(404, f"Field '{req.key}' not found")

    meta = json.loads(meta_path(doc_id).read_text())
    dpi = meta.get("params", {}).get("dpi", 220)
    pil = pdfium.PdfDocument(str(pdf_path(doc_id)))[req.page-1].render(scale=(dpi/72)).to_pil()
    img = preprocess_pil(pil, OCRParams(**meta.get("params", {})))
    x0,y0,x1,y1 = map(int,[req.rect.x0,req.rect.y0,req.rect.x1,req.rect.y1])
    x0,y0 = max(0,x0), max(0,y0)
    x1,y1 = min(img.width-1,x1), min(img.height-1,y1)
    crop = img.crop((x0,y0,x1,y1))
    text = pytesseract.image_to_string(crop, lang=meta.get("params",{}).get("lang","eng"), config="--oem 1 --psm 6").strip()

    target.value = text
    target.source = "ocr"
    target.confidence = 0.80 if text else 0.0
    target.bbox = {"page": req.page, "x0": req.rect.x0, "y0": req.rect.y0, "x1": req.rect.x1, "y1": req.rect.y1}

    state.audit.append({"ts": int(time.time()), "event":"bind", "key": req.key, "value": text})
    field_state_path(doc_id).write_text(state.model_dump_json(indent=2))
    return state

# ---- Mock ECM ----
async def _mock_ecm(doc_id: str, doctype: str):
    if doctype == "invoice":
        return [
            {"key":"partner_name", "value":"Acme Supply Co.", "confidence":0.89, "bbox": None},
            {"key":"invoice_number", "value":"INV-2025-0904-17", "confidence":0.94, "bbox": None},
            {"key":"city", "value":"Bangalore", "confidence":0.72, "bbox": None},
        ]
    else:
        return [
            {"key":"customer_name", "value":"Initech LLC", "confidence":0.92, "bbox": None},
            {"key":"ownership_type", "value":"Leased", "confidence":0.88, "bbox": None},
            {"key":"city", "value":"Chennai", "confidence":0.72, "bbox": None},
        ]

