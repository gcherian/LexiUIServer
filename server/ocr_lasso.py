from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from uuid import uuid4
import shutil, json, time, difflib, re

# Imaging / OCR
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

# Optional: embeddings for semantic search
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # fail gracefully

# Optional: RDF (tiny FIBO-lite KG)
try:
    from rdflib import Graph, Namespace, Literal, RDF, URIRef
except Exception:
    Graph = None  # KG becomes a no-op

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/lasso", tags=["lasso"])

# -----------------------------------------------------------------------------
# Globals (initialized once from main.py via init_paths_for_src)
# -----------------------------------------------------------------------------
SRC_DIR: Path = Path(".")
PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")   # project_root / "data"
PROM: Path = Path(".")   # project_root / "prom"
PUBLIC_BASE: str = "http://localhost:8000"

_EMB_MODEL = None  # sentence-transformers model (lazy)

def init_paths_for_src(src_dir: Path, public_base: str = "http://localhost:8000"):
    """
    Call this ONCE from src/main.py with: init_paths_for_src(Path(__file__).resolve().parent)
    It sets:
      PROJECT_ROOT = src_dir.parent
      DATA = PROJECT_ROOT / "data"
      PROM = PROJECT_ROOT / "prom"
    and ensures DATA/PROM exist.
    """
    global SRC_DIR, PROJECT_ROOT, DATA, PROM, PUBLIC_BASE, _EMB_MODEL
    SRC_DIR = src_dir.resolve()
    PROJECT_ROOT = SRC_DIR.parent
    PUBLIC_BASE = public_base.rstrip("/")

    DATA = (PROJECT_ROOT / "data").resolve(); DATA.mkdir(parents=True, exist_ok=True)
    PROM = (PROJECT_ROOT / "prom").resolve(); PROM.mkdir(parents=True, exist_ok=True)

    # Lazy-load embeddings model if available
    if SentenceTransformer is not None and _EMB_MODEL is None:
        try:
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB_MODEL = None

    print(f"[ocr_lasso] SRC_DIR={SRC_DIR}")
    print(f"[ocr_lasso] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[ocr_lasso] DATA={DATA}")
    print(f"[ocr_lasso] PROM={PROM}")
    print(f"[ocr_lasso] embeddings={'on' if _EMB_MODEL else 'off'}")

# -----------------------------------------------------------------------------
# File layout helpers: EVERYTHING lives under data/<doc_id>/
# -----------------------------------------------------------------------------
def doc_dir(doc_id: str) -> Path:
    d = DATA / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def pdf_path(doc_id: str) -> Path:       return doc_dir(doc_id) / "original.pdf"
def meta_path(doc_id: str) -> Path:      return doc_dir(doc_id) / "meta.json"
def boxes_path(doc_id: str) -> Path:     return doc_dir(doc_id) / "boxes.json"
def page_text_path(doc_id: str) -> Path: return doc_dir(doc_id) / "pages.txt"
def emb_path(doc_id: str) -> Path:       return doc_dir(doc_id) / "embeddings.npz"
def fields_path(doc_id: str) -> Path:    return doc_dir(doc_id) / "fields.json"
def kg_path(doc_id: str) -> Path:        return doc_dir(doc_id) / "kg.ttl"
def audit_log_path(doc_id: str) -> Path: return doc_dir(doc_id) / "audit.log"

# Parse doc_id from a /data/{doc_id}/original.pdf style URL (or raw doc_id)
_DOC_URL_RE = re.compile(r"/data/([^/]+)/original\.pdf$", re.IGNORECASE)
def doc_id_from_doc_url(doc_url: str) -> Optional[str]:
    if not doc_url:
        return None
    # raw doc_id support
    if (DATA / doc_url / "original.pdf").exists():
        return doc_url
    m = _DOC_URL_RE.search(doc_url)
    if m:
        did = m.group(1)
        if pdf_path(did).exists():
            return did
    # also accept bare ID-like strings (hex-ish)
    if re.fullmatch(r"[A-Za-z0-9_-]{6,32}", doc_url):
        return doc_url
    return None

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
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
    id: Optional[str] = None
    label: Optional[str] = None
    text: Optional[str] = None
    confidence: Optional[float] = None

class UploadResp(BaseModel):
    doc_id: str
    annotated_tokens_url: str
    pages: int

class MetaResp(BaseModel):
    pages: List[Dict[str, float]]

class SearchReq(BaseModel):
    doc_id: str; query: str; topk: int = 20

class LassoReq(Rect):
    doc_id: str; page: int

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
    value: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None  # {page,x0,y0,x1,y1}
    source: str = "user"  # ecm|ocr|user|llm
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

class BindReq(BaseModel):
    doc_id: str
    page: int
    rect: Rect
    key: str

# -----------------------------------------------------------------------------
# OCR helpers
# -----------------------------------------------------------------------------
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
    return pytesseract.image_to_data(img, lang=params.lang, config=cfg, output_type=pytesseract.Output.DICT)

def run_full_ocr(doc_id: str, params: OCRParams) -> MetaResp:
    pdf = pdf_path(doc_id)
    all_boxes: List[Dict[str, Any]] = []
    pages_meta: List[Dict[str, float]] = []
    all_page_texts: List[str] = []

    for page_no, pil, w, h in render_pdf_pages(pdf, params.dpi):
        img = preprocess_pil(pil, params)
        d = tesseract_image_to_data(img, params)
        tokens_this_page: List[str] = []
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            if not txt: continue
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            # Stored in image pixel space (origin top-left). Keep consistent everywhere.
            all_boxes.append({"page":page_no,"x0":float(x),"y0":float(y),"x1":float(x+ww),"y1":float(y+hh),"text":txt})
            tokens_this_page.append(txt)
        pages_meta.append({"page":page_no,"width":float(w),"height":float(h)})
        all_page_texts.append(" ".join(tokens_this_page))

    meta_path(doc_id).write_text(json.dumps({
      "pages": pages_meta,
      "params": params.model_dump(),
      "coord_space": {"origin": "top-left", "units": "px@dpi", "dpi": params.dpi}
    }, indent=2))
    boxes_path(doc_id).write_text(json.dumps(all_boxes))
    page_text_path(doc_id).write_text("\n\n".join(all_page_texts))
    build_embeddings(doc_id)
    return MetaResp(pages=pages_meta)

# -----------------------------------------------------------------------------
# Embeddings / semantic search (optional)
# -----------------------------------------------------------------------------
def build_embeddings(doc_id: str):
    if _EMB_MODEL is None:
        return
    txt_file = page_text_path(doc_id)
    if not txt_file.exists():
        return
    pages = txt_file.read_text().split("\n\n")
    embs = _EMB_MODEL.encode(pages, normalize_embeddings=True)
    np.savez_compressed(emb_path(doc_id), embs=embs)

def semantic_search_pages(doc_id: str, query: str, topk: int = 5):
    if _EMB_MODEL is None or not emb_path(doc_id).exists():
        return []
    data = np.load(emb_path(doc_id))
    embs = data["embs"]
    q = _EMB_MODEL.encode([query], normalize_embeddings=True)[0]
    sims = embs @ q
    idx = np.argsort(-sims)[:topk]
    return [(int(i+1), float(sims[i])) for i in idx]

# -----------------------------------------------------------------------------
# Tiny FIBO-lite KG (optional)
# -----------------------------------------------------------------------------
FIBO = None
EDIP = None
if Graph is not None:
    FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
    EDIP = Namespace("https://example.com/edip/")

def build_kg_from_fields(doc_id: str, doctype: Optional[str]):
    if Graph is None:
        return
    g = Graph()
    g.bind("fibo", FIBO); g.bind("edip", EDIP)
    doc_uri = URIRef(f"{EDIP}doc/{doc_id}")
    g.add((doc_uri, RDF.type, EDIP.Document))
    fp = fields_path(doc_id)
    if not fp.exists():
        g.add((doc_uri, EDIP.status, Literal("no_fields")))
        g.serialize(destination=str(kg_path(doc_id)), format="turtle"); return
    state = json.loads(fp.read_text())
    g.add((doc_uri, EDIP.doctype, Literal(state.get("doctype","unknown"))))
    for f in state.get("fields", []):
        key, val = f.get("key"), f.get("value")
        if key and val:
            g.add((doc_uri, URIRef(f"{EDIP}{key}"), Literal(val)))
    g.serialize(destination=str(kg_path(doc_id)), format="turtle")

# -----------------------------------------------------------------------------
# PROM helpers
# -----------------------------------------------------------------------------
def assert_prom_ready():
    if not PROM.exists():
        raise HTTPException(404, f"PROM directory not found at {PROM}")

# -----------------------------------------------------------------------------
# Base endpoints (under /lasso)
# -----------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"ok": True, "svc": "ocr_lasso", "embeddings": bool(_EMB_MODEL), "DATA": str(DATA)}

@router.get("/debug/paths")
def debug_paths(request: Request):
    return {
        "src_dir": str(SRC_DIR),
        "project_root": str(PROJECT_ROOT),
        "data": str(DATA),
        "prom": str(PROM),
        "example_pdf_url": f"{str(request.base_url).rstrip('/')}/data/DEMO/original.pdf"
    }

# ---- Upload ----
@router.post("/upload", response_model=UploadResp)
async def upload(request: Request, pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    doc_id = uuid4().hex[:12]
    p = pdf_path(doc_id)
    with p.open("wb") as f: shutil.copyfileobj(pdf.file, f)
    params = OCRParams()
    run_full_ocr(doc_id, params)
    pages = json.loads(meta_path(doc_id).read_text())["pages"]
    base = str(request.base_url).rstrip("/")
    return UploadResp(
        doc_id=doc_id,
        annotated_tokens_url=f"{base}/data/{doc_id}/original.pdf",
        pages=len(pages)
    )

# ---- Meta / Boxes ----
@router.get("/doc/{doc_id}/meta", response_model=MetaResp)
async def get_meta(doc_id: str):
    mp = meta_path(doc_id)
    if not mp.exists(): raise HTTPException(404, "Meta not found")
    return MetaResp(pages=json.loads(mp.read_text())["pages"])


@router.get("/doc/{doc_id}/boxes", response_model=List[Box])
async def get_boxes(doc_id: str, page: Optional[int] = Query(None)):
    bp = boxes_path(doc_id)
    if not bp.exists():
        return []
    items = [Box(**b) for b in json.loads(bp.read_text())]
    if page is not None:
        items = [b for b in items if b.page == page]
    return items

# ---- Token search (fuzzy) ----
@router.post("/search")
async def token_search(req: SearchReq):
    bp = boxes_path(req.doc_id)
    if not bp.exists(): return {"matches": []}
    tokens = json.loads(bp.read_text())
    q = req.query.strip().lower()
    if not q: return {"matches": []}
    hits = []
    for t in tokens:
        txt = (t.get("text") or "").lower()
        if not txt: continue
        score = difflib.SequenceMatcher(None, q, txt).ratio()
        if score > 0.45:
            hits.append({
                "page": int(t["page"]),
                "bbox": {k: float(t[k]) for k in ("x0","y0","x1","y1")},
                "text": t.get("text",""),
                "score": round(score,3),
            })
    hits.sort(key=lambda m: m["score"], reverse=True)
    return {"matches": hits[:req.topk]}

# ---- Semantic search (optional) ----
@router.get("/semantic_search")
async def semantic_search(doc_id: str = Query(...), q: str = Query(...), topk: int = 5):
    res = semantic_search_pages(doc_id, q, topk)
    return {"results": [{"page": p, "score": s} for p, s in res]}

# ---- Robust lasso OCR ----
@router.post("/lasso")
async def lasso_crop(req: LassoReq):
    mp = meta_path(req.doc_id)
    if not mp.exists(): raise HTTPException(404, "Meta missing")
    meta = json.loads(mp.read_text())
    dpi = meta.get("params", {}).get("dpi", 260)

    pdf = pdfium.PdfDocument(str(pdf_path(req.doc_id)))
    if req.page < 1 or req.page > len(pdf):
        raise HTTPException(400, f"Page out of range: {req.page}")
    pil = pdf[req.page - 1].render(scale=(dpi / 72)).to_pil()

    def prep(img: Image.Image) -> Image.Image:
        return ImageOps.autocontrast(img.convert("L"))

    img = prep(pil)

    base_pad = 6
    x0, y0, x1, y1 = float(req.x0), float(req.y0), float(req.x1), float(req.y1)
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    x0, y0 = max(0, int(x0 - base_pad)), max(0, int(y0 - base_pad))
    x1, y1 = min(img.width - 1, int(x1 + base_pad)), min(img.height - 1, int(y1 + base_pad))

    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    if cw < 180 or ch < 52:
        scale = 3 if max(cw, ch) < 70 else 2
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)

    def ocr_try(psm: int, im: Image.Image) -> str:
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(
            im, lang=meta.get("params",{}).get("lang","eng"), config=cfg
        ).strip()

    def score_text(s: str): return (len(s), s.count(" "), -sum(1 for ch in s if not ch.isprintable()))

    cands = [(score_text(t := ocr_try(p, crop)), t) for p in (6, 7, 11)]
    cands.sort(reverse=True)
    best = cands[0][1]

    if len(best) <= 2:
        inflate = 18
        X0 = max(0, x0 - inflate); Y0 = max(0, y0 - inflate)
        X1 = min(img.width - 1, x1 + inflate); Y1 = min(img.height - 1, y1 + inflate)
        bigger = img.crop((X0, Y0, X1, Y1))
        bw, bh = bigger.size
        if bw < 220 or bh < 70:
            bigger = bigger.resize((bw * 2, bh * 2), Image.BICUBIC)
        retry = [(score_text(t := ocr_try(p, bigger)), t) for p in (6, 7, 11)]
        retry.sort(reverse=True)
        if len(retry[0][1]) > len(best):
            best = retry[0][1]

    return {"text": best}

# ---- PROM ----
@router.get("/prom")
async def prom_list():
    assert_prom_ready()
    items = []
    for p in PROM.glob("*.json"):
        try:
            obj = json.loads(p.read_text())
            items.append({"doctype": obj.get("doctype", p.stem), "file": p.name})
        except Exception:
            pass
    return {"doctypes": items}

@router.get("/prom/{doctype}", response_model=PromCatalog)
async def prom_get(doctype: str):
    assert_prom_ready()
    p = PROM / f"{doctype}.json"
    if not p.exists():
        raise HTTPException(404, f"PROM catalog not found for '{doctype}'. Available: {[p.stem for p in PROM.glob('*.json')]}")
    return PromCatalog(**json.loads(p.read_text()))

# ---- Fields / ECM / Bind ----
def ensure_field_state(doc_id: str, doctype: str) -> FieldDocState:
    fp = fields_path(doc_id)
    if fp.exists():
        return FieldDocState(**json.loads(fp.read_text()))
    # seed from PROM
    pp = PROM / f"{doctype}.json"
    if not pp.exists():
        raise HTTPException(404, f"PROM not found for '{doctype}'")
    prom = json.loads(pp.read_text())
    fields = [FieldState(key=f["key"]) for f in prom.get("fields",[])]
    state = FieldDocState(doc_id=doc_id, doctype=doctype, fields=fields, audit=[{"ts":int(time.time()),"event":"init","doctype":doctype}])
    fp.write_text(state.model_dump_json(indent=2))
    return state

@router.post("/doc/{doc_id}/doctype")
async def set_doctype(doc_id: str, req: SetDocTypeReq):
    st = ensure_field_state(doc_id, req.doctype)
    build_kg_from_fields(doc_id, req.doctype)
    return {"ok": True, "doctype": st.doctype}

@router.get("/doc/{doc_id}/fields", response_model=FieldDocState)
async def get_fields(doc_id: str):
    fp = fields_path(doc_id)
    if not fp.exists():
        # Return an empty shell instead of 404
        return FieldDocState(doc_id=doc_id, doctype="unknown", fields=[], audit=[])
    return FieldDocState(**json.loads(fp.read_text()))


@router.put("/doc/{doc_id}/fields", response_model=FieldDocState)
async def put_fields(doc_id: str, state: FieldDocState):
    fields_path(doc_id).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(doc_id, state.doctype)
    return state

@router.post("/ecm/extract", response_model=FieldDocState)
async def ecm_extract(req: ECMExtractReq):
    state = ensure_field_state(req.doc_id, req.doctype)
    # mock ECM payload (replace with your ECM call)
    ecm_fields = [
        {"key":"customer_name","value":"Initech LLC","confidence":0.92},
        {"key":"city","value":"Chennai","confidence":0.75},
    ] if req.doctype != "invoice" else [
        {"key":"partner_name","value":"Acme Supply Co.","confidence":0.89},
        {"key":"invoice_number","value":"INV-2025-0905-01","confidence":0.94},
        {"key":"city","value":"Bangalore","confidence":0.70},
    ]
    # merge
    m = {f.key:f for f in state.fields}
    for ef in ecm_fields:
        k = ef["key"]; v = ef.get("value")
        fs = m.get(k) or FieldState(key=k)
        fs.value = v; fs.source = "ecm"; fs.confidence = float(ef.get("confidence",0.0))
        m[k] = fs
    state.fields = list(m.values())
    state.audit.append({"ts":int(time.time()),"event":"ecm_merge","count":len(ecm_fields)})
    fields_path(req.doc_id).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(req.doc_id, state.doctype)
    return state

@router.post("/doc/{doc_id}/bind", response_model=FieldDocState)
async def bind_field(doc_id: str, req: BindReq):
    fp = fields_path(doc_id)
    if not fp.exists(): raise HTTPException(404, "Fields not initialized")
    state = FieldDocState(**json.loads(fp.read_text()))
    target = next((f for f in state.fields if f.key == req.key), None)
    if not target: raise HTTPException(404, f"Field '{req.key}' not found")

    meta = json.loads(meta_path(doc_id).read_text())
    dpi = meta.get("params", {}).get("dpi", 260)
    pil = pdfium.PdfDocument(str(pdf_path(doc_id)))[req.page-1].render(scale=(dpi/72)).to_pil()
    img = ImageOps.autocontrast(pil.convert("L"))
    x0,y0,x1,y1 = map(int,[req.rect.x0,req.rect.y0,req.rect.x1,req.rect.y1])
    if x0 > x1: x0,x1 = x1,x0
    if y0 > y1: y0,y1 = y1,y0
    pad = 6
    x0,y0 = max(0,x0-pad), max(0,y0-pad)
    x1,y1 = min(img.width-1,x1+pad), min(img.height-1,y1+pad)
    crop = img.crop((x0,y0,x1,y1))
    if crop.width < 140 or crop.height < 40:
        scale = 3 if max(crop.width,crop.height) < 60 else 2
        crop = crop.resize((crop.width*scale, crop.height*scale), Image.BICUBIC)

    def ocr(psm:int):
        return pytesseract.image_to_string(
            crop, lang=meta.get("params",{}).get("lang","eng"),
            config=f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        ).strip()

    text = max((ocr(p) for p in (6,7,11)), key=lambda s: (len(s), s.count(" ")), default="")
    target.value = text; target.source = "ocr"; target.confidence = 0.8 if text else 0.0
    target.bbox = {"page": req.page, "x0": req.rect.x0, "y0": req.rect.y0, "x1": req.rect.x1, "y1": req.rect.y1}

    state.audit.append({"ts":int(time.time()),"event":"bind","key":req.key,"value":text})
    fields_path(doc_id).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(doc_id, state.doctype)
    return state

# ---- KG endpoints (optional) ----
@router.get("/doc/{doc_id}/kg")
async def get_kg(doc_id: str):
    if not kg_path(doc_id).exists():
        build_kg_from_fields(doc_id, None)
    return {"ttl_path": f"/data/{doc_id}/kg.ttl"}

# -----------------------------------------------------------------------------
# Frontend-compat endpoints (no /lasso prefix) so the React tab works as-is
# -----------------------------------------------------------------------------
# GET /boxes?doc_url=...&page=... -> array of boxes (x0,y0,x1,y1 in image pixel space; origin top-left)
async def _compat_boxes(doc_url: str, page: int):
    did = doc_id_from_doc_url(doc_url)
    if not did:
        raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    bp = boxes_path(did)
    if not bp.exists(): return []
    all_boxes = json.loads(bp.read_text())
    return [b for b in all_boxes if int(b.get("page", 0)) == int(page)]

# GET /fields?doc_url=... -> FieldDocState.fields (flattened)
async def _compat_fields_get(doc_url: str):
    did = doc_id_from_doc_url(doc_url)
    if not did:
        raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    fp = fields_path(did)
    if not fp.exists():
        # Initialize a minimal empty state if missing
        empty = FieldDocState(doc_id=did, doctype="unknown", fields=[], audit=[{"ts":int(time.time()),"event":"init-empty"}])
        fields_path(did).write_text(empty.model_dump_json(indent=2))
        return []
    state = FieldDocState(**json.loads(fp.read_text()))
    # Return just the fields list for compatibility with the UI client
    return [f.model_dump() for f in state.fields]

# POST /fields { doc_url, field } -> upsert field in FieldDocState
async def _compat_fields_post(doc_url: str, field: Dict[str, Any]):
    did = doc_id_from_doc_url(doc_url)
    if not did:
        raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    fp = fields_path(did)
    if fp.exists():
        state = FieldDocState(**json.loads(fp.read_text()))
    else:
        state = FieldDocState(doc_id=did, doctype="unknown", fields=[], audit=[{"ts":int(time.time()),"event":"init-empty"}])

    # Upsert by (key) if present, else by id, else append
    f_in = field.copy()
    # Normalize bbox numeric
    if isinstance(f_in.get("bbox"), dict):
        for k in ("x0","y0","x1","y1","page"):
            if k in f_in["bbox"]:
                try:
                    f_in["bbox"][k] = float(f_in["bbox"][k]) if k!="page" else int(f_in["bbox"][k])
                except Exception:
                    pass
    key = f_in.get("key") or f_in.get("name")
    if key and "key" not in f_in:
        f_in["key"] = key
    # map 'name' to 'key' for compatibility
    f_id = f_in.get("id")
    updated = False
    if key:
        for i, fx in enumerate(state.fields):
            if fx.key == key:
                # merge
                newf = fx.model_dump()
                newf.update({k:v for k,v in f_in.items() if k in ("key","value","bbox","source","confidence")})
                state.fields[i] = FieldState(**newf)
                updated = True
                break
    if not updated and f_id:
        for i, fx in enumerate(state.fields):
            if getattr(fx, "id", None) == f_id:
                newf = fx.model_dump()
                newf.update({k:v for k,v in f_in.items() if k in ("key","value","bbox","source","confidence")})
                state.fields[i] = FieldState(**newf)
                updated = True
                break
    if not updated:
        # append new field
        if "confidence" not in f_in: f_in["confidence"] = 0.0
        if "source" not in f_in: f_in["source"] = "user"
        state.fields.append(FieldState(**f_in))

    state.audit.append({"ts":int(time.time()),"event":"compat_fields_upsert","key":key or f_id})
    fields_path(did).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(did, state.doctype)
    return {"ok": True}

# FastAPI route wrappers mounted later by helper (so they live at app root)
def mount_compat_endpoints(app: FastAPI):
    @app.get("/boxes")
    async def boxes_compat(doc_url: str = Query(...), page: int = Query(1)):
        return await _compat_boxes(doc_url, page)

    @app.get("/fields")
    async def fields_compat_get(doc_url: str = Query(...)):
        return await _compat_fields_get(doc_url)

    @app.post("/fields")
    async def fields_compat_post(payload: Dict[str, Any]):
        doc_url = payload.get("doc_url")
        field = payload.get("field")
        if not doc_url or not isinstance(field, dict):
            raise HTTPException(400, "Body must include { doc_url, field }")
        return await _compat_fields_post(doc_url, field)

# -----------------------------------------------------------------------------
# Convenience to wire from main.py:
#   app.include_router(ocr_lasso.router)
#   ocr_lasso.init_paths_for_src(Path(__file__).resolve().parent)
#   ocr_lasso.mount_compat_endpoints(app)
# -----------------------------------------------------------------------------
