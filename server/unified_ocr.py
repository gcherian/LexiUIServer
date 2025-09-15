# ocr_unified.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, FastAPI
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

# Optional: Transformers (for BERT/DistilBERT)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

# Optional: TF-IDF
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
except Exception:
    TfidfVectorizer = None
    sk_cosine = None

# -----------------------------------------------------------------------------
# Router (everything lives here)
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/lasso", tags=["lasso"])

# -----------------------------------------------------------------------------
# Paths & globals
# -----------------------------------------------------------------------------
SRC_DIR: Path = Path(".")
PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")   # project_root / "data"
PROM: Path = Path(".")   # project_root / "prom"
PUBLIC_BASE: str = "http://localhost:8000"

_EMB_MODEL = None  # sentence-transformers model (lazy)

def init_paths_for_src(src_dir: Path, public_base: str = "http://localhost:8000"):
    """Call once from main.py"""
    global SRC_DIR, PROJECT_ROOT, DATA, PROM, PUBLIC_BASE, _EMB_MODEL
    SRC_DIR = src_dir.resolve()
    PROJECT_ROOT = SRC_DIR.parent
    PUBLIC_BASE = public_base.rstrip("/")
    DATA = (PROJECT_ROOT / "data").resolve(); DATA.mkdir(parents=True, exist_ok=True)
    PROM = (PROJECT_ROOT / "prom").resolve(); PROM.mkdir(parents=True, exist_ok=True)

    if SentenceTransformer is not None and _EMB_MODEL is None:
        try:
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB_MODEL = None

    print(f"[unified] SRC_DIR={SRC_DIR}")
    print(f"[unified] DATA={DATA}  PROM={PROM}  embeddings={'on' if _EMB_MODEL else 'off'}")

# -----------------------------------------------------------------------------
# File layout helpers: everything under data/<doc_id>/
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

_DOC_URL_RE = re.compile(r"/data/([^/]+)/original\.pdf$", re.IGNORECASE)
def doc_id_from_doc_url(doc_url: str) -> Optional[str]:
    if not doc_url: return None
    if (DATA / doc_url / "original.pdf").exists(): return doc_url
    m = _DOC_URL_RE.search(doc_url)
    if m:
        did = m.group(1)
        if pdf_path(did).exists(): return did
    if re.fullmatch(r"[A-Za-z0-9_-]{6,32}", doc_url): return doc_url
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

class FieldState(BaseModel):
    key: str
    value: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None  # {page,x0,y0,x1,y1}
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
        page = pdf[i]; zoom = dpi / 72
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
    if _EMB_MODEL is None: return
    txt_file = page_text_path(doc_id)
    if not txt_file.exists(): return
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
    if Graph is None: return
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
# Base endpoints (under /lasso)
# -----------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"ok": True, "svc": "ocr_unified", "embeddings": bool(_EMB_MODEL), "DATA": str(DATA)}

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

@router.get("/semantic_search")
async def semantic_search(doc_id: str = Query(...), q: str = Query(...), topk: int = 5):
    res = semantic_search_pages(doc_id, q, topk)
    return {"results": [{"page": p, "score": s} for p, s in res]}

@router.post("/lasso")
async def lasso_crop(req: LassoReq):
    mp = meta_path(req.doc_id)
    if not mp.exists(): raise HTTPException(404, "Meta missing")
    meta = json.loads(mp.read_text())
    dpi = meta.get("params", {}).get("dpi", 260)

    pdf = pdfium.PdfDocument(str(pdf_path(req.doc_id)))
    if req.page < 1 or req.page > len(pdf): raise HTTPException(400, f"Page out of range: {req.page}")

    pil = pdf[req.page - 1].render(scale=(dpi / 72)).to_pil()
    gray = ImageOps.autocontrast(pil.convert("L"))

    x0 = float(min(req.x0, req.x1)); y0 = float(min(req.y0, req.y1))
    x1 = float(max(req.x0, req.x1)); y1 = float(max(req.y0, req.y1))

    pad = 4
    X0 = max(0, int(x0 - pad)); Y0 = max(0, int(y0 - pad))
    X1 = min(gray.width - 1, int(x1 + pad)); Y1 = min(gray.height - 1, int(y1 + pad))

    crop = gray.crop((X0, Y0, X1, Y1))
    if crop.width < 140 or crop.height < 40:
        scale = 3 if max(crop.width, crop.height) < 60 else 2
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)

    def ocr(psm: int) -> str:
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(crop, lang=meta.get("params", {}).get("lang", "eng"), config=cfg).strip()

    cand = sorted(((len(t), t) for t in (ocr(6), ocr(7), ocr(11))), reverse=True)
    best = cand[0][1] if cand else ""

    crop_path = doc_dir(req.doc_id) / "last_crop.png"
    try:
        crop.save(crop_path); crop_url = f"/data/{req.doc_id}/last_crop.png"
    except Exception:
        crop_url = None

    return {"text": best, "rect_used": {"page": req.page, "x0": X0, "y0": Y0, "x1": X1, "y1": Y1},
            "page_size": {"width": gray.width, "height": gray.height}, "crop_url": crop_url}

# ---- PROM / Fields (same as before) ----
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

def assert_prom_ready():
    if not PROM.exists():
        raise HTTPException(404, f"PROM directory not found at {PROM}")

@router.get("/prom")
async def prom_list():
    assert_prom_ready()
    items = []
    for p in PROM.glob("*.json"):
        try:
            obj = json.loads(p.read_text()); items.append({"doctype": obj.get("doctype", p.stem), "file": p.name})
        except Exception: pass
    return {"doctypes": items}

@router.get("/prom/{doctype}", response_model=PromCatalog)
async def prom_get(doctype: str):
    assert_prom_ready()
    p = PROM / f"{doctype}.json"
    if not p.exists():
        raise HTTPException(404, f"PROM catalog not found for '{doctype}'. Available: {[p.stem for p in PROM.glob('*.json')]}")
    return PromCatalog(**json.loads(p.read_text()))

def ensure_field_state(doc_id: str, doctype: str) -> FieldDocState:
    fp = fields_path(doc_id)
    if fp.exists(): return FieldDocState(**json.loads(fp.read_text()))
    pp = PROM / f"{doctype}.json"
    if not pp.exists(): raise HTTPException(404, f"PROM not found for '{doctype}'")
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
        return FieldDocState(doc_id=doc_id, doctype="unknown", fields=[], audit=[])
    return FieldDocState(**json.loads(fp.read_text()))

@router.put("/doc/{doc_id}/fields", response_model=FieldDocState)
async def put_fields(doc_id: str, state: FieldDocState):
    fields_path(doc_id).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(doc_id, state.doctype)
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

@router.get("/doc/{doc_id}/kg")
async def get_kg(doc_id: str):
    if not kg_path(doc_id).exists(): build_kg_from_fields(doc_id, None)
    return {"ttl_path": f"/data/{doc_id}/kg.ttl"}

# -----------------------------------------------------------------------------
# DistilBERT Reranker (shared)
# -----------------------------------------------------------------------------
_RERANK = {"tok": None, "mdl": None, "device": "cpu"}  # DistilBERT

def _ensure_reranker(model_name_or_path: str = "distilbert-base-uncased"):
    if AutoTokenizer is None or AutoModel is None:
        raise HTTPException(500, "Transformers not available on server.")
    if _RERANK["mdl"] is not None: return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModel.from_pretrained(model_name_or_path)
    mdl.to(dev); mdl.eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
    print(f"[unified] DistilBERT reranker loaded on {dev} ({model_name_or_path})")

def _embed(texts: List[str]):
    tok = _RERANK["tok"]; mdl = _RERANK["mdl"]; dev = _RERANK["device"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(1); counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        return torch.nn.functional.normalize(emb, p=2, dim=1)

def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _union_rect(span):
    x0 = min(t["x0"] for t in span); y0 = min(t["y0"] for t in span)
    x1 = max(t["x1"] for t in span); y1 = max(t["y1"] for t in span)
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}

def _group_tokens_by_page(tokens):
    by = {}
    for t in tokens:
        by.setdefault(int(t["page"]), []).append(t)
    for arr in by.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))
    return by

def _concat_text(span):
    return _norm_text(" ".join(t.get("text","") for t in span if t.get("text")))

def _find_candidate_spans(value: str, tokens_page):
    v = _norm_text(value)
    out = []; n = len(tokens_page)
    for i in range(n):
        accum = []; text = ""
        for w in range(12):
            j = i + w
            if j >= n: break
            tok = tokens_page[j]
            t = _norm_text(tok.get("text",""))
            if not t: continue
            accum.append(tok)
            text = _concat_text(accum)
            if v and v in text:
                out.append({"span": list(accum), "rect": _union_rect(accum), "text": text})
                if w >= 3: break
    dedup = []; seen = set()
    for c in out:
        r = c["rect"]; key = (round(r["x0"]/5), round(r["y0"]/5), round((r["x1"]-r["x0"])/5), round((r["y1"]-r["y0"])/5))
        if key in seen: continue
        seen.add(key); dedup.append(c)
    return dedup

def _context_snippet(tokens_page, span, px_margin=120, py_margin=35):
    R = _union_rect(span)
    cx0 = R["x0"] - px_margin; cy0 = R["y0"] - py_margin
    cx1 = R["x1"] + px_margin; cy1 = R["y1"] + py_margin
    bag = [t for t in tokens_page if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _concat_text(bag)

# -----------------------------------------------------------------------------
# /lasso/rerank (kept from your version)
# -----------------------------------------------------------------------------
class RerankReq(BaseModel):
    doc_id: str
    key: str
    value: str
    model: Optional[str] = None
    topk: int = 8

@router.post("/rerank")
async def rerank(req: RerankReq):
    if AutoTokenizer is None: raise HTTPException(500, "Transformers not installed on server.")
    _ensure_reranker(req.model or "distilbert-base-uncased")
    bp = boxes_path(req.doc_id); mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists(): raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text()); by_pg = _group_tokens_by_page(tokens)

    vnorm = _norm_text(req.value); cands = []
    for pg, toks in by_pg.items():
        spans = _find_candidate_spans(vnorm, toks)
        for s in spans:
            local = _context_snippet(toks, s["span"])
            cands.append({"page": pg, "rect": s["rect"], "span_text": s["text"], "context": local})
    if not cands: return {"best": None, "alts": []}

    query = f"{req.key}: {req.value}".strip()
    texts = [query] + [c["context"] for c in cands]
    embs = _embed(texts); q = embs[0:1]; M = embs[1:]
    sims = (M @ q.t()).squeeze(1)

    scored = sorted(
        [{"score": float(sims[i].item()), **cands[i]} for i in range(len(cands))],
        key=lambda x: x["score"], reverse=True
    )
    best = scored[0]; alts = scored[1: min(len(scored), req.topk)]
    return {"best": {"page": best["page"], "rect": best["rect"], "score": best["score"]},
            "alts": [{"page": a["page"], "rect": a["rect"], "score": a["score"]} for a in alts]}

# -----------------------------------------------------------------------------
# MATCH: /lasso/match/field  (Autolocate + TF-IDF + BERT)
# -----------------------------------------------------------------------------
class _OCRToken(BaseModel):
    text: str
    bbox: List[float]
    page: int = 1

class _MatchReq(BaseModel):
    ocr_tokens: List[_OCRToken]
    key: str
    field: str
    llm_value: str
    use_bert: bool = True
    bert_model: Optional[str] = "distilbert-base-uncased"

class _MethodBox(BaseModel):
    text: Optional[str] = None
    conf: Optional[float] = None
    bbox: Optional[List[float]] = None
    page: Optional[int] = None

class _MatchResp(BaseModel):
    field: str
    llm_value: str
    autolocate: Optional[_MethodBox] = None
    bert: Optional[_MethodBox] = None
    tfidf: Optional[_MethodBox] = None

def _n(s: str) -> str: return " ".join((s or "").lower().strip().split())
def _lev_ratio(a: str, b: str) -> float: return difflib.SequenceMatcher(None, _n(a), _n(b)).ratio()

def _tfidf_best(tokens: List[Dict[str, Any]], key: str, llm: str):
    corpus = [_n(t.get("text","")) for t in tokens]
    if not corpus: return None
    q = ((_n(key) + " ") * 7 + (_n(llm) + " ") * 3).strip()
    if TfidfVectorizer and sk_cosine:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(corpus)
        qX = vec.transform([q])
        sims = sk_cosine(qX, X).ravel()
    else:
        from collections import Counter
        def bow(s): return Counter([w for w in s.split() if w])
        def cos(a,b):
            if not a or not b: return 0.0
            keys = set(a)|set(b)
            va = np.array([a.get(k,0.0) for k in keys]); vb = np.array([b.get(k,0.0) for k in keys])
            na = np.linalg.norm(va); nb = np.linalg.norm(vb)
            return 0.0 if na==0 or nb==0 else float(va @ vb) / float(na*nb)
        bag = [bow(t) for t in corpus]; qbag = bow(q)
        sims = np.array([cos(qbag, b) for b in bag], dtype=float)
    boost = np.zeros_like(sims)
    lv = _n(llm); lv_len = len(lv)
    for i, t in enumerate(tokens):
        txt = _n(t.get("text",""))
        if lv and txt == lv: boost[i] += 0.15
        if lv_len>0 and len(txt) > lv_len+3: boost[i] -= 0.05
        k = _n(key); if k and k in txt: boost[i] += 0.03
    sc = sims + boost
    ti = int(sc.argmax()) if len(sc) else -1
    if ti < 0: return None
    tt = tokens[ti]
    return {"text": tt.get("text",""), "conf": round(float(sc[ti]),4),
            "bbox": [float(tt["x0"]), float(tt["y0"]), float(tt["x1"]), float(tt["y1"])],
            "page": int(tt.get("page",1))}

def _match_core(tokens_in: List[Dict[str, Any]], key: str, field: str, llm_value: str, use_bert: bool, bert_model: Optional[str]):
    tokens = [{"text": t.get("text",""),
               "x0": float(t["bbox"][0]) if "bbox" in t else float(t["x0"]),
               "y0": float(t["bbox"][1]) if "bbox" in t else float(t["y0"]),
               "x1": float(t["bbox"][2]) if "bbox" in t else float(t["x1"]),
               "y1": float(t["bbox"][3]) if "bbox" in t else float(t["y1"]),
               "page": int(t.get("page",1))}
              for t in tokens_in]

    # 1) Autolocate (max fuzzy)
    best_i, best_s = -1, -1.0
    for i, t in enumerate(tokens):
        s = _lev_ratio(t["text"], llm_value)
        if s > best_s: best_s, best_i = s, i
    auto = None
    if best_i >= 0:
        tt = tokens[best_i]
        auto = {"text": tt["text"], "conf": round(float(best_s),4),
                "bbox": [tt["x0"], tt["y0"], tt["x1"], tt["y1"]], "page": tt["page"]}

    # 2) TF-IDF (key+value)
    tfidf = _tfidf_best(tokens, key, llm_value)

    # 3) BERT span rerank around llm_value
    bert = None
    if use_bert and AutoTokenizer is not None and llm_value:
        try:
            _ensure_reranker(bert_model or "distilbert-base-uncased")
            by_pg = _group_tokens_by_page(tokens)
            vnorm = _norm_text(llm_value)
            best = None
            for pg, toks in by_pg.items():
                spans = _find_candidate_spans(vnorm, toks)
                if not spans: continue
                cands = [{"page": pg, "rect": s["rect"], "text": s["text"], "ctx": _context_snippet(toks, s["span"])} for s in spans]
                query = f"{key}: {llm_value}".strip()
                texts = [query] + [c["ctx"] for c in cands]
                embs = _embed(texts); qv, M = embs[0:1], embs[1:]
                sims = (M @ qv.T).squeeze(1)
                bi = int(torch.argmax(sims).item())
                cand = {**cands[bi], "score": float(sims[bi].item())}
                if best is None or cand["score"] > best["score"]: best = cand
            if best:
                bert = {"text": best["text"], "conf": round(float(best["score"]),4),
                        "bbox": [best["rect"]["x0"], best["rect"]["y0"], best["rect"]["x1"], best["rect"]["y1"]],
                        "page": int(best["page"])}
        except Exception as e:
            print(f"[unified] BERT disabled: {e}")

    return {"field": field, "llm_value": llm_value, "autolocate": auto, "bert": bert, "tfidf": tfidf}

@router.post("/match/field", response_model=_MatchResp)
async def match_field(req: _MatchReq):
    tokens = [t.model_dump() for t in req.ocr_tokens]
    out = _match_core(tokens, req.key, req.field, req.llm_value, req.use_bert, req.bert_model)
    return _MatchResp(**out)

# -----------------------------------------------------------------------------
# DISTIL EXTRACT (minimal): /lasso/distil/extract  and root alias
# -----------------------------------------------------------------------------
class _DistilSpec(BaseModel):
    key: str
    label: str
    type: Optional[str] = "text"

class _DistilExtractReq(BaseModel):
    doc_id: str
    fields: List[_DistilSpec]
    max_window: int = 12
    dpi: int = 260

class _DistilBox(BaseModel):
    page: int; x0: float; y0: float; x1: float; y1: float

class _DistilField(BaseModel):
    key: str; label: str; type: Optional[str] = "text"
    page: Optional[int] = None
    key_box: Optional[_DistilBox] = None
    value: str = ""
    value_boxes: List[_DistilBox] = []
    value_union: Optional[_DistilBox] = None
    confidence: float = 0.0

class _DistilResp(BaseModel):
    doc_id: str
    fields: List[_DistilField]
    dpi: int

@router.post("/distil/extract", response_model=_DistilResp)
async def distil_extract(req: _DistilExtractReq):
    bp = boxes_path(req.doc_id); mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists(): raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text())
    by_pg = _group_tokens_by_page(tokens)

    out: List[_DistilField] = []
    for spec in req.fields:
        # best effort: try BERT first (if available), else fall back to autolocate
        if AutoTokenizer is not None:
            try:
                _ensure_reranker("distilbert-base-uncased")
                v = spec.label
                vnorm = _norm_text(v)
                best = None
                for pg, toks in by_pg.items():
                    spans = _find_candidate_spans(vnorm, toks)
                    if not spans: continue
                    cands = [{"page": pg, "rect": s["rect"], "text": s["text"], "ctx": _context_snippet(toks, s["span"])} for s in spans]
                    query = spec.label
                    texts = [query] + [c["ctx"] for c in cands]
                    embs = _embed(texts); qv, M = embs[0:1], embs[1:]
                    sims = (M @ qv.T).squeeze(1)
                    bi = int(torch.argmax(sims).item())
                    cand = {**cands[bi], "score": float(sims[bi].item())}
                    if best is None or cand["score"] > best["score"]: best = cand
                if best:
                    b = _DistilBox(page=int(best["page"]), x0=float(best["rect"]["x0"]), y0=float(best["rect"]["y0"]), x1=float(best["rect"]["x1"]), y1=float(best["rect"]["y1"]))
                    out.append(_DistilField(key=spec.key, label=spec.label, type=spec.type, page=b.page, value=spec.label, value_boxes=[b], value_union=b, confidence=float(best["score"])))
                    continue
            except Exception as e:
                print(f"[unified] distil (bert) disabled: {e}")

        # fallback: autolocate by fuzzy match on tokens
        best_i, best_s = -1, -1.0
        for i, t in enumerate(tokens):
            s = difflib.SequenceMatcher(None, _norm_text(t.get("text","")), _norm_text(spec.label)).ratio()
            if s > best_s: best_s, best_i = s, i
        if best_i >= 0:
            tt = tokens[best_i]
            b = _DistilBox(page=int(tt["page"]), x0=float(tt["x0"]), y0=float(tt["y0"]), x1=float(tt["x1"]), y1=float(tt["y1"]))
            out.append(_DistilField(key=spec.key, label=spec.label, type=spec.type, page=b.page, value=spec.label, value_boxes=[b], value_union=b, confidence=float(best_s)))
        else:
            out.append(_DistilField(key=spec.key, label=spec.label, type=spec.type, page=None, value="", value_boxes=[], value_union=None, confidence=0.0))

    meta = json.loads(mp.read_text()); dpi = meta.get("params",{}).get("dpi", req.dpi)
    return _DistilResp(doc_id=req.doc_id, fields=out, dpi=int(dpi))

# -----------------------------------------------------------------------------
# Frontend-compat endpoints mounted at app root (/boxes, /fields, /match/field, /distil/extract)
# -----------------------------------------------------------------------------
async def _compat_boxes(doc_url: str, page: int):
    did = doc_id_from_doc_url(doc_url)
    if not did: raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    bp = boxes_path(did); 
    if not bp.exists(): return []
    return [b for b in json.loads(bp.read_text()) if int(b.get("page",0)) == int(page)]

async def _compat_fields_get(doc_url: str):
    did = doc_id_from_doc_url(doc_url)
    if not did: raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    fp = fields_path(did)
    if not fp.exists():
        empty = FieldDocState(doc_id=did, doctype="unknown", fields=[], audit=[{"ts":int(time.time()),"event":"init-empty"}])
        fields_path(did).write_text(empty.model_dump_json(indent=2))
        return []
    state = FieldDocState(**json.loads(fp.read_text()))
    return [f.model_dump() for f in state.fields]

async def _compat_fields_post(doc_url: str, field: Dict[str, Any]):
    did = doc_id_from_doc_url(doc_url)
    if not did: raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    fp = fields_path(did)
    state = FieldDocState(**json.loads(fp.read_text())) if fp.exists() else FieldDocState(doc_id=did, doctype="unknown", fields=[], audit=[{"ts":int(time.time()),"event":"init-empty"}])
    f_in = field.copy()
    if isinstance(f_in.get("bbox"), dict):
        for k in ("x0","y0","x1","y1","page"):
            if k in f_in["bbox"]:
                try: f_in["bbox"][k] = float(f_in["bbox"][k]) if k!="page" else int(f_in["bbox"][k])
                except Exception: pass
    key = f_in.get("key") or f_in.get("name")
    if key and "key" not in f_in: f_in["key"] = key
    f_id = f_in.get("id"); updated = False
    if key:
        for i, fx in enumerate(state.fields):
            if fx.key == key:
                newf = fx.model_dump()
                for k in ("key","value","bbox","source","confidence"):
                    if k in f_in: newf[k] = f_in[k]
                state.fields[i] = FieldState(**newf); updated = True; break
    if not updated and f_id:
        for i, fx in enumerate(state.fields):
            if getattr(fx, "id", None) == f_id:
                newf = fx.model_dump()
                for k in ("key","value","bbox","source","confidence"):
                    if k in f_in: newf[k] = f_in[k]
                state.fields[i] = FieldState(**newf); updated = True; break
    if not updated:
        if "confidence" not in f_in: f_in["confidence"] = 0.0
        if "source" not in f_in: f_in["source"] = "user"
        state.fields.append(FieldState(**f_in))
    state.audit.append({"ts":int(time.time()),"event":"compat_fields_upsert","key":key or f_id})
    fields_path(did).write_text(state.model_dump_json(indent=2))
    build_kg_from_fields(did, state.doctype)
    return {"ok": True}

def mount_compat_endpoints(app: FastAPI):
    """Mount root-level shims so the React app can call without /lasso prefix."""
    @app.get("/boxes")
    async def boxes_compat(doc_url: str = Query(...), page: int = Query(1)):
        return await _compat_boxes(doc_url, page)

    @app.get("/fields")
    async def fields_compat_get(doc_url: str = Query(...)):
        return await _compat_fields_get(doc_url)

    @app.post("/fields")
    async def fields_compat_post(payload: Dict[str, Any]):
        doc_url = payload.get("doc_url"); field = payload.get("field")
        if not doc_url or not isinstance(field, dict): raise HTTPException(400, "Body must include { doc_url, field }")
        return await _compat_fields_post(doc_url, field)

    # Root alias for /lasso/match/field
    @app.post("/match/field")
    async def match_field_alias(payload: Dict[str, Any]):
        req = _MatchReq(**payload)
        return await match_field(req)

    # Root alias for /lasso/distil/extract
    @app.post("/distil/extract")
    async def distil_extract_alias(payload: Dict[str, Any]):
        req = _DistilExtractReq(**payload)
        return await distil_extract(req)