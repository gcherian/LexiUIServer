from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request, FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from uuid import uuid4
import shutil, json, time, difflib, re, os

# Imaging / OCR
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

# Optional: embeddings for per-page semantic search (MiniLM all-MiniLM-L6-v2)
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

# Transformers local-only (DistilBERT + optional LayoutLMv3)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoProcessor, LayoutLMv3Model
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None
    AutoProcessor = None
    LayoutLMv3Model = None

# ML utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz as _rfuzz

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/lasso", tags=["lasso"])

# -----------------------------------------------------------------------------
# Globals / Paths
# -----------------------------------------------------------------------------
SRC_DIR: Path = Path(".")
PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")   # project_root / "data"
PROM: Path = Path(".")   # project_root / "prom"
PUBLIC_BASE: str = "http://localhost:8000"

# Sentence-Transformers per-page embeddings (optional)
_EMB_MODEL = None

# === Local model root (NO NETWORK). Defaults to <repo>/src/models ============
MODELS_ROOT: Path = Path(os.environ.get(
    "MODELS_ROOT",
    str((Path(__file__).resolve().parent.parent / "models").resolve())
))

# Local singletons
_RERANK = {"tok": None, "mdl": None, "device": "cpu"}     # DistilBERT
_MINILM = None                                            # Sentence-Transformers MiniLM
_LLMV3 = {"proc": None, "mdl": None, "device": "cpu"}     # LayoutLMv3 (optional)

def init_paths_for_src(src_dir: Path, public_base: str = "http://localhost:8000"):
    """Call once from main.py"""
    global SRC_DIR, PROJECT_ROOT, DATA, PROM, PUBLIC_BASE, _EMB_MODEL
    SRC_DIR = src_dir.resolve()
    PROJECT_ROOT = SRC_DIR.parent
    PUBLIC_BASE = public_base.rstrip("/")

    DATA = (PROJECT_ROOT / "data").resolve(); DATA.mkdir(parents=True, exist_ok=True)
    PROM = (PROJECT_ROOT / "prom").resolve(); PROM.mkdir(parents=True, exist_ok=True)

    # Lazy-load per-page ST embeddings if available
    if SentenceTransformer is not None and _EMB_MODEL is None:
        try:
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB_MODEL = None

    print(f"[ocr_unified] SRC_DIR={SRC_DIR}")
    print(f"[ocr_unified] DATA={DATA}")
    print(f"[ocr_unified] MODELS_ROOT={MODELS_ROOT}")
    print(f"[ocr_unified] embeddings={'on' if _EMB_MODEL else 'off'}")

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
# Local-only model loaders
# -----------------------------------------------------------------------------
def _ensure_reranker(model_name_or_path: str = "distilbert-base-uncased"):
    """DistilBERT local loader (no network)."""
    if AutoTokenizer is None or AutoModel is None:
        return
    if _RERANK["mdl"] is not None:
        return
    candidates = [
        MODELS_ROOT / "distilbert-base-uncased",
        MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
    ]
    local_dir = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if not local_dir:
        print("[ocr_unified] DistilBERT local weights not found; skipping.")
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(local_dir), local_files_only=True)
    mdl.to(dev).eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
    print(f"[ocr_unified] DistilBERT loaded locally from {local_dir} on {dev}")

def _load_minilm_local():
    """MiniLM (SentenceTransformer) local loader (no network)."""
    global _MINILM
    if _MINILM is not None:
        return _MINILM
    if SentenceTransformer is None:
        return None
    candidates = [
        MODELS_ROOT / "sentence-transformers" / "all-MiniLM-L6-v2",
        MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
        MODELS_ROOT / "all-MiniLM-L6-v2",
        MODELS_ROOT / "MiniLML6-v2",  # your folder
    ]
    path = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if not path:
        print("[ocr_unified] MiniLM not found; skipping.")
        return None
    _MINILM = SentenceTransformer(str(path))
    print(f"[ocr_unified] MiniLM loaded locally from {path}")
    return _MINILM

def _load_layoutlmv3_local():
    """Optional LayoutLMv3 local loader (no network)."""
    if AutoProcessor is None or LayoutLMv3Model is None:
        return None
    if _LLMV3["mdl"] is not None:
        return _LLMV3
    candidates = [
        MODELS_ROOT / "microsoft" / "layoutlmv3-base",
        MODELS_ROOT / "microsoft__layoutlmv3-base",
        MODELS_ROOT / "layoutlmv3-base",
    ]
    path = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if not path:
        print("[ocr_unified] LayoutLMv3 not found; skipping.")
        return None
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    mdl = LayoutLMv3Model.from_pretrained(str(path), local_files_only=True)
    mdl.to(dev).eval()
    _LLMV3.update({"proc": proc, "mdl": mdl, "device": dev})
    print(f"[ocr_unified] LayoutLMv3 loaded locally from {path} on {dev}")
    return _LLMV3

# -----------------------------------------------------------------------------
# Upload / Meta / Boxes / Lasso crop
# -----------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"ok": True, "svc": "ocr_unified", "MODELS_ROOT": str(MODELS_ROOT)}

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
    if not bp.exists():
        return []
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

    x0 = float(min(req.x0, req.x1))
    y0 = float(min(req.y0, req.y1))
    x1 = float(max(req.x0, req.x1))
    y1 = float(max(req.y0, req.y1))

    pad = 4
    X0 = max(0, int(x0 - pad))
    Y0 = max(0, int(y0 - pad))
    X1 = min(gray.width - 1, int(x1 + pad))
    Y1 = min(gray.height - 1, int(y1 + pad))

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
    try:
        crop.save(crop_path)
        crop_url = f"/data/{req.doc_id}/last_crop.png"
    except Exception:
        crop_url = None

    return {
        "text": best,
        "rect_used": {"page": req.page, "x0": X0, "y0": Y0, "x1": X1, "y1": Y1},
        "page_size": {"width": gray.width, "height": gray.height},
        "crop_url": crop_url,
    }

# -----------------------------------------------------------------------------
# DistilBERT rerank (local)
# -----------------------------------------------------------------------------
def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _union_rect(span):
    return {"x0": float(min(t["x0"] for t in span)),
            "y0": float(min(t["y0"] for t in span)),
            "x1": float(max(t["x1"] for t in span)),
            "y1": float(max(t["y1"] for t in span))}

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
    out = []
    n = len(tokens_page)
    for i in range(n):
        accum = []
        text = ""
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
    # dedup
    dedup = []
    seen = set()
    for c in out:
        r = c["rect"]
        key = (round(r["x0"]/5), round(r["y0"]/5), round((r["x1"]-r["x0"])/5), round((r["y1"]-r["y0"])/5))
        if key in seen: continue
        seen.add(key)
        dedup.append(c)
    return dedup

def _context_snippet(tokens_page, span, px_margin=120, py_margin=35):
    R = _union_rect(span)
    cx0 = R["x0"] - px_margin; cy0 = R["y0"] - py_margin
    cx1 = R["x1"] + px_margin; cy1 = R["y1"] + py_margin
    bag = [t for t in tokens_page
           if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _concat_text(bag)

def _embed_distil(texts):
    tok = _RERANK["tok"]; mdl = _RERANK["mdl"]; dev = _RERANK["device"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

class RerankReq(BaseModel):
    doc_id: str
    key: str
    value: str
    model: Optional[str] = None
    topk: int = 8

@router.post("/rerank")
async def rerank(req: RerankReq):
    _ensure_reranker(req.model or "distilbert-base-uncased")
    if _RERANK["mdl"] is None:
        return {"best": None, "alts": []}
    bp = boxes_path(req.doc_id)
    mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text())
    by_pg = _group_tokens_by_page(tokens)

    vnorm = _norm_text(req.value)
    cands = []
    for pg, toks in by_pg.items():
        spans = _find_candidate_spans(vnorm, toks)
        for s in spans:
            local = _context_snippet(toks, s["span"])
            cands.append({"page": pg, "rect": s["rect"], "span_text": s["text"], "context": local})
    if not cands:
        return {"best": None, "alts": []}

    query = f"{req.key}: {req.value}".strip()
    texts = [query] + [c["context"] for c in cands]
    embs = _embed_distil(texts)
    q = embs[0]
    sims = (embs[1:] @ q)
    scored = sorted(
        [{"score": float(sims[i]), **cands[i]} for i in range(len(cands))],
        key=lambda x: x["score"], reverse=True,
    )
    best = scored[0]
    alts = scored[1: min(len(scored), req.topk)]
    return {
        "best": {"page": best["page"], "rect": best["rect"], "score": best["score"]},
        "alts": [{"page": a["page"], "rect": a["rect"], "score": a["score"]} for a in alts]
    }

# -----------------------------------------------------------------------------
# 5-method matcher (fuzzy, tfidf, minilm, distilbert, layoutlmv3)
# -----------------------------------------------------------------------------
def _cos_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9: return 0.0
    return float(np.clip(a @ b / (na * nb), 0.0, 1.0))

def _embed_minilm(model, texts):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

class MatchReq(BaseModel):
    doc_id: str
    key: str
    value: str
    page_hint: int | None = None
    max_window: int = 12

@router.post("/match/field")
async def match_field(req: MatchReq):
    bp = boxes_path(req.doc_id)
    mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text())

    # group tokens by page
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        pg = int(t["page"])
        pages.setdefault(pg, []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    key = _norm_text(req.key)
    val = _norm_text(req.value)
    q_val = val
    q_combo = f"{key} {val}".strip()

    # per-page tf-idf
    tfidf = {}
    for pg, toks in pages.items():
        corpus = [" ".join((t.get("text") or "").strip() for t in toks)]
        vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        try: vec.fit(corpus)
        except ValueError: vec.fit(["placeholder"])
        tfidf[pg] = vec

    def _slide_windows(tokens_page, max_w=12):
        n = len(tokens_page)
        for i in range(n):
            acc = []
            for w in range(max_w):
                j = i + w
                if j >= n: break
                txt = (tokens_page[j].get("text") or "").strip()
                if not txt: continue
                acc.append(tokens_page[j])
                yield acc

    def pick_best(scored):
        if not scored: return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    results = {"fuzzy": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # prepare models
    minilm = _load_minilm_local()
    _ensure_reranker("distilbert-base-uncased")
    distil_ok = _RERANK["mdl"] is not None
    lv3 = _load_layoutlmv3_local()

    pages_iter = [(req.page_hint, pages.get(req.page_hint))] if req.page_hint in pages else pages.items()

    # score all methods
    for method in ("fuzzy", "tfidf", "minilm", "distilbert"):
        scored = []
        if method == "minilm" and minilm is None: results["minilm"] = None; continue
        if method == "distilbert" and not distil_ok: results["distilbert"] = None; continue
        for pg, toks in pages_iter:
            if not toks: continue
            for span in _slide_windows(toks, max_w=req.max_window):
                ctx = _context_snippet(toks, span)
                rect = _union_rect(span)
                span_text = _norm_text(" ".join((t.get("text") or "") for t in span))
                if method == "fuzzy":
                    s = float(_rfuzz.QRatio(val, span_text) / 100.0)
                elif method == "tfidf":
                    # value-centric TF-IDF
                    cctx = ctx
                    if key:
                        for kw in key.split():
                            if len(kw) >= 2:
                                cctx = cctx.replace(kw, " ")
                        cctx = _norm_text(cctx)
                    s_span  = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]), tfidf[pg].transform([span_text]))[0,0], 0, 1))
                    s_ctx   = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]), tfidf[pg].transform([cctx]))[0,0], 0, 1)) if cctx else 0.0
                    s_combo = float(np.clip(cosine_similarity(tfidf[pg].transform([q_combo]), tfidf[pg].transform([cctx]))[0,0], 0, 1)) if cctx else 0.0
                    v_toks = [w for w in val.split() if w]
                    if v_toks:
                        covered = 0
                        span_words = span_text.split()
                        for w in v_toks:
                            covered += any((_rfuzz.QRatio(w, sw) >= 90) for sw in span_words)
                        coverage = covered / max(1, len(v_toks))
                    else:
                        coverage = 0.0
                    s = 0.70 * s_span + 0.20 * max(s_ctx, s_combo) + 0.10 * coverage
                elif method == "minilm":
                    E = _embed_minilm(minilm, [f"{key}: {val}", ctx]); s = _cos_np(E[0], E[1])
                else:  # distilbert
                    E = _embed_distil([f"{key}: {val}", ctx]); s = float((E[1] @ E[0]))
                # penalize multiline unions
                ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
                spread = (ys[-1] - ys[0]) if len(ys) > 1 else 0.0
                avg_h = np.mean([t["y1"] - t["y0"] for t in span]) if span else 1.0
                penalty = max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)
                s = float(max(0.0, s - 0.12 * penalty))
                scored.append({"page": pg, "rect": rect, "score": s})
        results[method] = pick_best(scored)

    # LayoutLMv3 heuristic (only if local weights exist)
    if lv3 and lv3["mdl"] is not None:
        scored = []
        for pg, toks in pages.items():
            for span in _slide_windows(toks, max_w=req.max_window):
                ctx = _context_snippet(toks, span)
                rect = _union_rect(span)
                base = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]), tfidf[pg].transform([ctx]))[0,0], 0, 1))
                kwords = [w for w in key.split() if len(w) >= 2]
                near = 0
                x0 = rect["x0"] - 80; y0 = rect["y0"] - 40; x1 = rect["x1"] + 80; y1 = rect["y1"] + 40
                for t in toks:
                    if t["x1"] < x0 or t["x0"] > x1 or t["y1"] < y0 or t["y0"] > y1: continue
                    tx = _norm_text(t.get("text") or "")
                    if any(w in tx for w in kwords): near += 1
                s = float(min(1.0, base + 0.05 * min(near, 6)))
                scored.append({"page": pg, "rect": rect, "score": s})
        results["layoutlmv3"] = pick_best(scored)
    else:
        results["layoutlmv3"] = None

    return {"doc_id": req.doc_id, "key": req.key, "value": req.value, "methods": results}