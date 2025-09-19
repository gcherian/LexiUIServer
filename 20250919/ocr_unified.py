# ocr_unified.py — lean server for Upload/OCR, 4-model locate, GT save+IOU, local-only models
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from uuid import uuid4
import shutil, json, re, os, time

# OCR
import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

# ML bits
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional local models
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoProcessor, LayoutLMv3Model
except Exception:
    torch = None
    AutoTokenizer = AutoModel = AutoProcessor = LayoutLMv3Model = None

# -----------------------------------------------------------------------------
# Router & paths
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/lasso", tags=["lasso"])

SRC_DIR: Path = Path(".")
PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")
PUBLIC_BASE: str = "http://localhost:8000"

# Models root (LOCAL ONLY)
MODELS_ROOT: Path = Path(os.environ.get(
    "MODELS_ROOT",
    str((Path(__file__).resolve().parent.parent / "models").resolve())
))

# Optional page-embeddings (unused by API, but we keep helper available)
_PAGE_EMB = None

# Singletons for local models
_MINILM = None
_RERANK = {"tok": None, "mdl": None, "device": "cpu"}          # DistilBERT
_LLMV3 = {"proc": None, "mdl": None, "device": "cpu"}          # optional

def init_paths_for_src(src_dir: Path, public_base: str = "http://localhost:8000"):
    global SRC_DIR, PROJECT_ROOT, DATA, PUBLIC_BASE, _PAGE_EMB
    SRC_DIR = src_dir.resolve()
    PROJECT_ROOT = SRC_DIR.parent
    DATA = (PROJECT_ROOT / "data").resolve(); DATA.mkdir(parents=True, exist_ok=True)
    PUBLIC_BASE = public_base.rstrip("/")

    # best-effort local page embedder (unused by endpoints but safe to keep)
    if SentenceTransformer is not None and _PAGE_EMB is None:
        for p in [
            MODELS_ROOT / "sentence-transformers" / "all-MiniLM-L6-v2",
            MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
            MODELS_ROOT / "all-MiniLM-L6-v2",
            MODELS_ROOT / "MiniLML6-v2",
        ]:
            if p.exists() and p.is_dir():
                try:
                    _PAGE_EMB = SentenceTransformer(str(p))
                except Exception:
                    _PAGE_EMB = None
                break

    print(f"[unified] DATA={DATA}")
    print(f"[unified] MODELS_ROOT={MODELS_ROOT}")
    print(f"[unified] page_emb={'on' if _PAGE_EMB else 'off'}")

# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------
def doc_dir(doc_id: str) -> Path:
    d = DATA / doc_id; d.mkdir(parents=True, exist_ok=True); return d

def pdf_path(doc_id: str) -> Path:       return doc_dir(doc_id) / "original.pdf"
def meta_path(doc_id: str) -> Path:      return doc_dir(doc_id) / "meta.json"
def boxes_path(doc_id: str) -> Path:     return doc_dir(doc_id) / "boxes.json"
def page_text_path(doc_id: str) -> Path: return doc_dir(doc_id) / "pages.txt"
def gt_path(doc_id: str) -> Path:        return doc_dir(doc_id) / "groundtruth.json"

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
# Pydantic models
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
    text: Optional[str] = None

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

class MatchReq(BaseModel):
    doc_id: str
    key: str
    value: str
    page_hint: Optional[int] = None
    max_window: int = 12

class GTPutReq(BaseModel):
    doc_id: str
    key: str
    page: int
    rect: Rect
    text: Optional[str] = None

# -----------------------------------------------------------------------------
# OCR pipeline
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
    return MetaResp(pages=pages_meta)

# -----------------------------------------------------------------------------
# Local-only model loaders
# -----------------------------------------------------------------------------
def _load_minilm():
    global _MINILM
    if _MINILM is not None: return _MINILM
    if SentenceTransformer is None: return None
    for p in [
        MODELS_ROOT / "sentence-transformers" / "all-MiniLM-L6-v2",
        MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
        MODELS_ROOT / "all-MiniLM-L6-v2",
        MODELS_ROOT / "MiniLML6-v2",
    ]:
        if p.exists() and p.is_dir():
            try:
                _MINILM = SentenceTransformer(str(p))
                print(f"[unified] MiniLM loaded from {p}")
                return _MINILM
            except Exception as e:
                print(f"[unified] MiniLM load error: {e}")
                return None
    print("[unified] MiniLM not found (skipping)")
    return None

def _ensure_distil():
    if _RERANK["mdl"] is not None: return True
    if AutoTokenizer is None or AutoModel is None: return False
    for p in [
        MODELS_ROOT / "distilbert-base-uncased",
        MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
    ]:
        if p.exists() and p.is_dir():
            try:
                dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
                tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
                mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
                mdl.to(dev).eval()
                _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
                print(f"[unified] DistilBERT loaded from {p} on {dev}")
                return True
            except Exception as e:
                print(f"[unified] Distil load error: {e}")
                return False
    print("[unified] DistilBERT not found (skipping)")
    return False

def _ensure_layoutlmv3():  # optional
    if _LLMV3["mdl"] is not None: return True
    if AutoProcessor is None or LayoutLMv3Model is None: return False
    for p in [
        MODELS_ROOT / "microsoft" / "layoutlmv3-base",
        MODELS_ROOT / "microsoft__layoutlmv3-base",
        MODELS_ROOT / "layoutlmv3-base",
    ]:
        if p.exists() and p.is_dir():
            try:
                dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
                proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
                mdl  = LayoutLMv3Model.from_pretrained(str(p), local_files_only=True)
                mdl.to(dev).eval()
                _LLMV3.update({"proc": proc, "mdl": mdl, "device": dev})
                print(f"[unified] LayoutLMv3 loaded from {p} on {dev}")
                return True
            except Exception as e:
                print(f"[unified] LayoutLMv3 load error: {e}")
                return False
    print("[unified] LayoutLMv3 not found (skipping)")
    return False

# -----------------------------------------------------------------------------
# Utilities (normalize, windows, etc.)
# -----------------------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\u00A0"," ").split())

def _union(span: List[Dict[str,Any]]) -> Dict[str,float]:
    return {"x0": float(min(t["x0"] for t in span)),
            "y0": float(min(t["y0"] for t in span)),
            "x1": float(max(t["x1"] for t in span)),
            "y1": float(max(t["y1"] for t in span))}

def _context(tokens: List[Dict[str,Any]], span: List[Dict[str,Any]], px=120, py=35) -> str:
    R = _union(span)
    cx0, cy0, cx1, cy1 = R["x0"]-px, R["y0"]-py, R["x1"]+px, R["y1"]+py
    bag = [t for t in tokens if not (t["x1"]<cx0 or t["x0"]>cx1 or t["y1"]<cy0 or t["y0"]>cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _norm(" ".join((t.get("text") or "") for t in bag if t.get("text")))

def _slide(tokens: List[Dict[str,Any]], max_w=12):
    n = len(tokens)
    for i in range(n):
        acc = []
        for w in range(max_w):
            j = i + w
            if j >= n: break
            txt = (tokens[j].get("text") or "").strip()
            if not txt: continue
            acc.append(tokens[j])
            yield acc

def _line_penalty(span: List[Dict[str,Any]]) -> float:
    if len(span) <= 1: return 0.0
    ys = sorted([(t["y0"] + t["y1"])*0.5 for t in span])
    spread = ys[-1] - ys[0]
    avg_h = float(np.mean([t["y1"]-t["y0"] for t in span]))
    return max(0.0, (spread - 0.6*avg_h)) / max(1.0, avg_h)

def _iou(a: Dict[str,float], b: Dict[str,float]) -> float:
    ax0, ay0, ax1, ay1 = min(a["x0"],a["x1"]), min(a["y0"],a["y1"]), max(a["x0"],a["x1"]), max(a["y0"],a["y1"])
    bx0, by0, bx1, by1 = min(b["x0"],b["x1"]), min(b["y0"],b["y1"]), max(b["x0"],b["x1"]), max(b["y0"],b["y1"])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1-ix0), max(0.0, iy1-iy0)
    inter = iw*ih
    area_a = max(0.0, ax1-ax0) * max(0.0, ay1-ay0)
    area_b = max(0.0, bx1-bx0) * max(0.0, by1-by0)
    union = max(1e-9, area_a + area_b - inter)
    return float(inter / union)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@router.get("/health")
def health():
    return {
        "ok": True,
        "svc": "ocr_unified",
        "DATA": str(DATA),
        "MODELS_ROOT": str(MODELS_ROOT),
        "minilm_loaded": bool(_MINILM),
        "distil_loaded": bool(_RERANK["mdl"]),
        "layoutlmv3_loaded": bool(_LLMV3["mdl"]),
    }

@router.post("/warmup")
def warmup():
    ok_minilm  = _load_minilm() is not None
    ok_distil  = _ensure_distil()
    ok_layout  = _ensure_layoutlmv3()
    return {
        "minilm": ok_minilm,
        "distilbert": ok_distil,
        "layoutlmv3": ok_layout,
    }

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

# ---- locate: fuzzy, tfidf, minilm, distilbert (layout optional) ----
def _embed_minilm(model, texts): return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
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

@router.post("/match/field")
async def match_field(req: MatchReq):
    bp = boxes_path(req.doc_id)
    mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text())

    # group by page
    pages: Dict[int, List[Dict[str,Any]]] = {}
    for t in tokens:
        pages.setdefault(int(t["page"]), []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    key = _norm(req.key); val = _norm(req.value)
    q_val = val; q_combo = f"{key} {val}".strip()

    # tfidf per page
    tfidf: Dict[int,TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        v = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        try: v.fit([txt])
        except ValueError: v.fit(["placeholder"])
        tfidf[pg] = v

    # preload models
    errors: List[str] = []
    minilm = _load_minilm()
    if minilm is None: errors.append("MiniLM not available (local folder missing).")
    distil_ok = _ensure_distil()
    if not distil_ok: errors.append("DistilBERT not available (local folder missing).")
    layout_ok = _ensure_layoutlmv3()  # optional

    pages_iter = [(req.page_hint, pages.get(req.page_hint))] if (req.page_hint and req.page_hint in pages) else list(pages.items())

    def pick(scored):
        if not scored: return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    out = {"fuzzy": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # common scan
    def scan(method: str, max_count: int = 1500):
        scored = []
        for pg, toks in pages_iter:
            if not toks: continue
            cnt = 0
            for span in _slide(toks, max_w=req.max_window):
                cnt += 1
                if cnt > max_count: break
                rect = _union(span)
                span_text = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)

                if method == "fuzzy":
                    s = float(_rfuzz.QRatio(val, span_text) / 100.0)
                elif method == "tfidf":
                    cctx = ctx
                    if key:
                        for kw in key.split():
                            if len(kw) >= 2: cctx = cctx.replace(kw, " ")
                        cctx = _norm(cctx)
                    s_span  = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]),   tfidf[pg].transform([span_text]))[0,0], 0, 1))
                    s_ctx   = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]),   tfidf[pg].transform([cctx]))[0,0],       0, 1)) if cctx else 0.0
                    s_combo = float(np.clip(cosine_similarity(tfidf[pg].transform([q_combo]), tfidf[pg].transform([cctx]))[0,0],       0, 1)) if cctx else 0.0
                    v_toks = [w for w in val.split() if w]
                    coverage = 0.0
                    if v_toks:
                        span_words = span_text.split()
                        covered = sum(1 for w in v_toks if any(_rfuzz.QRatio(w, sw) >= 90 for sw in span_words))
                        coverage = covered / max(1, len(v_toks))
                    s = 0.70*s_span + 0.20*max(s_ctx, s_combo) + 0.10*coverage
                elif method == "minilm":
                    if minilm is None: return []
                    try:
                        E = _embed_minilm(minilm, [f"{key}: {val}", ctx])
                        s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    except Exception as ex:
                        errors.append(f"MiniLM embed error: {ex}"); continue
                elif method == "distilbert":
                    if not distil_ok: return []
                    try:
                        E = _embed_distil([f"{key}: {val}", ctx])
                        s = float(np.clip(E[0].dot(E[1]), 0, 1))
                    except Exception as ex:
                        errors.append(f"DistilBERT embed error: {ex}"); continue
                else:
                    return []

                # penalize multi-line unions
                s = max(0.0, s - 0.12 * _line_penalty(span))
                scored.append({"page": pg, "rect": rect, "score": s})
        return scored

    out["fuzzy"]      = pick(scan("fuzzy"))
    out["tfidf"]      = pick(scan("tfidf"))
    out["minilm"]     = pick(scan("minilm"))
    out["distilbert"] = pick(scan("distilbert"))

    # optional LayoutLMv3 heuristic using tf-idf + key proximity
    if layout_ok and _LLMV3["mdl"] is not None:
        scored = []
        for pg, toks in pages.items():
            cnt = 0
            for span in _slide(toks, max_w=req.max_window):
                cnt += 1
                if cnt > 1500: break
                rect = _union(span)
                ctx = _context(toks, span)
                base = float(np.clip(cosine_similarity(tfidf[pg].transform([q_val]), tfidf[pg].transform([ctx]))[0,0], 0, 1))
                near = 0
                kwords = [w for w in key.split() if len(w) >= 2]
                x0 = rect["x0"]-80; y0 = rect["y0"]-40; x1 = rect["x1"]+80; y1 = rect["y1"]+40
                for t in toks:
                    if t["x1"]<x0 or t["x0"]>x1 or t["y1"]<y0 or t["y0"]>y1: continue
                    tx = _norm(t.get("text") or "")
                    if any(w in tx for w in kwords): near += 1
                s = float(min(1.0, base + 0.05*min(near, 6)))
                scored.append({"page": pg, "rect": rect, "score": s})
        out["layoutlmv3"] = pick(scored)
    else:
        out["layoutlmv3"] = None

    return {"doc_id": req.doc_id, "key": req.key, "value": req.value, "methods": out, "errors": errors}

# -----------------------------------------------------------------------------
# Ground Truth (incremental) + IOU report
# -----------------------------------------------------------------------------
@router.get("/groundtruth/get")
def gt_get(doc_id: str):
    gp = gt_path(doc_id)
    if not gp.exists():
        return {"doc_id": doc_id, "updated": 0, "fields": {}}
    data = json.loads(gp.read_text())
    return data

@router.post("/groundtruth/put")
def gt_put(req: GTPutReq):
    gp = gt_path(req.doc_id)
    data = {"doc_id": req.doc_id, "updated": int(time.time()), "fields": {}}
    if gp.exists():
        try: data = json.loads(gp.read_text())
        except Exception: pass
    fields = data.setdefault("fields", {})
    fields[req.key] = {
        "page": int(req.page),
        "rect": {"x0": req.rect.x0, "y0": req.rect.y0, "x1": req.rect.x1, "y1": req.rect.y1},
        "text": (req.text or "").strip(),
        "ts": int(time.time()),
    }
    data["updated"] = int(time.time())
    gp.write_text(json.dumps(data, indent=2))
    return {"ok": True, "doc_id": req.doc_id, "key": req.key}

@router.get("/groundtruth/report")
def gt_report(doc_id: str):
    gp = gt_path(doc_id)
    bp = boxes_path(doc_id)
    if not gp.exists(): raise HTTPException(404, "No groundtruth for doc")
    if not bp.exists(): raise HTTPException(404, "No tokens/boxes for doc")
    gt = json.loads(gp.read_text()).get("fields", {})
    # For each GT key, recompute predictions and IOU
    rows = []
    for key, g in gt.items():
        value = g.get("text","") or ""  # compare by geometry; value optional
        try:
            res = match_field(MatchReq(doc_id=doc_id, key=key, value=value, page_hint=g.get("page"), max_window=12))  # type: ignore
        except Exception as e:
            rows.append({"key": key, "error": str(e)}); continue
        grect = g.get("rect") or {}
        def iou_of(meth):
            hit = (res["methods"].get(meth) or None)
            if not hit or not hit.get("rect"): return None
            return _iou(grect, hit["rect"])
        rows.append({
            "key": key,
            "page": g.get("page"),
            "iou": {
                "fuzzy": iou_of("fuzzy"),
                "tfidf": iou_of("tfidf"),
                "minilm": iou_of("minilm"),
                "distilbert": iou_of("distilbert"),
                "layoutlmv3": iou_of("layoutlmv3"),
            }
        })
    # macro averages (ignore None)
    def avg(name):
        vals = [r["iou"][name] for r in rows if r.get("iou") and r["iou"].get(name) is not None]
        return float(np.mean(vals)) if vals else None
    summary = {k: avg(k) for k in ["fuzzy","tfidf","minilm","distilbert","layoutlmv3"]}
    return {"doc_id": doc_id, "rows": rows, "summary": summary}

# -----------------------------------------------------------------------------
# Minimal compat endpoint (legacy UIs)
# -----------------------------------------------------------------------------
@router.get("/compat/boxes")
async def boxes_compat(doc_url: str = Query(...), page: int = Query(1)):
    did = doc_id_from_doc_url(doc_url)
    if not did:
        raise HTTPException(400, f"Could not resolve doc_id from doc_url='{doc_url}'")
    bp = boxes_path(did)
    if not bp.exists(): return []
    all_boxes = json.loads(bp.read_text())
    return [b for b in all_boxes if int(b.get("page", 0)) == int(page)]
