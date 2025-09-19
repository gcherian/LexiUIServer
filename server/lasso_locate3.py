from __future__ import annotations
import os, re, json, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Keep PyTorch reasonable on CPU-only boxes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse helpers from your unified OCR router (adjust path if needed)
from .ocr_unified import boxes_path, meta_path  # noqa: F401

# --------- light deps ---------
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- heavy deps (local only, optional) ---------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoProcessor, LayoutLMv3Model
except Exception:
    torch = None
    AutoTokenizer = AutoModel = AutoProcessor = LayoutLMv3Model = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

router = APIRouter(prefix="/lasso", tags=["lasso-locate"])

# =============================================================================
# Local model roots (based on your folder names)
# =============================================================================
DEFAULT_MODELS_ROOT = Path("src/models").resolve()

CAND_MINILM = [
    DEFAULT_MODELS_ROOT / "sentence-transformers_all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "all-MiniLM-L6-v2",
]
CAND_DISTIL = [
    DEFAULT_MODELS_ROOT / "distilbert-base-uncased",
    DEFAULT_MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
]
CAND_LAYOUT = [
    DEFAULT_MODELS_ROOT / "microsoft_layoutlmv3-base",
    DEFAULT_MODELS_ROOT / "microsoft__layoutlmv3-base",
    DEFAULT_MODELS_ROOT / "layoutlmv3-base",
]

def _first_dir(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None

# =============================================================================
# Text utils, spans, penalties
# =============================================================================
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\u00A0", " ").split())

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
        acc: List[Dict[str,Any]] = []
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

# =============================================================================
# Anchors & type priors (regex)
# =============================================================================
ANCHORS: Dict[str, List[str]] = {
    "billing_address": ["bill to", "billing", "remit to", "invoice to", "billed to"],
    "shipping_address": ["ship to", "shipping", "deliver to"],
    "city": ["city", "town"],
    "state": ["state", "province", "region"],
    "zip": ["zip", "postal", "postcode"],
    "invoice_number": ["invoice", "invoice #", "inv #", "inv no", "invoice no"],
    "po_number": ["po", "purchase order", "po #", "po no"],
    "date": ["date", "invoice date", "issue date", "bill date", "due date"],
    "subtotal": ["subtotal"],
    "tax": ["tax", "vat", "gst"],
    "shipping": ["shipping", "freight"],
    "total": ["total", "amount due", "balance due", "grand total"],
    "vendor": ["vendor", "supplier", "from"],
    "customer": ["customer", "client", "to", "sold to"],
    "phone": ["phone", "tel", "telephone"],
    "email": ["email", "e-mail", "mail"],
}
def _anchor_terms_for_key(key: str) -> List[str]:
    k = key.lower()
    if "bill" in k and "address" in k: return ANCHORS["billing_address"]
    if "ship" in k and "address" in k: return ANCHORS["shipping_address"]
    if "city" in k: return ANCHORS["city"]
    if "state" in k: return ANCHORS["state"]
    if "zip" in k or "postal" in k or "postcode" in k: return ANCHORS["zip"]
    if "invoice" in k and ("number" in k or "no" in k or "#" in k): return ANCHORS["invoice_number"]
    if k.startswith("po") or "purchase order" in k: return ANCHORS["po_number"]
    if "due" in k and "date" in k: return ANCHORS["date"]
    if "date" in k: return ANCHORS["date"]
    if "subtotal" in k: return ANCHORS["subtotal"]
    if "tax" in k: return ANCHORS["tax"]
    if "shipping" in k or "freight" in k: return ANCHORS["shipping"]
    if "total" in k or "amount due" in k or "balance" in k: return ANCHORS["total"]
    if "vendor" in k or "supplier" in k or "from" in k: return ANCHORS["vendor"]
    if "customer" in k or "client" in k or "sold to" in k: return ANCHORS["customer"]
    if "phone" in k or "tel" in k: return ANCHORS["phone"]
    if "email" in k: return ANCHORS["email"]
    return []

RE_MONEY = re.compile(r"""(?x)(?:[$€£₹]\s*)?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?""")
RE_DATE  = re.compile(r"""(?ix)(?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})|(?:\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})|(?:[a-z]{3,9}\s+\d{1,2},?\s+\d{2,4})""")
RE_ZIP   = re.compile(r"""(?x)\b\d{5}(?:-\d{4})?\b""")
RE_INV   = re.compile(r"""(?ix)\b(?:inv|invoice)[\s#\-:]?\s*[A-Z0-9\-]{3,}\b""")

def _type_score(key: str, span_text_norm: str) -> float:
    k = key.lower()
    txt = span_text_norm
    if any(t in k for t in ("amount","total","subtotal","balance","tax","shipping")):
        if RE_MONEY.search(txt): return 1.0
        if any(ch.isdigit() for ch in txt): return 0.5
        return 0.0
    if "date" in k:
        return 0.9 if RE_DATE.search(txt) else 0.0
    if "zip" in k or "postal" in k or "postcode" in k:
        return 0.9 if RE_ZIP.search(txt) else 0.0
    if "invoice" in k and ("number" in k or "no" in k or "#" in k):
        return 0.7 if RE_INV.search(txt) or any(ch.isdigit() for ch in txt) else 0.0
    return 0.0

def _proximity_boost(rect: Dict[str,float], anchors: List[Tuple[float,float]]) -> float:
    if not anchors: return 0.0
    cx = 0.5*(rect["x0"]+rect["x1"]); cy = 0.5*(rect["y0"]+rect["y1"])
    best = 1e9
    for (ax, ay) in anchors:
        d = np.hypot(cx-ax, cy-ay)
        if d < best: best = d
    return float(np.clip(1.0 - best/600.0, 0.0, 1.0))

# =============================================================================
# Local model loaders + caches (STRICTLY local)
# =============================================================================
_loaded = {"minilm": None, "distil": None, "layout": None}

def _load_minilm(root: Optional[Path] = None) -> Optional[SentenceTransformer]:
    if _loaded["minilm"] is not None: return _loaded["minilm"]
    if SentenceTransformer is None: return None
    base = (root or DEFAULT_MODELS_ROOT)
    p = _first_dir([base / "sentence-transformers_all-MiniLM-L6-v2", *CAND_MINILM])
    if not p: return None
    _loaded["minilm"] = SentenceTransformer(str(p))
    return _loaded["minilm"]

class _DistilLocal:
    def __init__(self, tok, mdl, dev):
        self.tok, self.mdl, self.dev = tok, mdl, dev

def _load_distil(root: Optional[Path] = None) -> Optional[_DistilLocal]:
    if _loaded["distil"] is not None: return _loaded["distil"]
    if AutoTokenizer is None or AutoModel is None: return None
    base = (root or DEFAULT_MODELS_ROOT)
    p = _first_dir([base / "distilbert-base-uncased", *CAND_DISTIL])
    if not p: return None
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
    dev = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    _loaded["distil"] = _DistilLocal(tok, mdl, dev)
    return _loaded["distil"]

def _load_layout(root: Optional[Path] = None):
    if _loaded["layout"] is not None: return _loaded["layout"]
    if AutoProcessor is None or LayoutLMv3Model is None: return None
    base = (root or DEFAULT_MODELS_ROOT)
    p = _first_dir([base / "microsoft_layoutlmv3-base", *CAND_LAYOUT])
    if not p: return None
    proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
    mdl  = LayoutLMv3Model.from_pretrained(str(p), local_files_only=True)
    dev  = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    _loaded["layout"] = {"proc": proc, "mdl": mdl, "device": dev}
    return _loaded["layout"]

# =============================================================================
# API schemas
# =============================================================================
class LocateReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    models_root: Optional[str] = None
    model_root: Optional[str] = None
    methods: Optional[List[str]] = None  # e.g. ["fuzzy","tfidf","minilm","distilbert","layoutlmv3"]
    time_budget_ms: int = 1800          # hard cap per request

# =============================================================================
# Helper: char-gram vectorizer (robust to OCR noise)
# =============================================================================
_CHAR_VEC = HashingVectorizer(analyzer="char", ngram_range=(3,5), alternate_sign=False, n_features=2**18)

def _char_cosine(a: str, b: str) -> float:
    Xa = _CHAR_VEC.transform([a])
    Xb = _CHAR_VEC.transform([b])
    num = Xa.multiply(Xb).sum()
    da = float(np.sqrt((Xa.multiply(Xa)).sum()))
    db = float(np.sqrt((Xb.multiply(Xb)).sum()))
    if da == 0 or db == 0: return 0.0
    return float(np.clip(num / (da * db), 0.0, 1.0))

# =============================================================================
# Endpoints
# =============================================================================
@router.get("/locate/status")
def locate_status():
    return {
        "models_root": str(DEFAULT_MODELS_ROOT),
        "minilm": bool(_loaded["minilm"]),
        "distilbert": bool(_loaded["distil"]),
        "layoutlmv3": bool(_loaded["layout"]),
        "torch": bool(torch is not None),
    }

@router.post("/locate/warmup")
def locate_warmup(models_root: Optional[str] = Query(None)):
    mroot = Path(models_root).resolve() if models_root else None
    ok = {}
    try:
        ok["minilm"] = bool(_load_minilm(mroot))
    except Exception:
        ok["minilm"] = False
    try:
        ok["distilbert"] = bool(_load_distil(mroot))
    except Exception:
        ok["distilbert"] = False
    try:
        ok["layoutlmv3"] = bool(_load_layout(mroot))
    except Exception:
        ok["layoutlmv3"] = False
    return ok

@router.post("/locate")
def locate(req: LocateReq,
           models_root: Optional[str] = Query(None),
           model_root: Optional[str] = Query(None)):
    """
    Returns:
      {
        "methods": {
          "fuzzy": {page, rect:{x0,y0,x1,y1}, score} | null,
          "tfidf": { ... } | null,
          "minilm": { ... } | null,
          "distilbert": { ... } | null,
          "layoutlmv3": { ... } | null
        },
        "pages": [...]
      }
    Always LOCAL. Heavy models are optional & time-capped.
    """
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    root_str = models_root or model_root or req.models_root or req.model_root
    mroot: Optional[Path] = Path(root_str).resolve() if root_str else None

    tokens = json.loads(bp.read_text())
    meta   = json.loads(mp.read_text())
    pages: Dict[int, List[Dict[str,Any]]] = {}
    for t in tokens:
        pg = int(t["page"])
        pages.setdefault(pg, []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    # Per-page TF-IDF (word-level) for context
    tfidf: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        v = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        try: v.fit([txt])
        except ValueError: v.fit(["placeholder"])
        tfidf[pg] = v

    key_n = _norm(req.key)
    val_n = _norm(req.value)
    combo = f"{key_n} {val_n}".strip()
    max_w = max(4, min(int(req.max_window), 12))

    # Anchors from key + lexicon
    anchor_terms = set(_anchor_terms_for_key(req.key))
    anchor_terms.update([w for w in key_n.split() if len(w) >= 2])

    page_anchors: Dict[int, List[Tuple[float,float]]] = {}
    for pg, toks in pages.items():
        pts: List[Tuple[float,float]] = []
        for t in toks:
            tx = _norm(t.get("text") or "")
            if any(a in tx for a in anchor_terms):
                pts.append(((t["x0"]+t["x1"])*0.5, (t["y0"]+t["y1"])*0.5))
        page_anchors[pg] = pts

    def pick(scored: List[Dict[str,Any]]):
        if not scored: return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    # Allow caller to restrict which methods run
    allowed = set((req.methods or ["fuzzy","tfidf","minilm","distilbert","layoutlmv3"]))

    # Global time budget
    t0 = time.time()
    budget_ms = max(300, int(req.time_budget_ms))

    def time_left_ms() -> int:
        return int(budget_ms - 1000*(time.time() - t0))

    out: Dict[str, Any] = {"fuzzy": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # ---------- 1) FUZZY ----------
    if "fuzzy" in allowed and time_left_ms() > 50:
        scored: List[Dict[str,Any]] = []
        for pg, toks in pages.items():
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 600 or time_left_ms() <= 10: break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                # blend: token-level fuzzy + char-gram cosine
                s_raw = 0.6*(float(_rfuzz.QRatio(val_n, stext))/100.0) + 0.4*_char_cosine(val_n, stext)
                prox = _proximity_boost(rect, page_anchors.get(pg, []))
                tpri = _type_score(req.key, stext)
                s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
        out["fuzzy"] = pick(scored)

    # ---------- 2) TF-IDF ----------
    if "tfidf" in allowed and time_left_ms() > 50:
        scored: List[Dict[str,Any]] = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 600 or time_left_ms() <= 10: break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)
                s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0,0], 0, 1))
                s_ctx  = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                s_comb = float(np.clip(cosine_similarity(vec.transform([combo]),  vec.transform([ctx]))[0,0],  0, 1)) if ctx else 0.0
                # add a small char-gram to tolerate OCR noise
                s_char = _char_cosine(val_n, stext)
                s_raw = 0.55*s_span + 0.20*max(s_ctx, s_comb) + 0.25*s_char
                prox = _proximity_boost(rect, page_anchors.get(pg, []))
                tpri = _type_score(req.key, stext)
                s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
        out["tfidf"] = pick(scored)

    # ---------- 3) MiniLM ----------
    if "minilm" in allowed and time_left_ms() > 120:
        try:
            m = _load_minilm(Path(root_str).resolve() if (root_str:= (models_root or model_root or req.models_root or req.model_root)) else None)
            if m is not None:
                scored: List[Dict[str,Any]] = []
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 450 or time_left_ms() <= 40: break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        E = m.encode([f"{key_n} {val_n}", ctx], convert_to_numpy=True, normalize_embeddings=True)
                        s_raw = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                        prox = _proximity_boost(rect, page_anchors.get(pg, []))
                        tpri = _type_score(req.key, _norm(" ".join((t.get("text") or "") for t in span)))
                        s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                        scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
                out["minilm"] = pick(scored)
        except Exception:
            out["minilm"] = None

    # ---------- 4) DistilBERT ----------
    if "distilbert" in allowed and time_left_ms() > 120:
        try:
            d = _load_distil(Path(root_str).resolve() if (root_str:= (models_root or model_root or req.models_root or req.model_root)) else None)
            if d is not None and torch is not None:
                scored: List[Dict[str,Any]] = []
                with torch.no_grad():
                    for pg, toks in pages.items():
                        cnt = 0
                        for span in _slide(toks, max_w):
                            cnt += 1
                            if cnt > 450 or time_left_ms() <= 40: break
                            rect = _union(span)
                            ctx = _context(toks, span)
                            tok = d.tok([f"{key_n} {val_n}", ctx], padding=True, truncation=True,
                                        return_tensors="pt", max_length=256).to(d.dev)
                            hs = d.mdl(**tok).last_hidden_state
                            mask = tok["attention_mask"].unsqueeze(-1)
                            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                            emb = torch.nn.functional.normalize(pooled, dim=1)
                            s_raw = float((emb[0] @ emb[1].T).item())
                            prox = _proximity_boost(rect, page_anchors.get(pg, []))
                            tpri = _type_score(req.key, _norm(" ".join((t.get("text") or "") for t in span)))
                            s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                            scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
                out["distilbert"] = pick(scored)
        except Exception:
            out["distilbert"] = None

    # ---------- 5) LayoutLMv3 (proxy) ----------
    if "layoutlmv3" in allowed and time_left_ms() > 120:
        try:
            L = _load_layout(Path(root_str).resolve() if (root_str:= (models_root or model_root or req.models_root or req.model_root)) else None)
            if L is not None:
                scored: List[Dict[str,Any]] = []
                for pg, toks in pages.items():
                    vec = tfidf[pg]
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 450 or time_left_ms() <= 40: break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        base = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0], 0, 1))
                        # local key proximity bump
                        near = 0
                        x0 = rect["x0"]-80; y0 = rect["y0"]-40; x1 = rect["x1"]+80; y1 = rect["y1"]+40
                        key_terms = [w for w in key_n.split() if len(w) >= 2]
                        for tkn in toks:
                            if tkn["x1"]<x0 or tkn["x0"]>x1 or tkn["y1"]<y0 or tkn["y0"]>y1: continue
                            tx = _norm(tkn.get("text") or "")
                            if any(w in tx for w in key_terms): near += 1
                        s_raw = float(min(1.0, base + 0.05*min(near, 6)))
                        prox = _proximity_boost(rect, page_anchors.get(pg, []))
                        tpri = _type_score(req.key, _norm(" ".join((t.get("text") or "") for t in span)))
                        s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                        scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
                out["layoutlmv3"] = pick(scored)
        except Exception:
            out["layoutlmv3"] = None

    # Return shape the UI expects: "methods"
    return {"methods": out, "pages": meta.get("pages", [])}