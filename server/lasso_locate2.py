from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Make PyTorch friendlier on CPU boxes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Reuse helpers from your unified OCR router (adjust import if needed)
from .ocr_unified import boxes_path, meta_path  # noqa

# ---- lightweight deps ----
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- heavy deps (optional & local only) ----
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
# Model roots (based on your local folders)
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
# Normalization, spans, penalties
# =============================================================================
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
# Anchor lexicon (keywords near typical fields)
# Tune freely — this drives the proximity bonus
# =============================================================================
ANCHORS: Dict[str, List[str]] = {
    # addresses
    "billing_address": ["bill to", "billing", "remit to", "invoice to", "billed to"],
    "shipping_address": ["ship to", "shipping", "deliver to"],
    "mailing_address": ["mailing address", "correspondence address"],
    "city": ["city", "town"],
    "state": ["state", "province", "region"],
    "zip": ["zip", "postal", "postcode"],
    # invoice identity
    "invoice_number": ["invoice", "invoice #", "inv #", "inv no", "invoice no"],
    "po_number": ["po", "purchase order", "po #", "po no"],
    "date": ["date", "invoice date", "issue date", "bill date"],
    "due_date": ["due date", "payment due"],
    "terms": ["terms"],
    # amounts
    "subtotal": ["subtotal"],
    "tax": ["tax", "vat", "gst"],
    "shipping": ["shipping", "freight"],
    "total": ["total", "amount due", "balance due", "grand total"],
    # vendor/customer
    "vendor": ["vendor", "supplier", "from"],
    "customer": ["customer", "client", "to", "sold to"],
    # phone/email/etc
    "phone": ["phone", "tel", "telephone"],
    "email": ["email", "e-mail", "mail"],
}

def _anchor_terms_for_key(key: str) -> List[str]:
    k = key.lower()
    # heuristic mapping
    if "bill" in k and "address" in k: return ANCHORS["billing_address"]
    if "ship" in k and "address" in k: return ANCHORS["shipping_address"]
    if "mail" in k and "address" in k: return ANCHORS["mailing_address"]
    if "city" in k: return ANCHORS["city"]
    if "state" in k: return ANCHORS["state"]
    if "zip" in k or "postal" in k or "postcode" in k: return ANCHORS["zip"]
    if "invoice" in k and ("number" in k or "no" in k or "#" in k): return ANCHORS["invoice_number"]
    if k.startswith("po") or "purchase order" in k: return ANCHORS["po_number"]
    if "due" in k and "date" in k: return ANCHORS["due_date"]
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

# =============================================================================
# Regex cues (type prior)
# =============================================================================
RE_MONEY = re.compile(r"""(?x)
    (?:[$€£₹]\s*)?                # optional currency
    \d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?  # 1,234.56 / 1234 / 1 234,56 (loose)
""")
RE_DATE = re.compile(r"""(?ix)
    (?:\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})  # 12/31/2024 or 12-31-24
    | (?:\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})  # 2024-12-31
    | (?:[a-z]{3,9}\s+\d{1,2},?\s+\d{2,4})   # Dec 31, 2024
""")
RE_ZIP = re.compile(r"""(?x)\b\d{5}(?:-\d{4})?\b""")
RE_INVOICE_ID = re.compile(r"""(?ix)\b(?:inv|invoice)[\s#\-:]?\s*[A-Z0-9\-]{3,}\b""")

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
        return 0.7 if RE_INVOICE_ID.search(txt) or any(ch.isdigit() for ch in txt) else 0.0
    return 0.0

def _proximity_boost(rect: Dict[str,float], anchors: List[Tuple[float,float]]) -> float:
    if not anchors: return 0.0
    cx = 0.5*(rect["x0"]+rect["x1"]); cy = 0.5*(rect["y0"]+rect["y1"])
    best = 1e9
    for (ax, ay) in anchors:
        d = np.hypot(cx-ax, cy-ay); 
        if d < best: best = d
    # distance → boost [0..1]
    return float(np.clip(1.0 - best/600.0, 0.0, 1.0))

# =============================================================================
# Model loaders (LOCAL ONLY) + caches
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
# IO & request schema
# =============================================================================
class LocateReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    # Accept both spellings. If both provided, models_root wins.
    models_root: Optional[str] = None
    model_root: Optional[str] = None

# =============================================================================
# Main endpoint
# =============================================================================
@router.post("/locate")
def locate(req: LocateReq,
           models_root: Optional[str] = Query(None),
           model_root: Optional[str] = Query(None)):
    """
    Returns: { hits: {autolocate, tfidf, minilm, distilbert, layoutlmv3}, pages: [...] }
    Each hit is { page, rect:{x0,y0,x1,y1}, score } or None.
    Strictly local; if a model is missing, that method returns None and others still work.
    """
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    # Resolve root preference: explicit query param > body.models_root > body.model_root
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

    # per-page tfidf
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
    max_w = max(4, min(int(req.max_window), 10))

    # ----- build anchors from page tokens using keyword lexicon -----
    anchor_terms = set()
    for t in _anchor_terms_for_key(req.key):
        anchor_terms.add(_norm(t))
    # If no specialized anchors detected, still keep the key words as anchors:
    key_terms = [w for w in key_n.split() if len(w) >= 2]
    anchor_terms.update(key_terms)

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

    out: Dict[str, Any] = {"autolocate": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # =========================
    # 1) AUTOLOCATE (fuzzy)
    # =========================
    scored_auto: List[Dict[str,Any]] = []
    for pg, toks in pages.items():
        cnt = 0
        for span in _slide(toks, max_w):
            cnt += 1
            if cnt > 700: break
            rect = _union(span)
            stext = _norm(" ".join((t.get("text") or "") for t in span))
            prox = _proximity_boost(rect, page_anchors.get(pg, []))
            tpri = _type_score(req.key, stext)
            s_raw = float(_rfuzz.QRatio(val_n, stext)) / 100.0
            s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
            scored_auto.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
    out["autolocate"] = pick(scored_auto)

    # =========================
    # 2) TF-IDF baseline
    # =========================
    scored_tfidf: List[Dict[str,Any]] = []
    for pg, toks in pages.items():
        vec = tfidf[pg]
        cnt = 0
        for span in _slide(toks, max_w):
            cnt += 1
            if cnt > 700: break
            rect = _union(span)
            stext = _norm(" ".join((t.get("text") or "") for t in span))
            ctx = _context(toks, span)
            s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0,0], 0, 1))
            s_ctx  = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
            s_comb = float(np.clip(cosine_similarity(vec.transform([combo]),  vec.transform([ctx]))[0,0],  0, 1)) if ctx else 0.0
            prox = _proximity_boost(rect, page_anchors.get(pg, []))
            tpri = _type_score(req.key, stext)
            s_raw = 0.75*s_span + 0.25*max(s_ctx, s_comb)
            s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
            scored_tfidf.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
    out["tfidf"] = pick(scored_tfidf)

    # =========================
    # 3) MiniLM (local ST)
    # =========================
    try:
        m = _load_minilm(mroot)
        if m is not None:
            scored: List[Dict[str,Any]] = []
            for pg, toks in pages.items():
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 450: break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    E = m.encode([combo, ctx], convert_to_numpy=True, normalize_embeddings=True)
                    s_raw = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    prox = _proximity_boost(rect, page_anchors.get(pg, []))
                    tpri = _type_score(req.key, _norm(" ".join((t.get("text") or "") for t in span)))
                    s = s_raw + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                    scored.append({"page": pg, "rect": rect, "score": float(np.clip(s, 0.0, 1.0))})
            out["minilm"] = pick(scored)
    except Exception:
        out["minilm"] = None

    # =========================
    # 4) DistilBERT (local HF)
    # =========================
    try:
        d = _load_distil(mroot)
        if d is not None and torch is not None:
            scored: List[Dict[str,Any]] = []
            with torch.no_grad():
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 450: break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        tok = d.tok([combo, ctx], padding=True, truncation=True,
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

    # =========================
    # 5) LayoutLMv3 (proxy, still local)
    # =========================
    try:
        L = _load_layout(mroot)
        if L is not None:
            scored: List[Dict[str,Any]] = []
            for pg, toks in pages.items():
                vec = tfidf[pg]
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 450: break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    base = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0], 0, 1))
                    # key proximity bump inside a slightly larger box
                    near = 0
                    x0 = rect["x0"]-80; y0 = rect["y0"]-40; x1 = rect["x1"]+80; y1 = rect["y1"]+40
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

    return {"hits": out, "pages": meta.get("pages", [])}