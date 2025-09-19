# server/routers/lasso_locate.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json, math

# Reuse paths from your big unified router
# If your module name differs, adjust the import path below.
from .ocr_unified import boxes_path, meta_path

# Light deps
import numpy as np
from rapidfuzz import fuzz as rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Heavy deps (optional + local-only)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

router = APIRouter(prefix="/lasso", tags=["lasso-locate"])

# ----------------------------- Local model roots -----------------------------
DEFAULT_MODELS_ROOT = Path("src/models").resolve()

CANDIDATES_MINILM = [
    DEFAULT_MODELS_ROOT / "sentence-transformers_all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "all-MiniLM-L6-v2",
]
CANDIDATES_DISTIL = [
    DEFAULT_MODELS_ROOT / "distilbert-base-uncased",
    DEFAULT_MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
]

def _first_dir(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None

# ----------------------------- Text helpers ---------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("\u00A0", " ").split())

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
    ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
    spread = ys[-1] - ys[0]
    avg_h = float(np.mean([t["y1"] - t["y0"] for t in span]))
    return max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)

# ----------------------------- Anchors & priors ------------------------------
# lightweight anchors (labels commonly near values)
ANCHOR_WORDS = {
    "zip": ["zip", "postal", "postcode"],
    "date": ["date", "invoice date", "issue date", "billing date"],
    "amount": ["amount", "total", "balance due", "grand total", "subtotal"],
    "invoice": ["invoice", "inv", "invoice #", "invoice no", "invoice number"],
    "po": ["po", "purchase order", "po #", "po number"],
    "phone": ["phone", "tel", "telephone"],
    "fax": ["fax"],
    "email": ["email", "e-mail"],
}

def _likely_type(key_norm: str) -> str:
    k = key_norm
    if "zip" in k or "postal" in k: return "zip"
    if "date" in k: return "date"
    if "total" in k or "amount" in k or "balance" in k: return "amount"
    if "invoice" in k: return "invoice"
    if k.startswith("po") or "purchase order" in k: return "po"
    if "phone" in k or "tel" in k: return "phone"
    if "fax" in k: return "fax"
    if "email" in k or "e-mail" in k: return "email"
    return "text"

def _proximity_boost(rect: Dict[str,float], tokens: List[Dict[str,Any]], anchors: List[str]) -> float:
    if not anchors: return 0.0
    x0, y0, x1, y1 = rect["x0"], rect["y0"], rect["x1"], rect["y1"]
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    best = 0.0
    for t in tokens:
        txt = _norm(t.get("text") or "")
        if not txt: continue
        if any(a in txt for a in anchors):
            tx, ty = (t["x0"] + t["x1"]) * 0.5, (t["y0"] + t["y1"]) * 0.5
            d = math.hypot(tx - cx, ty - cy)
            # 0 at far (>= 600px), ~1 when overlapping
            best = max(best, float(max(0.0, 1.0 - d / 600.0)))
    return best

def _type_score(key_norm: str, span_text_norm: str) -> float:
    t = _likely_type(key_norm)
    s = span_text_norm
    if t == "zip":
        return 1.0 if any(len(tok) in (5,9) and tok.isdigit() for tok in s.split()) else 0.0
    if t == "date":
        return 1.0 if any(ch.isdigit() for ch in s) and any(sep in s for sep in ["-", "/", "."]) else 0.0
    if t == "amount":
        return 1.0 if any(c in s for c in "$€£₹") or any(ch.isdigit() for ch in s) else 0.0
    if t == "invoice":
        return 1.0 if any(p in s for p in ["inv", "invoice"]) else 0.0
    if t == "po":
        return 1.0 if "po" in s else 0.0
    if t == "phone":
        return 1.0 if any(c in s for c in "+()-") and any(ch.isdigit() for ch in s) else 0.0
    if t == "email":
        return 1.0 if "@" in s and "." in s else 0.0
    return 0.0

# ----------------------------- Optional models (local-only) ------------------
_loaded: Dict[str, Any] = {"minilm": None, "distil": None}

def _load_minilm(models_root: Optional[Path]) -> Optional[SentenceTransformer]:
    if _loaded["minilm"] is not None: return _loaded["minilm"]
    if SentenceTransformer is None: return None
    root = models_root or DEFAULT_MODELS_ROOT
    p = _first_dir([root / "sentence-transformers_all-MiniLM-L6-v2", *CANDIDATES_MINILM])
    if not p: return None
    _loaded["minilm"] = SentenceTransformer(str(p))
    return _loaded["minilm"]

class _DistilLocal:
    def __init__(self, tok, mdl, dev): self.tok, self.mdl, self.dev = tok, mdl, dev

def _load_distil(models_root: Optional[Path]) -> Optional[_DistilLocal]:
    if _loaded["distil"] is not None: return _loaded["distil"]
    if AutoTokenizer is None or AutoModel is None: return None
    root = models_root or DEFAULT_MODELS_ROOT
    p = _first_dir([root / "distilbert-base-uncased", *CANDIDATES_DISTIL])
    if not p: return None
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    _loaded["distil"] = _DistilLocal(tok, mdl, dev)
    return _loaded["distil"]

# ----------------------------- API models -----------------------------------
class LocateReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    models_root: Optional[str] = None  # override root if needed (local path string)
    want_minilm: bool = True
    want_distilbert: bool = True

# The compact scoring result used internally
def _pick(scored: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    if not scored: return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    b = scored[0]
    return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

@router.post("/locate")
def locate(req: LocateReq):
    # Load tokens/meta
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")
    tokens = json.loads(bp.read_text())
    meta = json.loads(mp.read_text())
    pages: Dict[int, List[Dict[str,Any]]] = {}
    for t in tokens:
        pages.setdefault(int(t["page"]), []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    # Build per-page TFIDF (for TF-IDF and some priors)
    tfidf: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        v = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
        try:
            v.fit([txt])
        except ValueError:
            v.fit(["placeholder"])
        tfidf[pg] = v

    key_n = _norm(req.key)
    val_n = _norm(req.value)
    combo = f"{key_n} {val_n}".strip()
    max_w = max(4, int(req.max_window))
    a_type = _likely_type(key_n)
    anchors = ANCHOR_WORDS.get(a_type, [])

    methods: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    # ----------------- FUZZY -----------------
    try:
        scored: List[Dict[str,Any]] = []
        for pg, toks in pages.items():
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 1500: break
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                rect = _union(span)
                # base fuzzy on the *value*
                s = float(rfuzz.QRatio(val_n, stext)) / 100.0
                # boosts
                prox = _proximity_boost(rect, toks, anchors)
                tpri = _type_score(key_n, stext)
                s = s + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                scored.append({"page": pg, "rect": rect, "score": s})
        methods["fuzzy"] = _pick(scored)
    except Exception as e:
        errors["fuzzy"] = f"{type(e).__name__}: {e}"

    # ----------------- TF-IDF -----------------
    try:
        scored = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 1500: break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)
                s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0,0], 0, 1))
                s_ctx  = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                s_comb = float(np.clip(cosine_similarity(vec.transform([combo]),  vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                base = 0.75*s_span + 0.25*max(s_ctx, s_comb)
                prox = _proximity_boost(rect, toks, anchors)
                tpri = _type_score(key_n, stext)
                s = base + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                scored.append({"page": pg, "rect": rect, "score": s})
        methods["tfidf"] = _pick(scored)
    except Exception as e:
        errors["tfidf"] = f"{type(e).__name__}: {e}"

    # ----------------- MiniLM (local) -----------------
    if req.want_minilm:
        try:
            mroot = Path(req.models_root).resolve() if req.models_root else None
            m = _load_minilm(mroot)
            if m is None:
                errors["minilm"] = "MiniLM local model not found"
            else:
                scored = []
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 800: break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        E = m.encode([combo, ctx], convert_to_numpy=True, normalize_embeddings=True)
                        base = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                        stext = _norm(" ".join((t.get("text") or "") for t in span))
                        prox = _proximity_boost(rect, toks, anchors)
                        tpri = _type_score(key_n, stext)
                        s = base + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                        scored.append({"page": pg, "rect": rect, "score": s})
                methods["minilm"] = _pick(scored)
        except Exception as e:
            errors["minilm"] = f"{type(e).__name__}: {e}"

    # ----------------- DistilBERT (local) -----------------
    if req.want_distilbert:
        try:
            mroot = Path(req.models_root).resolve() if req.models_root else None
            d = _load_distil(mroot)
            if d is None or torch is None:
                errors["distilbert"] = "DistilBERT local model not found"
            else:
                scored = []
                with torch.no_grad():
                    for pg, toks in pages.items():
                        cnt = 0
                        for span in _slide(toks, max_w):
                            cnt += 1
                            if cnt > 800: break
                            rect = _union(span)
                            ctx = _context(toks, span)
                            tok = d.tok([combo, ctx], padding=True, truncation=True,
                                        return_tensors="pt", max_length=256).to(d.dev)
                            hs = d.mdl(**tok).last_hidden_state
                            mask = tok["attention_mask"].unsqueeze(-1)
                            summed = (hs * mask).sum(1)
                            counts = mask.sum(1).clamp(min=1)
                            emb = torch.nn.functional.normalize(summed / counts, dim=1)
                            base = float((emb[0] @ emb[1].T).item())
                            stext = _norm(" ".join((t.get("text") or "") for t in span))
                            prox = _proximity_boost(rect, toks, anchors)
                            tpri = _type_score(key_n, stext)
                            s = base + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
                            scored.append({"page": pg, "rect": rect, "score": s})
                methods["distilbert"] = _pick(scored)
        except Exception as e:
            errors["distilbert"] = f"{type(e).__name__}: {e}"

    # layoutlmv3 intentionally omitted for now (blocked)
    return {"methods": methods, "errors": errors, "pages": meta.get("pages", [])}

# Adapter endpoint so the UI call name stays consistent:
class MatchFieldReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    models_root: Optional[str] = None
    want_minilm: bool = True
    want_distilbert: bool = True

@router.post("/match/field")
def match_field(req: MatchFieldReq):
    """Thin adapter: same result format the UI expects."""
    res = locate(LocateReq(**req.model_dump()))
    return res
