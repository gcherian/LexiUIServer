from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Reuse from your unified OCR router
from .ocr_unified import boxes_path, meta_path  # adjust if your module name differs

# light deps (always available)
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# heavy deps (optional, loaded lazily and locally)
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

# -------------------------------------------------------------------
# Local model roots (adapted to your screenshot)
# -------------------------------------------------------------------
DEFAULT_MODELS_ROOT = Path("src/models").resolve()

# sentence-transformers/all-MiniLM-L6-v2 variants you have
CANDIDATES_MINILM = [
    DEFAULT_MODELS_ROOT / "sentence-transformers_all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
    DEFAULT_MODELS_ROOT / "all-MiniLM-L6-v2",
]
# distilbert-base-uncased
CANDIDATES_DISTIL = [
    DEFAULT_MODELS_ROOT / "distilbert-base-uncased",
    DEFAULT_MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
]
# microsoft/layoutlmv3-base
CANDIDATES_LAYOUT = [
    DEFAULT_MODELS_ROOT / "microsoft_layoutlmv3-base",
    DEFAULT_MODELS_ROOT / "microsoft__layoutlmv3-base",
    DEFAULT_MODELS_ROOT / "layoutlmv3-base",
]

def _first_dir(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None

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

# caches
_loaded = {"minilm": None, "distil": None, "layout": None}

def _load_minilm(root: Optional[Path]) -> Optional[SentenceTransformer]:
    if _loaded["minilm"] is not None: return _loaded["minilm"]
    if SentenceTransformer is None: return None
    p = _first_dir([(root or DEFAULT_MODELS_ROOT) / "sentence-transformers_all-MiniLM-L6-v2", *CANDIDATES_MINILM])
    if not p: return None
    _loaded["minilm"] = SentenceTransformer(str(p))
    return _loaded["minilm"]

class _DistilLocal:  # simple bag
    def __init__(self, tok, mdl, dev):
        self.tok, self.mdl, self.dev = tok, mdl, dev

def _load_distil(root: Optional[Path]) -> Optional[_DistilLocal]:
    if _loaded["distil"] is not None: return _loaded["distil"]
    if AutoTokenizer is None or AutoModel is None: return None
    p = _first_dir([(root or DEFAULT_MODELS_ROOT) / "distilbert-base-uncased", *CANDIDATES_DISTIL])
    if not p: return None
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    _loaded["distil"] = _DistilLocal(tok, mdl, dev)
    return _loaded["distil"]

def _load_layout(root: Optional[Path]):
    if _loaded["layout"] is not None: return _loaded["layout"]
    if AutoProcessor is None or LayoutLMv3Model is None: return None
    p = _first_dir([(root or DEFAULT_MODELS_ROOT) / "microsoft_layoutlmv3-base", *CANDIDATES_LAYOUT])
    if not p: return None
    proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
    mdl  = LayoutLMv3Model.from_pretrained(str(p), local_files_only=True)
    dev  = "cuda" if torch and torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    _loaded["layout"] = {"proc": proc, "mdl": mdl, "device": dev}
    return _loaded["layout"]

# -------- request / response --------
class LocateReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    models_root: Optional[str] = None  # allow override

@router.post("/locate")
def locate(req: LocateReq):
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    tokens = json.loads(bp.read_text())
    meta   = json.loads(mp.read_text())
    pages: Dict[int, List[Dict[str,Any]]] = {}
    for t in tokens:
        pages.setdefault(int(t["page"]), []).append(t)
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
    max_w = max(4, int(req.max_window))

    def pick(scored: List[Dict[str,Any]]):
        if not scored: return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    out: Dict[str, Any] = {"autolocate": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # --- ALWAYS: autolocate + tfidf ---
    for method in ("autolocate","tfidf"):
        scored: List[Dict[str,Any]] = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 1500: break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)

                if method == "autolocate":
                    s = float(_rfuzz.QRatio(val_n, stext)) / 100.0
                else:
                    s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0,0], 0, 1))
                    s_ctx  = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                    s_comb = float(np.clip(cosine_similarity(vec.transform([combo]),  vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                    s = 0.75*s_span + 0.25*max(s_ctx, s_comb)

                s = max(0.0, s - 0.12*_line_penalty(span))
                scored.append({"page": pg, "rect": rect, "score": s})
        out[method] = pick(scored)

    # --- BEST-EFFORT: MiniLM ---
    try:
        mroot = Path(req.models_root).resolve() if req.models_root else None
        m = _load_minilm(mroot)
        if m is not None:
            scored: List[Dict[str,Any]] = []
            for pg, toks in pages.items():
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000: break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    E = m.encode([combo, ctx], convert_to_numpy=True, normalize_embeddings=True)
                    s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    s = max(0.0, s - 0.12*_line_penalty(span))
                    scored.append({"page": pg, "rect": rect, "score": s})
            out["minilm"] = pick(scored)
    except Exception:
        out["minilm"] = None

    # --- BEST-EFFORT: DistilBERT ---
    try:
        d = _load_distil(mroot if 'mroot' in locals() else None)
        if d is not None and torch is not None:
            scored: List[Dict[str,Any]] = []
            with torch.no_grad():
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 1000: break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        tok = d.tok([combo, ctx], padding=True, truncation=True,
                                    return_tensors="pt", max_length=256).to(d.dev)
                        hs = d.mdl(**tok).last_hidden_state
                        mask = tok["attention_mask"].unsqueeze(-1)
                        summed = (hs * mask).sum(1)
                        counts = mask.sum(1).clamp(min=1)
                        emb = torch.nn.functional.normalize(summed / counts, dim=1)
                        s = float((emb[0] @ emb[1].T).item())
                        s = max(0.0, s - 0.12*_line_penalty(span))
                        scored.append({"page": pg, "rect": rect, "score": s})
            out["distilbert"] = pick(scored)
    except Exception:
        out["distilbert"] = None

    # --- BEST-EFFORT: LayoutLMv3 (proxy score using TF-IDF + key vicinity) ---
    try:
        L = _load_layout(mroot if 'mroot' in locals() else None)
        if L is not None:
            scored: List[Dict[str,Any]] = []
            for pg, toks in pages.items():
                vec = tfidf[pg]
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000: break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    base = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0,0], 0, 1))
                    near = 0
                    # small key proximity boost
                    x0 = rect["x0"]-80; y0 = rect["y0"]-40; x1 = rect["x1"]+80; y1 = rect["y1"]+40
                    kwords = [w for w in _norm(req.key).split() if len(w) >= 2]
                    for tkn in toks:
                        if tkn["x1"]<x0 or tkn["x0"]>x1 or tkn["y1"]<y0 or tkn["y0"]>y1: continue
                        tx = _norm(tkn.get("text") or "")
                        if any(w in tx for w in kwords): near += 1
                    s = float(min(1.0, base + 0.05*min(near, 6)))
                    s = max(0.0, s - 0.12*_line_penalty(span))
                    scored.append({"page": pg, "rect": rect, "score": s})
            out["layoutlmv3"] = pick(scored)
    except Exception:
        out["layoutlmv3"] = None

    return {"hits": out, "pages": meta.get("pages", [])}
