# File: src/routers/lasso_locate.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import numpy as np

# reuse file helpers from your unified OCR router
from .ocr_unified import boxes_path, meta_path  # adjust relative import if needed

# light deps (bundled)
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# heavy deps (optional; all local-only loads)
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
# Local model roots (based on your tree)
# -------------------------------------------------------------------
MODELS = Path("src/models").resolve()

CANDIDATES_MINILM = [
    MODELS / "sentence-transformers_all-MiniLM-L6-v2",
    MODELS / "sentence-transformers__all-MiniLM-L6-v2",
    MODELS / "all-MiniLM-L6-v2",
]

CANDIDATES_DISTIL = [
    MODELS / "distilbert-base-uncased",
    MODELS / "DistilBERT" / "distilbert-base-uncased",
]

CANDIDATES_LAYOUT = [
    MODELS / "microsoft_layoutlmv3-base",
    MODELS / "microsoft__layoutlmv3-base",
    MODELS / "layoutlmv3-base",
]


def _first_dir(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists() and p.is_dir():
            return p
    return None


def _norm(s: str) -> str:
    return " ".join((s or "").replace("\u00A0", " ").strip().lower().split())


def _union(span: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "x0": float(min(t["x0"] for t in span)),
        "y0": float(min(t["y0"] for t in span)),
        "x1": float(max(t["x1"] for t in span)),
        "y1": float(max(t["y1"] for t in span)),
    }


def _context(tokens: List[Dict[str, Any]], span: List[Dict[str, Any]], px=120, py=35) -> str:
    R = _union(span)
    cx0, cy0, cx1, cy1 = R["x0"] - px, R["y0"] - py, R["x1"] + px, R["y1"] + py
    bag = [t for t in tokens if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _norm(" ".join((t.get("text") or "") for t in bag if t.get("text")))


def _slide(tokens: List[Dict[str, Any]], max_w=12):
    """Yield growing windows of tokens: [t[i]], [t[i],t[i+1]], ... up to max_w."""
    n = len(tokens)
    for i in range(n):
        acc = []
        for w in range(max_w):
            j = i + w
            if j >= n:
                break
            txt = (tokens[j].get("text") or "").strip()
            if not txt:
                continue
            acc.append(tokens[j])
            yield acc


def _line_penalty(span: List[Dict[str, Any]]) -> float:
    if len(span) <= 1:
        return 0.0
    ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
    spread = ys[-1] - ys[0]
    avg_h = float(np.mean([t["y1"] - t["y0"] for t in span]))
    return max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)


# ---------------- model caches ----------------
_loaded: Dict[str, Any] = {"minilm": None, "distil": None, "layout": None}


def _load_minilm() -> Optional[SentenceTransformer]:
    if _loaded["minilm"] is not None:
        return _loaded["minilm"]
    if SentenceTransformer is None:
        return None
    root = _first_dir(CANDIDATES_MINILM)
    if not root:
        return None
    model = SentenceTransformer(str(root))
    _loaded["minilm"] = model
    return model


class _DistilLocal:
    def __init__(self, tok, mdl, dev):
        self.tok, self.mdl, self.dev = tok, mdl, dev


def _load_distil() -> Optional[_DistilLocal]:
    if _loaded["distil"] is not None:
        return _loaded["distil"]
    if AutoTokenizer is None or AutoModel is None:
        return None
    root = _first_dir(CANDIDATES_DISTIL)
    if not root:
        return None
    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(root), local_files_only=True)
    dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    h = _DistilLocal(tok, mdl, dev)
    _loaded["distil"] = h
    return h


def _load_layout():
    if _loaded["layout"] is not None:
        return _loaded["layout"]
    if AutoProcessor is None or LayoutLMv3Model is None:
        return None
    root = _first_dir(CANDIDATES_LAYOUT)
    if not root:
        return None
    proc = AutoProcessor.from_pretrained(str(root), local_files_only=True)
    mdl = LayoutLMv3Model.from_pretrained(str(root), local_files_only=True)
    dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    _loaded["layout"] = {"proc": proc, "mdl": mdl, "device": dev}
    return _loaded["layout"]


# ---------------- request/response ----------------
class LocateReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12  # token window per candidate region


@router.post("/locate")
def locate(req: LocateReq):
    # ensure OCR artifacts exist
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    tokens = json.loads(bp.read_text())
    meta = json.loads(mp.read_text())

    # group tokens by page & sort
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        pages.setdefault(int(t["page"]), []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    # per-page tf-idf fit once (used by tfidf & layout proxy)
    tfidf: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        try:
            vec.fit([txt])
        except ValueError:
            vec.fit(["placeholder"])
        tfidf[pg] = vec

    key_n = _norm(req.key)
    val_n = _norm(req.value)
    combo = f"{key_n} {val_n}".strip()
    max_w = max(4, int(req.max_window))

    def pick(scored: List[Dict[str, Any]]):
        if not scored:
            return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    results: Dict[str, Any] = {
        "autolocate": None,
        "tfidf": None,
        "minilm": None,
        "distilbert": None,
        "layoutlmv3": None,
    }

    # ---------------- fuzzy (value) + tfidf (value/context) ----------------
    for method in ("autolocate", "tfidf"):
        scored: List[Dict[str, Any]] = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 1500:
                    break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)

                if method == "autolocate":
                    s = float(_rfuzz.QRatio(val_n, stext)) / 100.0
                else:
                    # TF-IDF: span text; plus context & combo blend
                    s_span = float(
                        np.clip(
                            cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0, 0], 0, 1
                        )
                    )
                    s_ctx = (
                        float(
                            np.clip(
                                cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0, 0], 0, 1
                            )
                        )
                        if ctx
                        else 0.0
                    )
                    s_comb = (
                        float(
                            np.clip(
                                cosine_similarity(vec.transform([combo]), vec.transform([ctx]))[0, 0], 0, 1
                            )
                        )
                        if ctx
                        else 0.0
                    )
                    s = 0.75 * s_span + 0.25 * max(s_ctx, s_comb)

                s = max(0.0, s - 0.12 * _line_penalty(span))
                scored.append({"page": pg, "rect": rect, "score": s})
        results[method] = pick(scored)

    # ---------------- MiniLM (local) ----------------
    try:
        m = _load_minilm()
        if m is not None:
            scored: List[Dict[str, Any]] = []
            for pg, toks in pages.items():
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000:
                        break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    E = m.encode([combo, ctx], convert_to_numpy=True, normalize_embeddings=True)
                    s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    s = max(0.0, s - 0.12 * _line_penalty(span))
                    scored.append({"page": pg, "rect": rect, "score": s})
            results["minilm"] = pick(scored)
    except Exception:
        results["minilm"] = None

    # ---------------- DistilBERT (local) ----------------
    try:
        d = _load_distil()
        if d is not None and torch is not None:
            scored: List[Dict[str, Any]] = []
            with torch.no_grad():
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 1000:
                            break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        tok = d.tok(
                            [combo, ctx],
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=256,
                        ).to(d.dev)
                        hs = d.mdl(**tok).last_hidden_state
                        mask = tok["attention_mask"].unsqueeze(-1)
                        summed = (hs * mask).sum(1)
                        counts = mask.sum(1).clamp(min=1)
                        emb = torch.nn.functional.normalize(summed / counts, dim=1)
                        s = float((emb[0] @ emb[1].T).item())
                        s = max(0.0, s - 0.12 * _line_penalty(span))
                        scored.append({"page": pg, "rect": rect, "score": s})
            results["distilbert"] = pick(scored)
    except Exception:
        results["distilbert"] = None

    # ---------------- LayoutLMv3 (proxy using TF-IDF + key vicinity) ----------------
    # We do NOT run the transformer (heavy); we give a quick heuristic signal.
    try:
        L = _load_layout()
        if L is not None:
            scored: List[Dict[str, Any]] = []
            for pg, toks in pages.items():
                vec = tfidf[pg]
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000:
                        break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    base = float(
                        np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0, 0], 0, 1)
                    )
                    # small key proximity boost
                    x0 = rect["x0"] - 80
                    y0 = rect["y0"] - 40
                    x1 = rect["x1"] + 80
                    y1 = rect["y1"] + 40
                    kwords = [w for w in key_n.split() if len(w) >= 2]
                    near = 0
                    for tkn in toks:
                        if tkn["x1"] < x0 or tkn["x0"] > x1 or tkn["y1"] < y0 or tkn["y0"] > y1:
                            continue
                        tx = _norm(tkn.get("text") or "")
                        if any(w in tx for w in kwords):
                            near += 1
                    s = float(min(1.0, base + 0.05 * min(near, 6)))
                    s = max(0.0, s - 0.12 * _line_penalty(span))
                    scored.append({"page": pg, "rect": rect, "score": s})
            results["layoutlmv3"] = pick(scored)
    except Exception:
        results["layoutlmv3"] = None

    # --- FINAL SHAPE expected by the UI ---
    methods = {
        "fuzzy": results.get("autolocate"),
        "tfidf": results.get("tfidf"),
        "minilm": results.get("minilm"),
        "distilbert": results.get("distilbert"),
        "layoutlmv3": results.get("layoutlmv3"),
    }

    def _norm_method(m):
        if not m or "page" not in m or "rect" not in m:
            return None
        r = m["rect"]
        return {
            "page": int(m["page"]),
            "rect": {"x0": float(r["x0"]), "y0": float(r["y0"]), "x1": float(r["x1"]), "y1": float(r["y1"])},
            "score": float(m.get("score", 0.0)),
        }

    methods = {k: _norm_method(v) for k, v in methods.items()}
    return {"methods": methods, "pages": meta.get("pages", [])}
