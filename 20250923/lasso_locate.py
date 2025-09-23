# File: src/routers/lasso_locate.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Iterable, Tuple
from pathlib import Path
import json, time

# Reuse paths from your unified OCR router
from .ocr_unified import boxes_path, meta_path  # keep these imports as-is

# Light deps only
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(prefix="/lasso", tags=["lasso-locate"])


# ------------------------------ helpers ------------------------------

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\u00A0", " ").split())

def _union(span: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "x0": float(min(t["x0"] for t in span)),
        "y0": float(min(t["y0"] for t in span)),
        "x1": float(max(t["x1"] for t in span)),
        "y1": float(max(t["y1"] for t in span)),
    }

def _line_penalty(span: List[Dict[str, Any]]) -> float:
    """Penalty for multi-line unions; keeps boxes tight."""
    if len(span) <= 1:
        return 0.0
    ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
    spread = ys[-1] - ys[0]
    avg_h = float(np.mean([t["y1"] - t["y0"] for t in span]))
    return max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)

def _group_tokens_by_page(tokens: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        pages.setdefault(int(t["page"]), []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))
    return pages

def _concat_text(span: Iterable[Dict[str, Any]]) -> str:
    return _norm(" ".join((t.get("text") or "") for t in span if t.get("text")))

def _context(tokens: List[Dict[str, Any]], span: List[Dict[str, Any]], px=120, py=35) -> str:
    """Small neighborhood text around span (still light)."""
    R = _union(span)
    cx0, cy0, cx1, cy1 = R["x0"] - px, R["y0"] - py, R["x1"] + px, R["y1"] + py
    bag = [t for t in tokens if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _concat_text(bag)

def _slide(tokens: List[Dict[str, Any]], max_w: int = 12) -> Iterable[List[Dict[str, Any]]]:
    """Generate growing token windows across a page, skipping empty tokens."""
    n = len(tokens)
    for i in range(n):
        span: List[Dict[str, Any]] = []
        for w in range(max_w):
            j = i + w
            if j >= n:
                break
            txt = (tokens[j].get("text") or "").strip()
            if not txt:
                continue
            span.append(tokens[j])
            yield span


# ------------------------------ I/O models ------------------------------

class MatchOptions(BaseModel):
    max_window: int = 12
    fast_only: bool = False  # if True => return after fuzzy (and possibly tfidf) quickly

class MatchReq(BaseModel):
    doc_id: str
    key: str
    value: str
    options: Optional[MatchOptions] = None

# wire format we send back to the client (matches your api.ts)
def _pick(scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not scored:
        return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    b = scored[0]
    return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}


# ------------------------------ endpoint ------------------------------

@router.post("/match/field")
def match_field(req: MatchReq):
    """Fast-first matcher: fuzzy immediately, tfidf next. Heavy models are disabled."""
    opts = req.options or MatchOptions()
    max_w = max(4, int(opts.max_window))

    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    try:
        tokens = json.loads(bp.read_text())
    except Exception:
        raise HTTPException(500, "Failed to read boxes.json")

    pages = _group_tokens_by_page(tokens)
    key_n = _norm(req.key)
    val_n = _norm(req.value)

    results: Dict[str, Any] = {"fuzzy": None, "tfidf": None, "minilm": None, "distilbert": None}

    # ---------- 0) FUZZY FIRST (independent, immediate) ----------
    t0 = time.time()
    FUZZY_BUDGET_S = 0.7   # ~700ms budget so UI sees green quickly (tune)
    FUZZY_SPAN_CAP = 3500  # safety cap on windows scored

    fuzzy_scored: List[Dict[str, Any]] = []
    spans_seen = 0
    for pg, toks in pages.items():
        for span in _slide(toks, max_w=max_w):
            spans_seen += 1
            if spans_seen > FUZZY_SPAN_CAP:
                break
            rect = _union(span)
            span_text = _concat_text(span)
            s = float(_rfuzz.QRatio(val_n, span_text)) / 100.0
            s = max(0.0, s - 0.12 * _line_penalty(span))
            fuzzy_scored.append({"page": pg, "rect": rect, "score": s})
        if time.time() - t0 > FUZZY_BUDGET_S:
            break

    results["fuzzy"] = _pick(fuzzy_scored)

    # If caller asked for fast_only, or we already exceeded our fuzzy budget, return now.
    if opts.fast_only or (time.time() - t0 > FUZZY_BUDGET_S and results["fuzzy"] is not None):
        return {"methods": results}

    # ---------- 1) TF-IDF (light; value-centric + context) ----------
    tfidf_per_page: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        corpus = [" ".join((t.get("text") or "").strip() for t in toks)]
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        try:
            vec.fit(corpus)
        except ValueError:
            vec.fit(["placeholder"])
        tfidf_per_page[pg] = vec

    tf_scored: List[Dict[str, Any]] = []
    TF_SPAN_CAP = 2500
    for pg, toks in pages.items():
        vec = tfidf_per_page[pg]
        cnt = 0
        for span in _slide(toks, max_w=max_w):
            cnt += 1
            if cnt > TF_SPAN_CAP:
                break
            rect = _union(span)
            span_text = _concat_text(span)
            ctx = _context(toks, span)
            s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([span_text]))[0, 0], 0, 1))
            s_ctx = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0, 0], 0, 1)) if ctx else 0.0
            s = 0.80 * s_span + 0.20 * s_ctx
            s = max(0.0, s - 0.12 * _line_penalty(span))
            tf_scored.append({"page": pg, "rect": rect, "score": s})

    results["tfidf"] = _pick(tf_scored)

    # ---------- 2) Heavy models (disabled on purpose to keep UI snappy) ----------
    # results["minilm"] = None
    # results["distilbert"] = None

    return {"methods": results}
