from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json, os

# Paths from OCR router
from .ocr_unified import boxes_path, meta_path, MODELS_ROOT

# Light deps
import numpy as np
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Heavy deps (optional, local only)
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

# -------- Helpers -------------------------------------------------------------
def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\u00A0", " ").split())

def _group_by_page(tokens: List[Dict[str, Any]]):
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        pg = int(t["page"])
        pages.setdefault(pg, []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))
    return pages

def _union_rect(span: List[Dict[str, Any]]):
    return {
        "x0": float(min(t["x0"] for t in span)),
        "y0": float(min(t["y0"] for t in span)),
        "x1": float(max(t["x1"] for t in span)),
        "y1": float(max(t["y1"] for t in span)),
    }

def _slide(tokens_page: List[Dict[str, Any]], max_w=12):
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

def _context(tokens: List[Dict[str, Any]], span: List[Dict[str, Any]], px=120, py=35) -> str:
    R = _union_rect(span)
    cx0, cy0, cx1, cy1 = R["x0"]-px, R["y0"]-py, R["x1"]+px, R["y1"]+py
    bag = [t for t in tokens if not (t["x1"]<cx0 or t["x0"]>cx1 or t["y1"]<cy0 or t["y0"]>cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _norm(" ".join((t.get("text") or "") for t in bag if t.get("text")))

def _line_penalty(span: List[Dict[str, Any]]) -> float:
    if len(span) <= 1: return 0.0
    ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
    spread = ys[-1] - ys[0]
    avg_h = float(np.mean([t["y1"] - t["y0"] for t in span]))
    return max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)

def _pick(scored: List[Dict[str, Any]]):
    if not scored: return None
    scored.sort(key=lambda x: x["score"], reverse=True)
    b = scored[0]
    return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

# -------- Local loaders -------------------------------------------------------
_RERANK = {"tok": None, "mdl": None, "device": "cpu"}  # DistilBERT
_MINILM = None                                         # Sentence-Transformers

def _ensure_reranker():
    if AutoTokenizer is None or AutoModel is None: return
    if _RERANK["mdl"] is not None: return
    candidates = [
        MODELS_ROOT / "distilbert-base-uncased",
        MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
    ]
    print("[Distil] MODELS_ROOT =", MODELS_ROOT)
    for p in candidates:
        print("[Distil] candidate:", p, "exists:", p.exists())
    local_dir = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if not local_dir:
        print("[Distil] weights not found; skipping")
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(local_dir), local_files_only=True)
    mdl.to(dev).eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
    print(f"[Distil] loaded from {local_dir} on {dev}")

def _load_minilm():
    global _MINILM
    if _MINILM is not None: return _MINILM
    if SentenceTransformer is None: return None
    candidates = [
        MODELS_ROOT / "sentence-transformers_all-MiniLM-L6-v2",
        MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
        MODELS_ROOT / "sentence-transformers" / "all-MiniLM-L6-v2",
        MODELS_ROOT / "all-MiniLM-L6-v2",
    ]
    print("[MiniLM] MODELS_ROOT =", MODELS_ROOT)
    for p in candidates:
        print("[MiniLM] candidate:", p, "exists:", p.exists())
    path = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if not path:
        print("[MiniLM] no local folder matched; skipping")
        return None
    _MINILM = SentenceTransformer(str(path))
    print(f"[MiniLM] loaded from {path}")
    return _MINILM

def _embed_distil(texts: List[str]) -> np.ndarray:
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

# -------- API -----------------------------------------------------------------
class MatchReq(BaseModel):
    doc_id: str
    key: str
    value: str
    max_window: int = 12
    models: Optional[List[str]] = None     # ["fuzzy","tfidf"] etc.
    wait_heavy: bool = False               # False => heavy are best-effort

@router.post("/match/field")
def match_field(req: MatchReq):
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    tokens = json.loads(bp.read_text())
    pages = _group_by_page(tokens)

    # Fit per-page TF-IDF (shared by tfidf scoring)
    tfidf: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        v = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        try: v.fit([txt])
        except ValueError: v.fit(["placeholder"])
        tfidf[pg] = v

    key_n = _norm(req.key)
    val_n = _norm(req.value)
    combo = f"{key_n} {val_n}".strip()

    results = {"fuzzy": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    want = set([m.lower() for m in (req.models or ["fuzzy","tfidf","minilm","distilbert"])])
    do_fuzzy   = "fuzzy" in want
    do_tfidf   = "tfidf" in want
    do_minilm  = (not req.wait_heavy) and ("minilm" in want)
    do_distil  = (not req.wait_heavy) and ("distilbert" in want)

    # ---- 1) Fuzzy (fast)
    if do_fuzzy:
        scored = []
        for pg, toks in pages.items():
            cnt = 0
            for span in _slide(toks, max_w=req.max_window):
                cnt += 1
                if cnt > 1500: break
                rect = _union_rect(span)
                span_text = _norm(" ".join((t.get("text") or "") for t in span))
                s = float(_rfuzz.QRatio(val_n, span_text)) / 100.0
                s = max(0.0, s - 0.12 * _line_penalty(span))
                scored.append({"page": pg, "rect": rect, "score": s})
        results["fuzzy"] = _pick(scored)

    # ---- 2) TF-IDF (fast)
    if do_tfidf:
        scored = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w=req.max_window):
                cnt += 1
                if cnt > 1500: break
                rect = _union_rect(span)
                ctx = _context(toks, span)
                span_text = _norm(" ".join((t.get("text") or "") for t in span))

                cctx = ctx
                if key_n:
                    for kw in key_n.split():
                        if len(kw) >= 2:
                            cctx = cctx.replace(kw, " ")
                    cctx = _norm(cctx)

                s_span  = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([span_text]))[0,0], 0, 1))
                s_ctx   = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([cctx]))[0,0],       0, 1)) if cctx else 0.0
                s_combo = float(np.clip(cosine_similarity(vec.transform([combo]),  vec.transform([cctx]))[0,0],       0, 1)) if cctx else 0.0

                v_toks = [w for w in val_n.split() if w]
                coverage = 0.0
                if v_toks:
                    span_words = span_text.split()
                    covered = sum(1 for w in v_toks if any(_rfuzz.QRatio(w, sw) >= 90 for sw in span_words))
                    coverage = covered / max(1, len(v_toks))

                s = 0.70*s_span + 0.20*max(s_ctx, s_combo) + 0.10*coverage
                s = max(0.0, s - 0.12 * _line_penalty(span))
                scored.append({"page": pg, "rect": rect, "score": s})
        results["tfidf"] = _pick(scored)

    # ---- 3) MiniLM (best-effort)
    if do_minilm:
        try:
            m = _load_minilm()
            if m is not None:
                scored = []
                combo2 = f"{key_n}: {val_n}".strip()
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w=req.max_window):
                        cnt += 1
                        if cnt > 1000: break
                        rect = _union_rect(span)
                        ctx = _context(toks, span)
                        E = m.encode([combo2, ctx], convert_to_numpy=True, normalize_embeddings=True)
                        s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                        s = max(0.0, s - 0.12 * _line_penalty(span))
                        scored.append({"page": pg, "rect": rect, "score": s})
                results["minilm"] = _pick(scored)
            else:
                print("[match] MiniLM not available; skipping")
        except Exception as e:
            print(f"[match] MiniLM error: {e}")
            results["minilm"] = None

    # ---- 4) DistilBERT (best-effort)
    if do_distil:
        try:
            _ensure_reranker()
            if _RERANK["mdl"] is not None and torch is not None:
                scored = []
                combo2 = f"{key_n}: {val_n}".strip()
                with torch.no_grad():
                    for pg, toks in pages.items():
                        cnt = 0
                        for span in _slide(toks, max_w=req.max_window):
                            cnt += 1
                            if cnt > 1000: break
                            rect = _union_rect(span)
                            ctx = _context(toks, span)
                            tok = _RERANK["tok"]([combo2, ctx], padding=True, truncation=True,
                                                 return_tensors="pt", max_length=256).to(_RERANK["device"])
                            hs = _RERANK["mdl"](**tok).last_hidden_state
                            mask = tok["attention_mask"].unsqueeze(-1)
                            summed = (hs * mask).sum(1)
                            counts = mask.sum(1).clamp(min=1)
                            emb = torch.nn.functional.normalize(summed / counts, dim=1)
                            s = float((emb[0] @ emb[1].T).item())
                            s = max(0.0, s - 0.12 * _line_penalty(span))
                            scored.append({"page": pg, "rect": rect, "score": s})
                results["distilbert"] = _pick(scored)
            else:
                print("[match] DistilBERT not available; skipping")
        except Exception as e:
            print(f"[match] DistilBERT error: {e}")
            results["distilbert"] = None

    # LayoutLMv3 deliberately disabled
    results["layoutlmv3"] = None

    return {"doc_id": req.doc_id, "key": req.key, "value": req.value, "methods": results}