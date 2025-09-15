# match_router.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rapidfuzz import fuzz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Optional BERT (lazy) ----
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

router = APIRouter(prefix="/match", tags=["match"])

# ========================
# Utilities
# ========================

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _fuzzy01(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(_norm(a), _norm(b)) / 100.0

def _build_tfidf(corpus: List[str]):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

def _qvec(vec, q: str):
    return vec.transform([q])

def _union_rect(span):
    x0 = min(t["bbox"][0] for t in span); y0 = min(t["bbox"][1] for t in span)
    x1 = max(t["bbox"][2] for t in span); y1 = max(t["bbox"][3] for t in span)
    return [float(x0), float(y0), float(x1), float(y1)]

def _concat_text(span):
    return _norm(" ".join(t.get("text", "") for t in span if t.get("text")))

def _group_by_page(tokens):
    pages = {}
    for t in tokens:
        pages.setdefault(int(t.get("page", 1)), []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))  # top->bottom, left->right
    return pages

def _find_candidate_spans(value_norm: str, tokens_page: List[Dict[str, Any]], max_win: int = 12):
    """Slide a window; when window text contains value_norm, emit a tight span+rect."""
    out = []
    n = len(tokens_page)
    for i in range(n):
        accum = []
        text = ""
        for w in range(max_win):
            j = i + w
            if j >= n:
                break
            tok = tokens_page[j]
            t = _norm(tok.get("text", ""))
            if not t:
                continue
            accum.append(tok)
            text = _concat_text(accum)
            if value_norm and value_norm in text:
                out.append({"span": list(accum), "rect": _union_rect(accum), "text": text})
                if w >= 3:  # keep box tight
                    break
    # dedup by coarse rect grid
    dedup, seen = [], set()
    for c in out:
        x0, y0, x1, y1 = c["rect"]
        key = (round(x0 / 5), round(y0 / 5), round((x1 - x0) / 5), round((y1 - y0) / 5))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup

def _context_snippet(tokens_page: List[Dict[str, Any]], span, px_margin=120, py_margin=35):
    R = _union_rect(span)
    cx0 = R[0] - px_margin; cy0 = R[1] - py_margin
    cx1 = R[2] + px_margin; cy1 = R[3] + py_margin
    bag = [
        t for t in tokens_page
        if not (t["bbox"][2] < cx0 or t["bbox"][0] > cx1 or t["bbox"][3] < cy0 or t["bbox"][1] > cy1)
    ]
    bag.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return _concat_text(bag)

# ========================
# BERT lazy singleton
# ========================

_RERANK = {"tok": None, "mdl": None, "dev": "cpu"}

def _ensure_reranker(model_name: str = "distilbert-base-uncased"):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("Transformers not installed on server.")
    if _RERANK["mdl"] is not None:
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "dev": dev})
    print(f"[match_router] DistilBERT loaded on {dev} ({model_name})")

def _embed(texts: List[str]):
    tok, mdl, dev = _RERANK["tok"], _RERANK["mdl"], _RERANK["dev"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state  # [B,T,H]
        mask = batch["attention_mask"].unsqueeze(-1)      # [B,T,1]
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        return torch.nn.functional.normalize(emb, p=2, dim=1)

# ========================
# API Models
# ========================

class OCRToken(BaseModel):
    text: str
    bbox: List[float]   # [x0,y0,x1,y1] in your PEC / page space
    page: int = 1

class MatchReq(BaseModel):
    ocr_tokens: List[OCRToken]
    key: str
    field: str
    llm_value: str
    use_bert: bool = True
    bert_model: Optional[str] = "distilbert-base-uncased"

class MethodBox(BaseModel):
    text: Optional[str] = None
    conf: Optional[float] = None
    bbox: Optional[List[float]] = None
    page: Optional[int] = None

class MatchResp(BaseModel):
    field: str
    llm_value: str
    autolocate: Optional[MethodBox] = None
    bert: Optional[MethodBox] = None
    tfidf: Optional[MethodBox] = None

# ========================
# Endpoint
# ========================

@router.post("/field", response_model=MatchResp)
def match_field(req: MatchReq):
    tokens = [t.model_dump() for t in req.ocr_tokens]
    field, key, llm = req.field, req.key, req.llm_value

    # ---- 1) Autolocate (fuzzy vs LLM value)
    best_i, best_s = -1, -1.0
    for i, t in enumerate(tokens):
        s = _fuzzy01(t["text"], llm)
        if s > best_s:
            best_s, best_i = s, i
    auto = None
    if best_i >= 0:
        tt = tokens[best_i]
        auto = MethodBox(text=tt["text"], conf=round(float(best_s), 4), bbox=tt["bbox"], page=int(tt.get("page", 1)))

    # ---- 2) TF-IDF (key+value) vs tokens
    tfidf = None
    if tokens:
        vec, X = _build_tfidf([_norm(t["text"]) for t in tokens])
        q = ((_norm(key) + " ") * 7 + (_norm(llm) + " ") * 3).strip()
        qX = _qvec(vec, q)
        sims = cosine_similarity(qX, X).ravel()
        # small boosts
        lv = _norm(llm); lv_len = len(lv)
        boost = np.zeros_like(sims)
        for i, t in enumerate(tokens):
            txt = _norm(t["text"])
            if lv and txt == lv: boost[i] += 0.15
            if lv_len > 0 and len(txt) > lv_len + 3: boost[i] -= 0.05
            k = _norm(key)
            if k and k in txt: boost[i] += 0.03
        sc = sims + boost
        ti = int(sc.argmax()) if len(sc) else -1
        if ti >= 0:
            tt = tokens[ti]
            tfidf = MethodBox(text=tt["text"], conf=round(float(sc[ti]), 4), bbox=tt["bbox"], page=int(tt.get("page", 1)))

    # ---- 3) BERT (optional): semantic span rerank around LLM value
    bert = None
    if req.use_bert and llm:
        if AutoTokenizer is None:
            # transformers not installed; skip, but do NOT fail the call
            pass
        else:
            try:
                _ensure_reranker(req.bert_model or "distilbert-base-uncased")
                by_pg = _group_by_page(tokens)
                vnorm = _norm(llm)
                best = None
                for pg, toks in by_pg.items():
                    spans = _find_candidate_spans(vnorm, toks)
                    if not spans:
                        continue
                    cands = [
                        {"page": pg, "rect": s["rect"], "text": s["text"], "ctx": _context_snippet(toks, s["span"])}
                        for s in spans
                    ]
                    qtext = f"{key}: {llm}".strip()
                    texts = [qtext] + [c["ctx"] for c in cands]
                    embs = _embed(texts)
                    qv, M = embs[0:1], embs[1:]
                    sims = (M @ qv.T).squeeze(1)  # cosine; vectors are L2-normalized
                    bi = int(torch.argmax(sims).item())
                    cand = {**cands[bi], "score": float(sims[bi].item())}
                    if best is None or cand["score"] > best["score"]:
                        best = cand
                if best:
                    bert = MethodBox(
                        text=best["text"],
                        conf=round(float(best["score"]), 4),
                        bbox=best["rect"],
                        page=int(best["page"])
                    )
            except Exception as e:
                # Fail-safe: log and keep going
                print(f"[match_router] BERT disabled: {e}")

    return MatchResp(field=field, llm_value=llm, autolocate=auto, bert=bert, tfidf=tfidf)