# Semantic reranking (optional)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

_RERANK = {"tok": None, "mdl": None, "device": "cpu"}  # DistilBERT

def _ensure_reranker(model_name_or_path: str = "distilbert-base-uncased"):
    if AutoTokenizer is None or AutoModel is None:
        raise HTTPException(500, "Transformers not available on server.")
    if _RERANK["mdl"] is not None:
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModel.from_pretrained(model_name_or_path)
    mdl.to(dev)
    mdl.eval()
    _RERANK["tok"] = tok
    _RERANK["mdl"] = mdl
    _RERANK["device"] = dev
    print(f"[ocr_lasso] DistilBERT reranker loaded on {dev} ({model_name_or_path})")


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _union_rect(span):
    x0 = min(t["x0"] for t in span); y0 = min(t["y0"] for t in span)
    x1 = max(t["x1"] for t in span); y1 = max(t["y1"] for t in span)
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}

def _group_tokens_by_page(tokens):
    by = {}
    for t in tokens:
        by.setdefault(int(t["page"]), []).append(t)
    for arr in by.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))
    return by

def _window_tokens(tokens, i, j):
    return tokens[i:j+1]

def _concat_text(span):
    return _norm_text(" ".join(t.get("text","") for t in span if t.get("text")))


def _find_candidate_spans(value: str, tokens_page):
    """
    Slide over tokens and find short windows whose concatenated text contains the value (normalized).
    Returns a list of dicts: { 'span': [tokens], 'rect': {...}, 'text': '...' }
    """
    v = _norm_text(value)
    out = []
    n = len(tokens_page)
    for i in range(n):
        accum = []
        text = ""
        for w in range(12):  # small windows keep boxes tight
            j = i + w
            if j >= n: break
            tok = tokens_page[j]
            t = _norm_text(tok.get("text",""))
            if not t: continue
            accum.append(tok)
            text = _concat_text(accum)
            if v and v in text:
                out.append({
                    "span": list(accum),
                    "rect": _union_rect(accum),
                    "text": text
                })
                # try a couple expansions but avoid growing too far
                if w >= 3: break
    # de-duplicate near-identical rects by (x,y,width,height)
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
    """Grab nearby tokens around the span rectangle to form a local snippet."""
    R = _union_rect(span)
    cx0 = R["x0"] - px_margin; cy0 = R["y0"] - py_margin
    cx1 = R["x1"] + px_margin; cy1 = R["y1"] + py_margin
    bag = [t for t in tokens_page
           if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return _concat_text(bag)



def _embed(texts):
    tok = _RERANK["tok"]; mdl = _RERANK["mdl"]; dev = _RERANK["device"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state  # [B, T, H]
        # mean pooling (mask-aware)
        mask = batch["attention_mask"].unsqueeze(-1)  # [B,T,1]
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb



class RerankReq(BaseModel):
    doc_id: str
    key: str               # e.g., "billing.zip"
    value: str             # e.g., "46204"
    model: Optional[str] = None  # path or HF id for DistilBERT
    topk: int = 8

@router.post("/rerank")
async def rerank(req: RerankReq):
    """
    Re-rank candidate boxes that match `value` using DistilBERT semantic similarity with a query
    formed from (key + value). Returns the best rect (and some alternates) on the correct page.
    """
    if AutoTokenizer is None:
        raise HTTPException(500, "Transformers not installed on server.")
    _ensure_reranker(req.model or "distilbert-base-uncased")

    bp = boxes_path(req.doc_id)
    mp = meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "Document tokens/meta missing.")
    tokens = json.loads(bp.read_text())
    by_pg = _group_tokens_by_page(tokens)

    # collect candidates across pages
    vnorm = _norm_text(req.value)
    cands = []
    for pg, toks in by_pg.items():
        spans = _find_candidate_spans(vnorm, toks)
        for s in spans:
            local = _context_snippet(toks, s["span"])
            cands.append({
                "page": pg,
                "rect": s["rect"],
                "span_text": s["text"],
                "context": local
            })

    if not cands:
        return {"best": None, "alts": []}

    # Build query from key + value so "billing zip" vs "shipping zip" matters
    query = f"{req.key}: {req.value}".strip()
    texts = [query] + [c["context"] for c in cands]
    embs = _embed(texts)
    q = embs[0:1]
    M = embs[1:]
    sims = (M @ q.t()).squeeze(1)  # cosine because we normalized

    scored = sorted(
        [{"score": float(sims[i].item()), **cands[i]} for i in range(len(cands))],
        key=lambda x: x["score"],
        reverse=True,
    )

    best = scored[0]
    alts = scored[1: min(len(scored), req.topk)]
    return {
        "best": {"page": best["page"], "rect": best["rect"], "score": best["score"]},
        "alts": [{"page": a["page"], "rect": a["rect"], "score": a["score"]} for a in alts]
    }




