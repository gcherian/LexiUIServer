# server_simple.py
# Lightweight: Autolocate (fuzzy) + TF-IDF + (optional) DistilBERT reranker

import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from rapidfuzz import fuzz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- OPTIONAL DistilBERT (lazy) ----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

_RERANK = {"tok": None, "mdl": None, "device": "cpu"}  # lazy singleton

def _ensure_reranker(model_name_or_path: str = "distilbert-base-uncased"):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("Transformers not available on server.")
    if _RERANK["mdl"] is not None:
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModel.from_pretrained(model_name_or_path)
    mdl.to(dev).eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
    print(f"[validate_simple] DistilBERT reranker on {dev} ({model_name_or_path})")

def _embed(texts: List[str]):
    tok, mdl, dev = _RERANK["tok"], _RERANK["mdl"], _RERANK["device"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state  # [B,T,H]
        mask = batch["attention_mask"].unsqueeze(-1)       # [B,T,1]
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

# ---------- Shared utils ----------
def norm_txt(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def fuzzy01(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return fuzz.token_set_ratio(norm_txt(a), norm_txt(b)) / 100.0

def build_tfidf(corpus: List[str]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

def qvec(vec, q: str):
    return vec.transform([q])

def _union_rect(span):
    x0 = min(t["bbox"][0] for t in span); y0 = min(t["bbox"][1] for t in span)
    x1 = max(t["bbox"][2] for t in span); y1 = max(t["bbox"][3] for t in span)
    return [float(x0), float(y0), float(x1), float(y1)]

def _concat_text(span):
    return norm_txt(" ".join(t.get("text","") for t in span if t.get("text")))

def _group_by_page(tokens):
    pages = {}
    for t in tokens:
        pages.setdefault(int(t.get("page",1)), []).append(t)
    # sort LTR, top→bottom if your coords are PDF-space; adjust as needed
    for arr in pages.values():
        arr.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return pages

def _find_candidate_spans(value_norm: str, tokens_page: List[Dict[str,Any]], max_win: int = 12):
    out = []
    n = len(tokens_page)
    for i in range(n):
        accum = []
        text = ""
        for w in range(max_win):
            j = i + w
            if j >= n: break
            tok = tokens_page[j]
            t = norm_txt(tok.get("text",""))
            if not t: continue
            accum.append(tok)
            text = _concat_text(accum)
            if value_norm and value_norm in text:
                out.append({"span": list(accum), "rect": _union_rect(accum), "text": text})
                if w >= 3: break  # keep boxes tight
    # light dedup by coarse rect grid
    dedup, seen = [], set()
    for c in out:
        x0,y0,x1,y1 = c["rect"]
        key = (round(x0/5), round(y0/5), round((x1-x0)/5), round((y1-y0)/5))
        if key in seen: continue
        seen.add(key); dedup.append(c)
    return dedup

def _context_snippet(tokens_page: List[Dict[str,Any]], span, px_margin=120, py_margin=35):
    R = _union_rect(span)
    cx0 = R[0]-px_margin; cy0 = R[1]-py_margin
    cx1 = R[2]+px_margin; cy1 = R[3]+py_margin
    bag = [t for t in tokens_page if not (t["bbox"][2] < cx0 or t["bbox"][0] > cx1 or t["bbox"][3] < cy0 or t["bbox"][1] > cy1)]
    bag.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
    return _concat_text(bag)

# ---------- TF-IDF locator ----------
def pick_best_token_by_tfidf(key: str, llm_value: str, tokens: List[Dict[str, Any]]):
    token_texts = [t["text"] for t in tokens]
    vec, X = build_tfidf([norm_txt(t) for t in token_texts])
    q = ((key or "") + " ") * 7 + ((llm_value or "") + " ") * 3
    q = norm_txt(q)
    qX = qvec(vec, q)
    sims = cosine_similarity(qX, X).ravel()
    lv = norm_txt(llm_value); lv_len = len(lv)
    boost = np.zeros_like(sims)
    for i, t in enumerate(token_texts):
        tnorm = norm_txt(t)
        if lv and tnorm == lv: boost[i] += 0.15
        if lv_len > 0 and len(tnorm) > lv_len + 3: boost[i] -= 0.05
        k = norm_txt(key)
        if k and k in tnorm: boost[i] += 0.03
    score = sims + boost
    idx = int(score.argmax()) if len(score) else -1
    sc = float(score[idx]) if idx >= 0 else 0.0
    return idx, sc

# ---------- Accuracy proxies ----------
def ok_at_fuzzy(candidate_text: str, llm_value: str, thr: float = 0.9) -> int:
    return int(fuzzy01(candidate_text, llm_value) >= thr)

def char_f1(a: str, b: str) -> float:
    a = list(norm_txt(a)); b = list(norm_txt(b))
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if common == 0: return 0.0
    precision = common / max(1, sum(ca.values()))
    recall    = common / max(1, sum(cb.values()))
    return 0.0 if precision+recall==0 else 2*precision*recall/(precision+recall)

# ---------- API models ----------
class OCRToken(BaseModel):
    text: str
    bbox: List[float]  # [x0,y0,x1,y1] in your PEC/PDF space
    page: int = 1

class ValidateItem(BaseModel):
    key: str
    field: str
    llm_value: str

class ValidateRequest(BaseModel):
    ocr_tokens: List[OCRToken]
    items: List[ValidateItem]
    client_autolocate: Dict[str, Dict[str, Any]] | None = None
    use_bert: bool = False
    bert_model: Optional[str] = "distilbert-base-uncased"

class ValidateResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

# ---------- Core: 3 locators ----------
def run_three_locators(ocr_tokens: List[Dict[str, Any]],
                       items: List[Dict[str, Any]],
                       client_autolocate: Optional[Dict[str, Dict[str, Any]]] = None,
                       use_bert: bool = False,
                       bert_model: Optional[str] = "distilbert-base-uncased"):
    rows = []
    pages = _group_by_page(ocr_tokens)
    tokens_all = [t for arr in pages.values() for t in arr]
    token_texts = [t["text"] for t in tokens_all]

    # Build TF-IDF once per doc
    tfidf_vec, tfidf_X = build_tfidf([norm_txt(t) for t in token_texts])

    # Optional: init DistilBERT
    bert_ready = False
    if use_bert:
        try:
            _ensure_reranker(bert_model or "distilbert-base-uncased")
            bert_ready = True
        except Exception as e:
            print(f"[validate_simple] BERT disabled: {e}")

    for it in items:
        key, field, llm_value = it.get("key",""), it.get("field",""), it.get("llm_value","")

        # 1) Autolocate (fuzzy against LLM value)
        best_auto_i, best_auto_s = -1, -1.0
        for i, t in enumerate(token_texts):
            s = fuzzy01(t, llm_value)
            if s > best_auto_s:
                best_auto_s, best_auto_i = s, i
        auto_tok = tokens_all[best_auto_i] if best_auto_i >= 0 else None
        auto_box = auto_tok["bbox"] if auto_tok else None
        auto_page = auto_tok["page"] if auto_tok else None
        auto_text = auto_tok["text"] if auto_tok else ""
        # override with client autolocate if provided
        if client_autolocate and field in client_autolocate:
            cli = client_autolocate[field]
            auto_text = cli.get("text", auto_text)
            auto_box  = cli.get("bbox", auto_box)
            auto_page = cli.get("page", auto_page)
            best_auto_s = fuzzy01(auto_text, llm_value)

        # 2) TF-IDF
        q = ((key or "") + " ") * 7 + ((llm_value or "") + " ") * 3
        qX = qvec(tfidf_vec, norm_txt(q))
        sims = cosine_similarity(qX, tfidf_X).ravel()
        # reuse same boosts as pick_best_token_by_tfidf
        lv = norm_txt(llm_value); lv_len = len(lv)
        boost = np.zeros_like(sims)
        for i, t in enumerate(token_texts):
            tnorm = norm_txt(t)
            if lv and tnorm == lv: boost[i] += 0.15
            if lv_len > 0 and len(tnorm) > lv_len + 3: boost[i] -= 0.05
            k = norm_txt(key)
            if k and k in tnorm: boost[i] += 0.03
        score = sims + boost
        idx3 = int(score.argmax()) if len(score) else -1
        sc3 = float(score[idx3]) if idx3 >= 0 else 0.0
        tok3 = tokens_all[idx3] if idx3 >= 0 else None
        box3 = tok3["bbox"] if tok3 else None
        page3 = tok3["page"] if tok3 else None
        text3 = tok3["text"] if tok3 else ""

        # 3) BERT reranker (optional, semantic spans)
        bert_box = None; bert_page = None; bert_text = ""; bert_score = 0.0
        if bert_ready and llm_value:
            vnorm = norm_txt(llm_value)
            best = None
            # collect candidate spans on each page where value appears in local windows
            for pg, toks in pages.items():
                spans = _find_candidate_spans(vnorm, toks)
                if not spans: continue
                # build local context snippets per span
                cands = []
                for s in spans:
                    ctx = _context_snippet(toks, s["span"])
                    cands.append({"page": pg, "rect": s["rect"], "text": s["text"], "ctx": ctx})
                # embed query + contexts
                qtext = f"{key}: {llm_value}".strip()
                texts = [qtext] + [c["ctx"] for c in cands]
                embs = _embed(texts)
                qv = embs[0:1]; M = embs[1:]
                sims = (M @ qv.T).squeeze(1)  # cosine (normalized)
                # best on this page
                b_idx = int(torch.argmax(sims).item())
                cand = {**cands[b_idx], "score": float(sims[b_idx].item())}
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best:
                bert_box, bert_page, bert_text, bert_score = best["rect"], best["page"], best["text"], float(best["score"])

        rows.append({
            "field": field,
            "key": key,
            "llm_value": llm_value,

            "auto_text": auto_text,
            "auto_conf_fuzzy": round(float(best_auto_s), 4),
            "auto_bbox": auto_box,
            "auto_page": auto_page,
            "auto_charF1": round(char_f1(auto_text, llm_value), 4),
            "auto_ok": int(ok_at_fuzzy(auto_text, llm_value)),

            "tfidf_text": text3,
            "tfidf_conf_cos": round(sc3, 4),
            "tfidf_bbox": box3,
            "tfidf_page": page3,
            "tfidf_charF1": round(char_f1(text3, llm_value), 4),
            "tfidf_ok": int(ok_at_fuzzy(text3, llm_value)),

            "bert_text": bert_text,
            "bert_conf_sem": round(float(bert_score), 4),
            "bert_bbox": bert_box,
            "bert_page": bert_page,
            "bert_charF1": round(char_f1(bert_text, llm_value), 4),
            "bert_ok": int(ok_at_fuzzy(bert_text, llm_value)) if bert_text else 0,
        })

    # summary
    if rows:
        auto_acc = sum(r["auto_ok"]  for r in rows)/len(rows)
        tfidf_acc= sum(r["tfidf_ok"] for r in rows)/len(rows)
        bert_acc = sum(r["bert_ok"]  for r in rows)/len(rows) if any(r["bert_text"] for r in rows) else 0.0
    else:
        auto_acc = tfidf_acc = bert_acc = 0.0

    return rows, {
        "fields": len(rows),
        "auto_acc@0.9": round(auto_acc, 4),
        "tfidf_acc@0.9": round(tfidf_acc, 4),
        "bert_acc@0.9": round(bert_acc, 4)
    }

# ---------- FastAPI ----------
app = FastAPI(title="EDIP Simple Validation (+BERT)")

class ValidateReq(BaseModel):
    ocr_tokens: List[OCRToken]
    items: List[ValidateItem]
    client_autolocate: Dict[str, Dict[str, Any]] | None = None
    use_bert: bool = False
    bert_model: Optional[str] = "distilbert-base-uncased"

class ValidateResp(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

@app.post("/validate_simple", response_model=ValidateResp)
def validate_simple(req: ValidateReq):
    ocr = [t.model_dump() for t in req.ocr_tokens]
    items = [i.model_dump() for i in req.items]
    rows, summary = run_three_locators(
        ocr, items, req.client_autolocate, req.use_bert, req.bert_model
    )
    return ValidateResp(results=rows, summary=summary)

# ---------- CLI batch (unchanged usage) ----------
def run_batch(root: Path, out_csv: Path, use_bert: bool = False, bert_model: Optional[str] = "distilbert-base-uncased"):
    import pandas as pd
    all_rows = []
    for llm in root.rglob("document.json"):
        ocr1 = llm.with_name("ocr_tokens.json")
        ocr2 = llm.with_suffix(".ocr.json")
        ocr = ocr1 if ocr1.exists() else (ocr2 if ocr2.exists() else None)
        if not ocr: 
            continue
        ocr_tokens = json.loads(ocr.read_text(encoding="utf-8"))
        llm_json = json.loads(llm.read_text(encoding="utf-8"))
        items = []
        for f in llm_json.get("fields", []):
            name = f.get("name")
            val  = f.get("value","")
            key  = f.get("key", name) or name
            items.append({"key": key, "field": name, "llm_value": val})
        rows, _ = run_three_locators(ocr_tokens, items, use_bert=use_bert, bert_model=bert_model)
        for r in rows:
            r["doc_path"] = str(llm.parent)
        all_rows.extend(rows)
    if not all_rows:
        print("No rows found."); return
    import pandas as pd
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(all_rows)} rows.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--batch", help="root folder with documents")
    ap.add_argument("--out", help="CSV path", default="simple_validation.csv")
    ap.add_argument("--use-bert", action="store_true")
    ap.add_argument("--bert-model", default="distilbert-base-uncased")
    args = ap.parse_args()

    if args.batch:
        run_batch(Path(args.batch), Path(args.out), use_bert=args.use_bert, bert_model=args.bert_model)
    elif args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        ap.print_help()