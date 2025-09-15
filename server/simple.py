# server_simple.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from rapidfuzz import fuzz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- OPTIONAL BERT
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

# ---- Our OCR
from ocr import pdf_to_tokens  # same folder import

# ===================== Basic text utils =====================

def norm_txt(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def fuzzy01(a: str, b: str) -> float:
    if not a or not b: return 0.0
    return fuzz.token_set_ratio(norm_txt(a), norm_txt(b)) / 100.0

# ===================== TF-IDF =====================

def build_tfidf(corpus: List[str]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

def qvec(vec, q: str):
    return vec.transform([q])

# ===================== Optional BERT reranker =====================

_RERANK = {"tok": None, "mdl": None, "device": "cpu"}

def _ensure_reranker(model_name_or_path: str = "distilbert-base-uncased"):
    if AutoTokenizer is None or AutoModel is None:
        raise RuntimeError("Transformers not available.")
    if _RERANK["mdl"] is not None:
        return
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    mdl = AutoModel.from_pretrained(model_name_or_path)
    mdl.to(dev).eval()
    _RERANK.update({"tok": tok, "mdl": mdl, "device": dev})
    print(f"[validate] DistilBERT on {dev}")

def _embed(texts: List[str]):
    tok, mdl, dev = _RERANK["tok"], _RERANK["mdl"], _RERANK["device"]
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        out = mdl(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(1); counts = mask.sum(1).clamp(min=1)
        emb = summed / counts
        return torch.nn.functional.normalize(emb, p=2, dim=1)

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
                if w >= 3: break
    # simple rect dedup
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

# ===================== Metrics =====================

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

# ===================== API models =====================

class OCRToken(BaseModel):
    text: str
    bbox: List[float]  # [x0,y0,x1,y1] page pixel coords (top-left origin)
    page: int = 1

class ValidateItem(BaseModel):
    key: str
    field: str
    llm_value: str

class ValidateRequest(BaseModel):
    ocr_tokens: List[OCRToken]
    items: List[ValidateItem]
    use_bert: bool = False
    bert_model: Optional[str] = "distilbert-base-uncased"

class ValidateResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

# ===================== Core locator (Auto + TF-IDF + optional BERT) =====================

def run_locators(ocr_tokens: List[Dict[str, Any]],
                 items: List[Dict[str, Any]],
                 use_bert: bool = False,
                 bert_model: Optional[str] = "distilbert-base-uncased"):
    rows = []
    tokens_all = list(ocr_tokens)
    token_texts = [t["text"] for t in tokens_all]

    # TF-IDF index once per doc
    vec, X = build_tfidf([norm_txt(t) for t in token_texts])

    # Optional BERT
    bert_ready = False
    if use_bert:
        try:
            _ensure_reranker(bert_model or "distilbert-base-uncased")
            bert_ready = True
        except Exception as e:
            print(f"[validate] BERT disabled: {e}")

    for it in items:
        key, field, llm_value = it.get("key",""), it.get("field",""), it.get("llm_value","")

        # 1) Autolocate = best fuzzy vs llm_value
        auto_idx, auto_s = -1, -1.0
        for i, t in enumerate(token_texts):
            s = fuzzy01(t, llm_value)
            if s > auto_s:
                auto_s, auto_idx = s, i
        auto_tok = tokens_all[auto_idx] if auto_idx >= 0 else None
        auto_text = auto_tok["text"] if auto_tok else ""
        auto_box  = auto_tok["bbox"] if auto_tok else None
        auto_page = auto_tok["page"] if auto_tok else None

        # 2) TF-IDF = key(+value) query vs tokens
        q = ((key or "") + " ") * 7 + ((llm_value or "") + " ") * 3
        qX = qvec(vec, norm_txt(q))
        sims = cosine_similarity(qX, X).ravel()
        # boosts
        lv = norm_txt(llm_value); lv_len = len(lv)
        boost = np.zeros_like(sims)
        for i, t in enumerate(token_texts):
            tnorm = norm_txt(t)
            if lv and tnorm == lv: boost[i] += 0.15
            if lv_len>0 and len(tnorm) > lv_len+3: boost[i] -= 0.05
            k = norm_txt(key)
            if k and k in tnorm: boost[i] += 0.03
        score = sims + boost
        tidx = int(score.argmax()) if len(score) else -1
        tsc = float(score[tidx]) if tidx >= 0 else 0.0
        ttok = tokens_all[tidx] if tidx >= 0 else None
        t_text = ttok["text"] if ttok else ""
        t_box  = ttok["bbox"] if ttok else None
        t_page = ttok["page"] if ttok else None

        # 3) BERT reranker (semantic span around value)
        b_text=""; b_box=None; b_page=None; b_sc=0.0
        if bert_ready and llm_value:
            pages = _group_by_page(tokens_all)
            vnorm = norm_txt(llm_value)
            best = None
            for pg, toks in pages.items():
                spans = _find_candidate_spans(vnorm, toks)
                if not spans: continue
                cands = [{"page": pg, "rect": s["rect"], "text": s["text"],
                          "ctx": _context_snippet(toks, s["span"])} for s in spans]
                query = f"{key}: {llm_value}".strip()
                texts = [query] + [c["ctx"] for c in cands]
                embs = _embed(texts)
                qv, M = embs[0:1], embs[1:]
                sims = (M @ qv.T).squeeze(1)
                bi = int(torch.argmax(sims).item())
                cand = {**cands[bi], "score": float(sims[bi].item())}
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best:
                b_text, b_box, b_page, b_sc = best["text"], best["rect"], best["page"], float(best["score"])

        rows.append({
            "field": field, "key": key, "llm_value": llm_value,

            "auto_text": auto_text, "auto_conf_fuzzy": round(float(auto_s),4),
            "auto_bbox": auto_box,  "auto_page": auto_page,
            "auto_charF1": round(char_f1(auto_text, llm_value),4),
            "auto_ok": int(ok_at_fuzzy(auto_text, llm_value)),

            "tfidf_text": t_text,   "tfidf_conf_cos": round(tsc,4),
            "tfidf_bbox": t_box,    "tfidf_page": t_page,
            "tfidf_charF1": round(char_f1(t_text, llm_value),4),
            "tfidf_ok": int(ok_at_fuzzy(t_text, llm_value)),

            "bert_text": b_text,    "bert_conf_sem": round(float(b_sc),4),
            "bert_bbox": b_box,     "bert_page": b_page,
            "bert_charF1": round(char_f1(b_text, llm_value),4) if b_text else 0.0,
            "bert_ok": int(ok_at_fuzzy(b_text, llm_value)) if b_text else 0
        })

    # summary
    n = len(rows) or 1
    summary = {
        "fields": len(rows),
        "auto_acc@0.9": round(sum(r["auto_ok"]  for r in rows)/n, 4),
        "tfidf_acc@0.9": round(sum(r["tfidf_ok"] for r in rows)/n, 4),
        "bert_acc@0.9":  round(sum(r["bert_ok"]  for r in rows)/n, 4) if any(r["bert_text"] for r in rows) else 0.0
    }
    return rows, summary

# ===================== API =====================

app = FastAPI(title="EDIP OCR + Validation")

class ValidateReq(BaseModel):
    ocr_tokens: List[OCRToken]
    items: List[ValidateItem]
    use_bert: bool = False
    bert_model: Optional[str] = "distilbert-base-uncased"

class ValidateResp(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

@app.post("/validate_simple", response_model=ValidateResp)
def validate_simple(req: ValidateReq):
    ocr = [t.model_dump() for t in req.ocr_tokens]
    items = [i.model_dump() for i in req.items]
    rows, summary = run_locators(ocr, items, use_bert=req.use_bert, bert_model=req.bert_model)
    return ValidateResp(results=rows, summary=summary)

# ===================== Folder runner =====================

def _maybe_ocr_one(folder: Path, dpi: int = 300, poppler_path: Optional[str] = None) -> Optional[Path]:
    pdf = folder / "document.pdf"
    if not pdf.exists():
        return None
    ocr_json = folder / "ocr_tokens.json"
    if ocr_json.exists():
        return ocr_json
    print(f"[ocr] {pdf}")
    tokens = pdf_to_tokens(str(pdf), dpi=dpi, poppler_path=poppler_path)
    ocr_json.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return ocr_json

def _collect_items(llm_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for f in llm_json.get("fields", []):
        name = f.get("name")
        val  = f.get("value","")
        key  = f.get("key", name) or name
        if name:
            items.append({"key": key, "field": name, "llm_value": val})
    return items

def run_batch(root: Path, out_csv: Path, dpi: int = 300,
              use_bert: bool = False, bert_model: Optional[str] = "distilbert-base-uncased",
              poppler_path: Optional[str] = None):
    import pandas as pd
    all_rows = []
    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue
        llm = folder / "document.json"
        if not llm.exists():
            continue
        ocr_path = _maybe_ocr_one(folder, dpi=dpi, poppler_path=poppler_path)
        if not ocr_path:
            continue
        ocr_tokens = json.loads(ocr_path.read_text(encoding="utf-8"))
        llm_json = json.loads(llm.read_text(encoding="utf-8"))
        items = _collect_items(llm_json)
        rows, _ = run_locators(ocr_tokens, items, use_bert=use_bert, bert_model=bert_model)
        for r in rows:
            r["doc_path"] = str(folder)
        all_rows.extend(rows)
    if not all_rows:
        print("No documents found.")
        return
    import pandas as pd
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (rows={len(all_rows)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="start API")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--batch", help="root folder to scan recursively")
    ap.add_argument("--out", default="validation_simple.csv")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--use-bert", action="store_true")
    ap.add_argument("--bert-model", default="distilbert-base-uncased")
    ap.add_argument("--poppler", default=None, help="poppler bin path (Windows)")
    args = ap.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.batch:
        run_batch(Path(args.batch), Path(args.out), dpi=args.dpi,
                  use_bert=args.use_bert, bert_model=args.bert_model,
                  poppler_path=args.poppler)
    else:
        ap.print_help()