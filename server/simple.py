# server_pairs_ocr_validate.py
# One-file server & batch tool: pairs PDFs with JSON by stem, runs vertical-aware OCR,
# and validates fields via Autolocate (fuzzy), TF-IDF, and optional DistilBERT.
#
# Quick start (Windows-friendly):
#   python -m venv .venv
#   . .\.venv\Scripts\Activate.ps1
#   pip install fastapi uvicorn pydantic pytesseract pdf2image Pillow rapidfuzz scikit-learn pandas
#   # optional for BERT:
#   pip install torch transformers
#
# API:
#   python server_pairs_ocr_validate.py --serve --host 0.0.0.0 --port 8080
#   POST /validate_simple     (inline OCR tokens + items)
#   POST /validate_pair       (pdf_path + json_path; OCR generated if needed)
#
# Batch:
#   python server_pairs_ocr_validate.py --batch C:\data\root --out .\out.csv --dpi 300 --use-bert
#
# NOTE: On Windows, install Poppler and pass --poppler "C:\\path\\to\\poppler\\bin" if pdf2image can't find it.
from __future__ import annotations

import os, math, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator, Set
from dataclasses import dataclass

# ----------- Web/API -----------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ----------- OCR deps -----------
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

# ----------- Validation deps -----------
from rapidfuzz import fuzz
import numpy as np as _np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------- Optional BERT -----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None


# ===========================
#           OCR
# ===========================

@dataclass
class _OCRTok:
    text: str
    bbox: List[float]  # [x0,y0,x1,y1] page pixel coords (top-left origin)
    page: int
    angle: int
    conf: float

def _rotate_image(img: Image.Image, angle: int) -> Image.Image:
    if angle % 360 == 0:
        return img
    return img.rotate(angle, expand=False, resample=Image.BICUBIC)

def _rotate_box_back(b: Tuple[float,float,float,float], angle: int, W: int, H: int) -> Tuple[float,float,float,float]:
    x0, y0, x1, y1 = b
    if angle % 360 == 0:
        return x0, y0, x1, y1
    pts = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
    if angle == 90:
        inv = np.stack([pts[:,1], W - pts[:,0]], axis=1)
    elif angle == 180:
        inv = np.stack([W - pts[:,0], H - pts[:,1]], axis=1)
    elif angle == 270:
        inv = np.stack([H - pts[:,1], pts[:,0]], axis=1)
    else:
        # generic inverse around center
        rad = -math.radians(angle)
        cx, cy = W/2.0, H/2.0
        c, s = math.cos(rad), math.sin(rad)
        inv = np.empty_like(pts)
        for i, (x,y) in enumerate(pts):
            dx, dy = x - cx, y - cy
            rx =  c*dx - s*dy + cx
            ry =  s*dx + c*dy + cy
            inv[i] = [rx, ry]
    nx0, ny0 = inv.min(0)
    nx1, ny1 = inv.max(0)
    nx0, ny0 = max(0, nx0), max(0, ny0)
    nx1, ny1 = min(W, nx1), min(H, ny1)
    return float(nx0), float(ny0), float(nx1), float(ny1)

def _image_to_tokens(img: Image.Image, angle: int, page: int, psm: int = 6) -> List[_OCRTok]:
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    W, H = img.size
    toks: List[_OCRTok] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        rx0, ry0, rx1, ry1 = float(x), float(y), float(x+w), float(y+h)
        bx0, by0, bx1, by1 = _rotate_box_back((rx0, ry0, rx1, ry1), angle, W, H)
        toks.append(_OCRTok(text=txt, bbox=[bx0,by0,bx1,by1], page=page, angle=angle, conf=conf))
    return toks

def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1,bx1) - max(ax0,bx0))
    inter_h = max(0.0, min(ay1,by1) - max(ay0,by0))
    inter = inter_w * inter_h
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax1-ax0)) * max(0.0, (ay1-ay0))
    area_b = max(0.0, (bx1-bx0)) * max(0.0, (by1-by0))
    denom = area_a + area_b - inter + 1e-6
    return inter / denom

def _dedup_tokens(tokens: List[_OCRTok], iou_thr: float = 0.6) -> List[_OCRTok]:
    if not tokens: return []
    boxes = np.array([t.bbox for t in tokens], dtype=np.float32)
    confs = np.array([t.conf for t in tokens], dtype=np.float32)
    order = sorted(range(len(tokens)),
                   key=lambda i: (confs[i] + (0.1 if tokens[i].angle==0 else 0.0)),
                   reverse=True)
    keep, suppressed = [], set()
    for i in order:
        if i in suppressed: continue
        keep.append(i)
        for j in order:
            if j <= i or j in suppressed: continue
            if _iou(tuple(boxes[i]), tuple(boxes[j])) >= iou_thr:
                suppressed.add(j)
    return [tokens[i] for i in keep]

def pdf_to_tokens(pdf_path: str | Path, dpi: int = 300, psm: int = 6,
                  poppler_path: Optional[str] = None) -> List[Dict[str, Any]]:
    pages = convert_from_path(str(pdf_path), dpi=dpi, poppler_path=poppler_path)
    all_toks: List[_OCRTok] = []
    for pnum, img in enumerate(pages, start=1):
        for angle in (0, 90, 180, 270):
            rimg = _rotate_image(img, angle)
            all_toks.extend(_image_to_tokens(rimg, angle=angle, page=pnum, psm=psm))
    dedup = _dedup_tokens(all_toks, iou_thr=0.6)
    return [{
        "text": t.text, "bbox": [float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])],
        "page": int(t.page), "angle": int(t.angle), "conf": float(t.conf)
    } for t in dedup]


# ===========================
#      Pairing by stem
# ===========================

PDF_EXTS = {".pdf"}
JSON_EXTS = {".json"}

def find_pdf_json_pairs(root: Path,
                        exclude_dirs: Set[str] = {"node_modules", ".git", "__pycache__"}) -> Iterator[Tuple[Path, Path]]:
    """
    Yield (pdf_path, json_path) where filenames share the same stem (case-insensitive) in the same folder.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs and not d.startswith((".", "_"))]
        here = Path(dirpath)
        by_stem: Dict[str, Dict[str, Path]] = {}
        for name in filenames:
            p = here / name
            ext = p.suffix.lower()
            stem = p.stem.lower()
            if ext in PDF_EXTS:
                by_stem.setdefault(stem, {})["pdf"] = p
            elif ext in JSON_EXTS:
                by_stem.setdefault(stem, {})["json"] = p
        for _, d in by_stem.items():
            if "pdf" in d and "json" in d:
                yield d["pdf"], d["json"]

def ensure_ocr_for(pdf_path: Path, dpi: int = 300, poppler_path: Optional[str] = None) -> Path:
    """
    Preferred OCR tokens file: <stem>.ocr.json next to the PDF.
    """
    ocr_json = pdf_path.with_suffix(".ocr.json")
    if ocr_json.exists():
        return ocr_json
    legacy = pdf_path.parent / "ocr_tokens.json"
    if legacy.exists():
        return legacy
    print(f"[ocr] {pdf_path}")
    tokens = pdf_to_tokens(pdf_path, dpi=dpi, poppler_path=poppler_path)
    ocr_json.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return ocr_json


# ===========================
#        Validation
# ===========================

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

# ---- Optional BERT (lazy singleton) ----
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

def _group_by_page(tokens: List[Dict[str,Any]]):
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
    # dedup by coarse rect grid
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


# ===========================
#       Core validation
# ===========================

def run_locators(ocr_tokens: List[Dict[str, Any]],
                 items: List[Dict[str, Any]],
                 use_bert: bool = False,
                 bert_model: Optional[str] = "distilbert-base-uncased"):
    rows = []
    token_texts = [t["text"] for t in ocr_tokens]
    # TF-IDF once per doc
    vec, X = build_tfidf([norm_txt(t) for t in token_texts])

    bert_ready = False
    if use_bert:
        try:
            _ensure_reranker(bert_model or "distilbert-base-uncased")
            bert_ready = True
        except Exception as e:
            print(f"[validate] BERT disabled: {e}")

    for it in items:
        key, field, llm_value = it.get("key",""), it.get("field",""), it.get("llm_value","")

        # 1) Autolocate (fuzzy vs llm_value)
        auto_idx, auto_s = -1, -1.0
        for i, t in enumerate(token_texts):
            s = fuzzy01(t, llm_value)
            if s > auto_s:
                auto_s, auto_idx = s, i
        auto_tok = ocr_tokens[auto_idx] if auto_idx >= 0 else None
        auto_text = auto_tok["text"] if auto_tok else ""
        auto_box  = auto_tok["bbox"] if auto_tok else None
        auto_page = auto_tok["page"] if auto_tok else None

        # 2) TF-IDF (key+value)
        q = ((key or "") + " ") * 7 + ((llm_value or "") + " ") * 3
        qX = qvec(vec, norm_txt(q))
        sims = cosine_similarity(qX, X).ravel()
        lv = norm_txt(llm_value); lv_len = len(lv)
        boost = _np.zeros_like(sims)
        for i, t in enumerate(token_texts):
            tnorm = norm_txt(t)
            if lv and tnorm == lv: boost[i] += 0.15
            if lv_len>0 and len(tnorm) > lv_len+3: boost[i] -= 0.05
            k = norm_txt(key)
            if k and k in tnorm: boost[i] += 0.03
        score = sims + boost
        tidx = int(score.argmax()) if len(score) else -1
        tsc  = float(score[tidx]) if tidx >= 0 else 0.0
        ttok = ocr_tokens[tidx] if tidx >= 0 else None
        t_text = ttok["text"] if ttok else ""
        t_box  = ttok["bbox"] if ttok else None
        t_page = ttok["page"] if ttok else None

        # 3) BERT reranker (semantic spans around value)
        b_text=""; b_box=None; b_page=None; b_sc=0.0
        if bert_ready and llm_value:
            pages = _group_by_page(ocr_tokens)
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

    n = max(1, len(rows))
    summary = {
        "fields": len(rows),
        "auto_acc@0.9": round(sum(r["auto_ok"]  for r in rows)/n, 4),
        "tfidf_acc@0.9": round(sum(r["tfidf_ok"] for r in rows)/n, 4),
        "bert_acc@0.9":  round(sum(r["bert_ok"]  for r in rows)/n, 4) if any(r["bert_text"] for r in rows) else 0.0
    }
    return rows, summary


# ===========================
#            API
# ===========================

app = FastAPI(title="EDIP Pairs OCR + Validation")

class OCRToken(BaseModel):
    text: str
    bbox: List[float]  # [x0,y0,x1,y1] in page pixels (top-left)
    page: int = 1

class ValidateItem(BaseModel):
    key: str
    field: str
    llm_value: str

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

class PathValidateReq(BaseModel):
    pdf_path: str
    json_path: str
    dpi: int = 300
    use_bert: bool = False
    bert_model: Optional[str] = "distilbert-base-uncased"
    poppler_path: Optional[str] = None

@app.post("/validate_pair", response_model=ValidateResp)
def validate_pair(req: PathValidateReq):
    pdf = Path(req.pdf_path); js = Path(req.json_path)
    if not pdf.exists() or not js.exists():
        raise HTTPException(404, "pdf or json not found")
    ocr_path = ensure_ocr_for(pdf, dpi=req.dpi, poppler_path=req.poppler_path)
    ocr_tokens = json.loads(ocr_path.read_text(encoding="utf-8"))
    llm_json = json.loads(js.read_text(encoding="utf-8"))
    items = _collect_items(llm_json)
    rows, summary = run_locators(ocr_tokens, items, use_bert=req.use_bert, bert_model=req.bert_model)
    return ValidateResp(results=rows, summary=summary)


# ===========================
#       Batch runner
# ===========================

def _collect_items(llm_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expecting {"fields":[{"name":..., "value":..., "key"?:...}, ...]}.
    Falls back to 'name' as key if 'key' missing.
    """
    items = []
    for f in llm_json.get("fields", []):
        name = f.get("name")
        if not name:
            continue
        val  = f.get("value","")
        key  = f.get("key", name) or name
        items.append({"key": key, "field": name, "llm_value": val})
    return items

def run_batch(root: Path, out_csv: Path, dpi: int = 300,
              use_bert: bool = False, bert_model: Optional[str] = "distilbert-base-uncased",
              poppler_path: Optional[str] = None):
    import pandas as pd
    all_rows: List[Dict[str, Any]] = []

    for pdf_path, json_path in find_pdf_json_pairs(root):
        try:
            ocr_path = ensure_ocr_for(pdf_path, dpi=dpi, poppler_path=poppler_path)
            ocr_tokens = json.loads(ocr_path.read_text(encoding="utf-8"))
            llm_json = json.loads(json_path.read_text(encoding="utf-8"))
            items = _collect_items(llm_json)
            rows, _ = run_locators(ocr_tokens, items, use_bert=use_bert, bert_model=bert_model)
            for r in rows:
                r["pdf"] = str(pdf_path)
                r["json"] = str(json_path)
                r["ocr_tokens"] = str(ocr_path)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[warn] Skipping pair ({pdf_path.name}, {json_path.name}): {e}")

    if not all_rows:
        print("No pdf+json pairs found.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (rows={len(all_rows)})")


# ===========================
#            Main
# ===========================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="start API server")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)

    ap.add_argument("--batch", help="root folder to scan for pdf/json pairs")
    ap.add_argument("--out", default="validation_pairs.csv")
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