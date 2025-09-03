# src/routers/ocr.py
from uuid import uuid4
import os, json
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
from rapidfuzz import fuzz

router = APIRouter(prefix="/ocr", tags=["ocr"])

# storage roots (reuse your app’s ENV if set)
DATA_DIR = os.environ.get("LEXI_DATA", "data")
OUT_DIR  = os.environ.get("LEXI_OUT",  "out")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

def _save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _tesseract_tokens(pdf_path: str):
    """Return word tokens and merged line segments with bboxes (page coords)."""
    pages = convert_from_path(pdf_path, dpi=300)
    tokens, segments = [], []
    for pageno, img in enumerate(pages, start=1):
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n = len(data["text"])
        lines = {}
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt: 
                continue
            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = 0.0
            x,y,w,h = map(float, (data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
            tok = {"text": txt, "conf": conf/100.0,
                   "bbox": {"page": pageno, "x0": x, "y0": y, "x1": x+w, "y1": y+h}}
            tokens.append(tok)
            lines.setdefault(int(y/6), []).append(tok)

        for _, line in sorted(lines.items()):
            xs0=[t["bbox"]["x0"] for t in line]; ys0=[t["bbox"]["y0"] for t in line]
            xs1=[t["bbox"]["x1"] for t in line]; ys1=[t["bbox"]["y1"] for t in line]
            segments.append({
                "page": pageno,
                "text": " ".join(t["text"] for t in line),
                "bbox": {"page": pageno, "x0": min(xs0), "y0": min(ys0), "x1": max(xs1), "y1": max(ys1)}
            })
    return tokens, segments

@router.get("/healthz")
def healthz(): 
    return {"ok": True}

@router.post("/upload")
async def upload(pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    """Save PDF, run OCR (tesseract), persist tokens/segments under OUT_DIR/<doc_id>."""
    doc_id = uuid4().hex[:12]
    pdf_path = os.path.join(DATA_DIR, f"{doc_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # swap this if you later call ECM DocAI
    tokens, segments = _tesseract_tokens(pdf_path)

    out_dir = os.path.join(OUT_DIR, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    _save_json(os.path.join(out_dir, "tokens.json"),   {"tokens": tokens})
    _save_json(os.path.join(out_dir, "segments.json"), {"segments": segments})

    return {
        "ok": True,
        "doc_id": doc_id,
        "pdf_url": f"/data/{doc_id}.pdf",             # if you mount /data as static
        "annotated_tokens_url": f"/data/{doc_id}.pdf",
        "out_dir": f"/out/{doc_id}",
        "stats": {"tokens": len(tokens), "segments": len(segments)}
    }

class SearchReq(BaseModel):
    doc_id: str
    query: str
    topk: int = 5

@router.post("/search")
def search(req: SearchReq):
    p = os.path.join(OUT_DIR, req.doc_id, "segments.json")
    if not os.path.exists(p):
        raise HTTPException(404, "doc not found")
    segs = json.load(open(p, encoding="utf-8"))["segments"]

    q = req.query.lower().strip()
    scored = []
    for s in segs:
        sc = fuzz.token_set_ratio(q, s["text"].lower())
        if sc >= 60:
            scored.append({"score": int(sc), "segment": s})

    scored.sort(key=lambda x: -x["score"])
    out, seen = [], set()
    for it in scored:
        s = it["segment"]; b = s["bbox"]; key = (s["page"], int(b["y0"]/10))
        if key in seen: 
            continue
        seen.add(key)
        out.append({"score": it["score"], "page": s["page"], "bbox": b, "text": s["text"]})
        if len(out) >= req.topk: 
            break
    return {"matches": out}

class LassoReq(BaseModel):
    doc_id: str
    page: int
    x0: float; y0: float; x1: float; y1: float
    join_lines: bool = True
    y_tol: float = 8.0

@router.post("/lasso")
def lasso(req: LassoReq):
    p = os.path.join(OUT_DIR, req.doc_id, "tokens.json")
    if not os.path.exists(p):
        raise HTTPException(404, "doc not found")
    toks = json.load(open(p, encoding="utf-8"))["tokens"]

    x0,y0 = min(req.x0,req.x1), min(req.y0,req.y1)
    x1,y1 = max(req.x0,req.x1), max(req.y0,req.y1)

    inside = [t for t in toks if t["bbox"]["page"]==req.page and not(
        t["bbox"]["x1"]<x0 or t["bbox"]["x0"]>x1 or t["bbox"]["y1"]<y0 or t["bbox"]["y0"]>y1)]
    inside.sort(key=lambda t:(t["bbox"]["y0"], t["bbox"]["x0"]))

    lines, cur = [], []
    for t in inside:
        if not cur:
            cur=[t]; continue
        if abs(t["bbox"]["y0"]-cur[-1]["bbox"]["y0"])<=req.y_tol:
            cur.append(t)
        else:
            lines.append(cur); cur=[t]
    if cur: lines.append(cur)

    def union(ln):
        xs0=[w["bbox"]["x0"] for w in ln]; ys0=[w["bbox"]["y0"] for w in ln]
        xs1=[w["bbox"]["x1"] for w in ln]; ys1=[w["bbox"]["y1"] for w in ln]
        return {"page": ln[0]["bbox"]["page"], "x0":min(xs0),"y0":min(ys0),"x1":max(xs1),"y1":max(ys1)}

    text = ("\n".join(" ".join(w["text"] for w in ln) for ln in lines)
            if req.join_lines else " ".join(w["text"] for w in inside))
    return {"text": text, "lines":[{"text":" ".join(w["text"] for w in ln), "bbox": union(ln)} for ln in lines]}
