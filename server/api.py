import os, json
from uuid import uuid4
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output
from rapidfuzz import fuzz

# -------- Paths --------
DATA = os.environ.get("LEXI_DATA", "data")
OUT  = os.environ.get("LEXI_OUT",  "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT,  exist_ok=True)

app = FastAPI(title="Lexi OCR Service")

# (Optional) CORS for direct calls from your React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve input PDFs and OCR artifacts
app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/out",  StaticFiles(directory=OUT),  name="out")

# ---------- Utils ----------
def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def tesseract_tokens(pdf_path: str):
    """Return word tokens and simple line segments with bboxes (page space)."""
    pages = convert_from_path(pdf_path, dpi=300)
    tokens, segments = [], []
    for pageno, img in enumerate(pages, start=1):
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n = len(data["text"])
        # quick line bucketing by y
        lines = {}
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not txt:
                continue
            conf = data["conf"][i]
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            x, y, w, h = map(float, (data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
            tok = {
                "text": txt,
                "conf": conf / 100.0,
                "bbox": {"page": pageno, "x0": x, "y0": y, "x1": x + w, "y1": y + h},
            }
            tokens.append(tok)
            key = int(y / 6)
            lines.setdefault(key, []).append(tok)

        # merge tokens per line to segments
        for _, line in sorted(lines.items()):
            xs0 = [t["bbox"]["x0"] for t in line]
            ys0 = [t["bbox"]["y0"] for t in line]
            xs1 = [t["bbox"]["x1"] for t in line]
            ys1 = [t["bbox"]["y1"] for t in line]
            segments.append({
                "page": pageno,
                "text": " ".join(t["text"] for t in line),
                "bbox": {"page": pageno, "x0": min(xs0), "y0": min(ys0), "x1": max(xs1), "y1": max(ys1)}
            })
    return tokens, segments

# ---------- Models ----------
class SearchReq(BaseModel):
    doc_id: str
    query: str
    topk: int = 5

class LassoReq(BaseModel):
    doc_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    join_lines: bool = True
    y_tol: float = 8.0  # px tolerance to group tokens into a line

class AuditEvent(BaseModel):
    event: str
    payload: dict

# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/upload")
async def upload(pdf: UploadFile = File(...), backend: str = Form("tesseract")):
    """Save PDF, run OCR (tesseract), write tokens/segments to /out/<doc_id>."""
    doc_id = uuid4().hex[:12]
    pdf_path = os.path.join(DATA, f"{doc_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await pdf.read())

    # OCR (tesseract default)
    tokens, segments = tesseract_tokens(pdf_path)

    out_dir = os.path.join(OUT, doc_id)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "tokens.json"),   {"tokens": tokens})
    save_json(os.path.join(out_dir, "segments.json"), {"segments": segments})

    return {
        "ok": True,
        "doc_id": doc_id,
        "pdf_url": f"/data/{doc_id}.pdf",
        # for the viewer we can just render the original; you can swap for an annotated PDF later
        "annotated_tokens_url": f"/data/{doc_id}.pdf",
        "out_dir": f"/out/{doc_id}",
        "stats": {"tokens": len(tokens), "segments": len(segments)}
    }

@app.get("/doc/{doc_id}/tokens")
def get_tokens(doc_id: str):
    with open(os.path.join(OUT, doc_id, "tokens.json"), "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/doc/{doc_id}/segments")
def get_segments(doc_id: str):
    with open(os.path.join(OUT, doc_id, "segments.json"), "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/search")
def search(req: SearchReq):
    """Fuzzy search over line segments; return top-K matches with bboxes."""
    with open(os.path.join(OUT, req.doc_id, "segments.json"), "r", encoding="utf-8") as f:
        segs = json.load(f)["segments"]

    q = req.query.lower().strip()
    scored = []
    for s in segs:
        sc = fuzz.token_set_ratio(q, s["text"].lower())
        if sc >= 60:
            scored.append({"score": int(sc), "segment": s})

    scored.sort(key=lambda x: -x["score"])
    out = []
    seen = set()
    for it in scored:
        s = it["segment"]; b = s["bbox"]; key = (s["page"], int(b["y0"] / 10))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "score": it["score"],
            "page": s["page"],
            "bbox": b,
            "text": s["text"]
        })
        if len(out) >= req.topk:
            break
    return {"matches": out}

@app.post("/lasso")
def lasso(req: LassoReq):
    """Return text and line boxes inside a rectangle (page coords)."""
    with open(os.path.join(OUT, req.doc_id, "tokens.json"), "r", encoding="utf-8") as f:
        toks = json.load(f)["tokens"]

    x0, y0 = min(req.x0, req.x1), min(req.y0, req.y1)
    x1, y1 = max(req.x0, req.x1), max(req.y0, req.y1)

    # tokens intersecting rect
    inside = [
        t for t in toks if t["bbox"]["page"] == req.page and not (
            t["bbox"]["x1"] < x0 or t["bbox"]["x0"] > x1 or
            t["bbox"]["y1"] < y0 or t["bbox"]["y0"] > y1
        )
    ]
    inside.sort(key=lambda t: (t["bbox"]["y0"], t["bbox"]["x0"]))

    # group into lines
    lines, cur = [], []
    for t in inside:
        if not cur:
            cur = [t]; continue
        if abs(t["bbox"]["y0"] - cur[-1]["bbox"]["y0"]) <= req.y_tol:
            cur.append(t)
        else:
            lines.append(cur); cur = [t]
    if cur:
        lines.append(cur)

    def union_box(ln):
        xs0 = [w["bbox"]["x0"] for w in ln]; ys0 = [w["bbox"]["y0"] for w in ln]
        xs1 = [w["bbox"]["x1"] for w in ln]; ys1 = [w["bbox"]["y1"] for w in ln]
        return {"page": ln[0]["bbox"]["page"], "x0": min(xs0), "y0": min(ys0), "x1": max(xs1), "y1": max(ys1)}

    text = (
        "\n".join(" ".join(w["text"] for w in ln) for ln in lines)
        if req.join_lines else
        " ".join(w["text"] for w in inside)
    )
    return {"text": text, "lines": [{"text": " ".join(w["text"] for w in ln), "bbox": union_box(ln)} for ln in lines]}

@app.post("/audit")
def audit(evt: AuditEvent):
    """Append audit events to out/audit.jsonl (you can replace with DB/S3)."""
    path = os.path.join(OUT, "audit.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": os.environ.get("TS",""), **evt.dict()}) + "\n")
    return {"ok": True}
