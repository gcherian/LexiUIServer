from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

router = APIRouter(prefix="/lasso", tags=["lasso-core"])

# ---------- storage layout ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adjust if needed
DATA = (PROJECT_ROOT / "data")
DATA.mkdir(parents=True, exist_ok=True)

def doc_dir(doc_id: str) -> Path: return (DATA / doc_id)
def pdf_path(doc_id: str) -> Path: return doc_dir(doc_id) / "original.pdf"
def meta_path(doc_id: str) -> Path: return doc_dir(doc_id) / "meta.json"
def boxes_path(doc_id: str) -> Path: return doc_dir(doc_id) / "boxes.json"

# ---------- models ----------
class Rect(BaseModel):
    x0: float; y0: float; x1: float; y1: float

class Box(Rect):
    page: int
    text: Optional[str] = None

class MetaResp(BaseModel):
    pages: List[Dict[str, float]]

# ---------- endpoints ----------
@router.get("/health")
def health():  # quick sanity
    return {"ok": True}

@router.get("/doc/{doc_id}/meta", response_model=MetaResp)
def get_meta(doc_id: str):
    mp = meta_path(doc_id)
    if not mp.exists():
        raise HTTPException(404, "meta.json missing")
    j = json.loads(mp.read_text())
    return MetaResp(pages=j.get("pages", []))

@router.get("/doc/{doc_id}/boxes", response_model=List[Box])
def get_boxes(doc_id: str):
    bp = boxes_path(doc_id)
    if not bp.exists():
        return []
    raw = json.loads(bp.read_text())
    out: List[Box] = []
    for t in raw:
        out.append(Box(
            page=int(t["page"]),
            x0=float(t["x0"]), y0=float(t["y0"]), x1=float(t["x1"]), y1=float(t["y1"]),
            text=(t.get("text") or None),
        ))
    return out

# Optional OCR preview (keep shape-compatible; stub returns empty quickly)
class LassoReq(Rect):
    doc_id: str; page: int

@router.post("/lasso")
def lasso_preview(req: LassoReq):
    # Keep fast and deterministic for demos: just echo and empty text.
    # If you have Tesseract wired, you can swap in your crop+OCR here.
    return {
        "text": "",
        "rect_used": {"page": req.page, "x0": req.x0, "y0": req.y0, "x1": req.x1, "y1": req.y1},
        "page_size": _page_size(req.doc_id, req.page),
        "crop_url": None,
    }

def _page_size(doc_id: str, page: int):
    try:
        j = json.loads(meta_path(doc_id).read_text())
        for p in j.get("pages", []):
            if int(p.get("page")) == int(page):
                return {"width": float(p.get("width")), "height": float(p.get("height"))}
    except Exception:
        pass
    return None
