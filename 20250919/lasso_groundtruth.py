from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path
import json, math

from .ocr_unified import doc_dir

router = APIRouter(prefix="/lasso", tags=["lasso-gt"])

GT_FILE = "groundtruth.json"

def _gt_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / GT_FILE

class GTRect(BaseModel):
    page: int
    x0: float; y0: float; x1: float; y1: float

class GTRecord(BaseModel):
    key: str
    value: Optional[str] = None
    rect: Optional[GTRect] = None
    updated_by: Optional[str] = None
    updated_at: Optional[str] = None

class SaveGTReq(BaseModel):
    doc_id: str
    key: str
    value: Optional[str] = None
    rect: Optional[GTRect] = None
    user: Optional[str] = None
    ts: Optional[str] = None

@router.get("/gt/{doc_id}")
def read_gt(doc_id: str):
    p = _gt_path(doc_id)
    if not p.exists():
        return {"doc_id": doc_id, "gt": {}}
    return {"doc_id": doc_id, "gt": json.loads(p.read_text())}

@router.post("/gt/save")
def save_gt(req: SaveGTReq):
    p = _gt_path(req.doc_id)
    data: Dict[str, Any] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}

    data.setdefault(req.key, {})
    rec = data[req.key]
    if req.value is not None:
        rec["value"] = req.value
    if req.rect is not None:
        rec["rect"] = req.rect.model_dump()
    if req.user: rec["updated_by"] = req.user
    if req.ts:   rec["updated_at"] = req.ts

    p.write_text(json.dumps(data, indent=2))
    return {"ok": True, "doc_id": req.doc_id, "key": req.key}

# -------- IoU report ----------------------------------------------------------
def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax0, ay0, ax1, ay1 = min(a["x0"], a["x1"]), min(a["y0"], a["y1"]), max(a["x0"], a["x1"]), max(a["y0"], a["y1"])
    bx0, by0, bx1, by1 = min(b["x0"], b["x1"]), min(b["y0"], b["y1"]), max(b["x0"], b["x1"]), max(b["y0"], b["y1"])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    au = (ax1 - ax0) * (ay1 - ay0); bu = (bx1 - bx0) * (by1 - by0)
    denom = au + bu - inter
    return float(inter / denom) if denom > 0 else 0.0

class IoUReq(BaseModel):
    doc_id: str
    preds: Dict[str, Dict[str, Any]]  # {"fuzzy":{key:{page:int,rect:{...}}}, "tfidf":{...}, ...}

@router.post("/gt/iou")
def iou_report(req: IoUReq):
    p = _gt_path(req.doc_id)
    if not p.exists():
        raise HTTPException(404, "No ground truth saved for this doc")
    gt = json.loads(p.read_text())

    report: Dict[str, Dict[str, float]] = {}
    for method, keys in req.preds.items():
        m_scores: Dict[str, float] = {}
        for key, pred in keys.items():
            gt_rec = gt.get(key)
            if not gt_rec or "rect" not in gt_rec: 
                continue
            if not pred or "rect" not in pred: 
                m_scores[key] = 0.0
                continue
            if gt_rec["rect"].get("page") != pred.get("page"):
                m_scores[key] = 0.0
                continue
            m_scores[key] = _iou(gt_rec["rect"], pred["rect"])
        report[method] = m_scores

    # summarize
    summary = {m: float(np.mean(list(kvs.values()))) if kvs else 0.0
               for m, kvs in report.items()}
    return {"doc_id": req.doc_id, "report": report, "summary": summary}