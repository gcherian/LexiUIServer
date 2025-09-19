from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from pathlib import Path
import json, time

# Reuse path helpers & OCR bits from your big unified router
# (adjust import path if your package layout differs)
from .ocr_unified import doc_dir, boxes_path, meta_path, fields_path

# Reuse locate implementation so we score all 4 models (fuzzy, tfidf, minilm, distilbert)
# You already added this router earlier.
from .lasso_locate import LocateReq, locate as locate_api

router = APIRouter(prefix="/lasso", tags=["lasso-gt"])

# ------------------------------------------------------------
# Storage model
#   data/<doc_id>/gt.json
#   {
#     "doc_id": "...",
#     "fields": {
#        "<key>": {
#           "bbox": {"page":1,"x0":...,"y0":...,"x1":...,"y1":...},
#           "text": "ground truth text (optional)",
#           "ts": 169xxxxxxx
#        },
#        ...
#     }
#   }
# ------------------------------------------------------------

def _gt_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / "gt.json"

def _read_json(p: Path, default: Any) -> Any:
    if not p.exists(): return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default

def _write_json(p: Path, obj: Any):
    p.write_text(json.dumps(obj, indent=2))

def _norm_rect(r: Dict[str, float]) -> Dict[str, float]:
    x0 = float(min(r["x0"], r["x1"]))
    y0 = float(min(r["y0"], r["y1"]))
    x1 = float(max(r["x0"], r["x1"]))
    y1 = float(max(r["y0"], r["y1"]))
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    # assumes same page is checked by caller
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    inter_x0 = max(ax0, bx0); inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1); inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    a_area = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = a_area + b_area - inter
    if union <= 0: return 0.0
    return float(inter / union)

# -------------------- Schemas --------------------

class GTBox(BaseModel):
    page: int
    x0: float; y0: float; x1: float; y1: float

class GTSaveReq(BaseModel):
    doc_id: str
    key: str
    bbox: GTBox
    text: Optional[str] = None  # allow saving the corrected OCR value, too

class GTOne(BaseModel):
    key: str
    bbox: Optional[GTBox] = None
    text: Optional[str] = None
    ts: Optional[int] = None

class GTEvalReq(BaseModel):
    doc_id: str
    # optional: evaluate specific keys; else evaluate all that have GT
    keys: Optional[List[str]] = None
    # run locate with this value (if provided) to simulate "query" context
    # if omitted we try to use the GT text; if that’s also empty we fall back to empty string
    value_override: Optional[str] = None
    # how many token windows per page to scan in locate; default = lasso_locate default
    max_window: Optional[int] = None
    # optional model root override
    models_root: Optional[str] = None

# -------------------- Endpoints --------------------

@router.get("/gt/get")
def gt_get(doc_id: str = Query(...), key: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    Get GT for a doc (all fields) or a single field.
    """
    gp = _gt_path(doc_id)
    data = _read_json(gp, {"doc_id": doc_id, "fields": {}})
    if key:
        entry = data.get("fields", {}).get(key)
        return {"doc_id": doc_id, "key": key, "gt": entry or None}
    return data

@router.post("/gt/save")
def gt_save(req: GTSaveReq) -> Dict[str, Any]:
    """
    Save/overwrite ground truth for (doc_id, key), incrementally.
    """
    if not meta_path(req.doc_id).exists():
        raise HTTPException(404, "doc not found; OCR meta missing")

    # normalize
    norm = _norm_rect(req.bbox.model_dump())
    entry = {"bbox": {"page": int(req.bbox.page), **norm}, "text": (req.text or ""), "ts": int(time.time())}

    gp = _gt_path(req.doc_id)
    data = _read_json(gp, {"doc_id": req.doc_id, "fields": {}})
    if "fields" not in data or not isinstance(data["fields"], dict):
        data["fields"] = {}
    data["fields"][req.key] = entry
    _write_json(gp, data)
    return {"ok": True, "saved": {"key": req.key, **entry}}

@router.post("/gt/eval_one")
def gt_eval_one(req: GTEvalReq) -> Dict[str, Any]:
    """
    Evaluate IoU for one or more keys in a doc using GT versus locate results for each model.
    Returns per-key IoU and a small summary.
    """
    gp = _gt_path(req.doc_id)
    data = _read_json(gp, {"doc_id": req.doc_id, "fields": {}})
    fields: Dict[str, Any] = data.get("fields", {})

    if not fields:
        raise HTTPException(404, "No GT saved for this document yet.")

    keys = req.keys or list(fields.keys())
    results = {}
    summary = {"count": 0, "iou_avg": {}, "iou_macro": 0.0}

    # accumulate per-model sums for macro
    per_model_sums: Dict[str, float] = {"autolocate": 0.0, "tfidf": 0.0, "minilm": 0.0, "distilbert": 0.0}
    per_model_count = 0

    for key in keys:
        gt = fields.get(key)
        if not gt or "bbox" not in gt:
            results[key] = {"error": "no_gt_for_key"}
            continue

        gt_page = int(gt["bbox"]["page"])
        gt_rect = {k: float(gt["bbox"][k]) for k in ("x0","y0","x1","y1")}

        # choose a query "value": override > GT text > ""
        q_val = req.value_override if (req.value_override is not None) else (gt.get("text") or "")

        # Call locate (in-process)
        loc_req = LocateReq(
            doc_id=req.doc_id,
            key=key,
            value=q_val,
            max_window=(req.max_window or 12),
            models_root=(req.models_root or None)
        )
        try:
            loc_res = locate_api(loc_req)  # returns dict {"hits": {...}, "errors": {...}, ...}
        except Exception as e:
            results[key] = {"error": f"locate_failed: {e!r}"}
            continue

        hits = (loc_res or {}).get("hits", {})
        errs = (loc_res or {}).get("errors", {})

        key_out = {"iou": {}, "pages_match": {}, "errors": errs}
        for model_name in ("autolocate","tfidf","minilm","distilbert"):
            hit = hits.get(model_name)
            if not hit or not hit.get("rect"):
                key_out["iou"][model_name] = None
                key_out["pages_match"][model_name] = None
                continue

            page_ok = int(hit.get("page", -1)) == gt_page
            key_out["pages_match"][model_name] = bool(page_ok)
            iou = _iou(gt_rect, hit["rect"]) if page_ok else 0.0
            key_out["iou"][model_name] = float(iou)

            if iou is not None:
                per_model_sums[model_name] += float(iou)
        results[key] = key_out
        summary["count"] += 1
        per_model_count += 1

    # summarize
    if per_model_count > 0:
        for m in per_model_sums:
            summary["iou_avg"][m] = per_model_sums[m] / max(1, per_model_count)
        # simple macro as mean of available model averages
        if summary["iou_avg"]:
            summary["iou_macro"] = sum(summary["iou_avg"].values()) / max(1, len(summary["iou_avg"]))

    return {"doc_id": req.doc_id, "keys": keys, "results": results, "summary": summary}

@router.get("/gt/eval_doc")
def gt_eval_doc(doc_id: str, models_root: Optional[str] = None, max_window: int = 12) -> Dict[str, Any]:
    """
    Convenience wrapper to evaluate ALL GT keys in a doc.
    """
    payload = GTEvalReq(doc_id=doc_id, keys=None, models_root=models_root, max_window=max_window)
    return gt_eval_one(payload)

@router.get("/gt/file")
def gt_file(doc_id: str):
    """
    Return the raw gt.json (handy for debugging/download).
    """
    gp = _gt_path(doc_id)
    if not gp.exists():
        raise HTTPException(404, "gt.json not found")
    return _read_json(gp, {})
