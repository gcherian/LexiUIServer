# File: src/routers/lasso_gt.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, Optional, List
import json, time, traceback
import numpy as np

# Reuse file-layout helpers from your unified OCR router
from .ocr_unified import doc_dir, boxes_path, meta_path  # adjust import to your layout

# Reuse locate endpoint by calling it in-process
from .lasso_locate import locate as locate_rpc, LocateReq  # adjust if your module name differs

router = APIRouter(prefix="/lasso", tags=["lasso-gt"])

# ---------------- models ----------------
class Rect(BaseModel):
    page: int
    x0: float; y0: float; x1: float; y1: float

class SaveGTReq(BaseModel):
    doc_id: str
    key: str
    value_text: Optional[str] = None  # text you want as truth (optional)
    rect: Rect                        # pink box you placed (truth)
    models_root: Optional[str] = None # optional override for locate

def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def _char_cos(a: str, b: str) -> float:
    # lightweight character-ngram cosine (HashingVectorizer would be overkill here)
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b: return 0.0
    def grams(s, n=4):
        s = f"  {s}  "
        return [s[i:i+n] for i in range(max(0, len(s)-n+1))]
    from collections import Counter
    ca, cb = Counter(grams(a)), Counter(grams(b))
    keys = set(ca) | set(cb)
    va = np.array([ca[k] for k in keys], dtype="float32")
    vb = np.array([cb[k] for k in keys], dtype="float32")
    na = float(np.linalg.norm(va)); nb = float(np.linalg.norm(vb))
    return 0.0 if na == 0 or nb == 0 else float(np.clip(np.dot(va, vb) / (na * nb), 0, 1))

def _safe(obj: Any, default=None):
    try:
        return obj if obj is not None else default
    except Exception:
        return default

def _now_ts() -> int:
    return int(time.time())

@router.post("/groundtruth/save")
def save_groundtruth(req: SaveGTReq):
    """
    Save the pink box you drew as ground truth for (doc_id, key), compare all
    current methods’ predictions to this truth, compute IoU and text-sim, and log it.

    Returns: { saved: {...}, eval: {method -> {iou, ok, score?}}, errors: {...} }
    """
    # ---- validate doc state ----
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "Document tokens/meta missing. Upload & OCR first.")

    # ---- persist GT on disk (per-doc JSON) ----
    gtf = doc_dir(req.doc_id) / "groundtruth.json"
    gt = {}
    if gtf.exists():
        try:
            gt = json.loads(gtf.read_text())
        except Exception:
            gt = {}
    gt.setdefault(req.key, {})
    gt[req.key] = {
        "rect": req.rect.model_dump(),
        "value_text": req.value_text,
        "ts": _now_ts(),
    }
    gtf.write_text(json.dumps(gt, indent=2))

    # ---- call locate for the 5 methods (whatever works) ----
    errors: Dict[str, str] = {}
    methods_out: Dict[str, Any] = {}
    try:
        # We call the existing locate with the user-provided key/value_text
        lreq = LocateReq(
            doc_id=req.doc_id,
            key=req.key,
            value=req.value_text or "",
            max_window=12,
            models_root=req.models_root,
        )
        loc = locate_rpc(lreq)
        # Accept either {"methods":{...}} or {"hits":{...}} based on your current file
        methods_out = _safe(loc.get("methods")) or _safe(loc.get("hits")) or {}
        # Also bubble up model loading errors if returned
        if "errors" in loc and isinstance(loc["errors"], dict):
            errors.update({k:str(v) for k,v in loc["errors"].items()})
    except Exception:
        errors["locate"] = traceback.format_exc()

    # ---- compute IoU & text-sim vs GT for each method ----
    eval_out: Dict[str, Any] = {}
    truth_rect = {
        "x0": req.rect.x0, "y0": req.rect.y0, "x1": req.rect.x1, "y1": req.rect.y1
    }
    for name in ("fuzzy","tfidf","minilm","distilbert","layoutlmv3"):
        item = methods_out.get(name)
        if not item:
            eval_out[name] = {"ok": False, "iou": 0.0, "reason": errors.get(name) or "no_result"}
            continue
        try:
            R = item.get("rect") or {}
            pred_rect = {"x0": float(R["x0"]), "y0": float(R["y0"]), "x1": float(R["x1"]), "y1": float(R["y1"])}
            iou = _iou(truth_rect, pred_rect)
            # If you also OCR each method’s rect to text, you can put it here; for now we only have value_text
            tscore = _char_cos(req.value_text or "", req.value_text or "")
            eval_out[name] = {
                "ok": True,
                "iou": round(iou, 4),
                "score": float(item.get("score", 0.0)),
            }
        except Exception:
            eval_out[name] = {"ok": False, "iou": 0.0, "reason": traceback.format_exc()[:200]}

    # ---- append a row to global CSV for training later ----
    gtlog_dir = doc_dir(req.doc_id).parent / "_gt"
    gtlog_dir.mkdir(parents=True, exist_ok=True)
    gtlog_csv = gtlog_dir / "gt_log.csv"
    row = {
        "ts": _now_ts(),
        "doc_id": req.doc_id,
        "key": req.key,
        "page": req.rect.page,
        "x0": req.rect.x0, "y0": req.rect.y0, "x1": req.rect.x1, "y1": req.rect.y1,
        "value_text": req.value_text or "",
        **{f"iou_{k}": _safe(eval_out.get(k,{}).get("iou"), 0.0) for k in ("fuzzy","tfidf","minilm","distilbert","layoutlmv3")},
        **{f"score_{k}": _safe(methods_out.get(k,{}).get("score"), 0.0) for k in ("fuzzy","tfidf","minilm","distilbert","layoutlmv3")},
        **{f"err_{k}": errors.get(k, "") for k in ("locate","minilm","distilbert","layoutlmv3")},
    }
    hdr = list(row.keys())
    if not gtlog_csv.exists():
        gtlog_csv.write_text(",".join(hdr) + "\n")
    with gtlog_csv.open("a", encoding="utf-8") as f:
        f.write(",".join(str(row[h]) for h in hdr) + "\n")

    return {
        "saved": gt[req.key],
        "eval": eval_out,
        "methods": methods_out,   # echo current predictions back to UI (so you can redraw boxes)
        "errors": errors,         # make failures explicit in UI
    }

@router.get("/groundtruth/{doc_id}")
def get_groundtruth(doc_id: str):
    gtf = doc_dir(doc_id) / "groundtruth.json"
    if not gtf.exists():
        return {"doc_id": doc_id, "groundtruth": {}}
    try:
        return {"doc_id": doc_id, "groundtruth": json.loads(gtf.read_text())}
    except Exception:
        return {"doc_id": doc_id, "groundtruth": {}, "error": "invalid_json"}