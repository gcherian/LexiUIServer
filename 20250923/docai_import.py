from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4
import json
import re

# reuse ocr_unified paths
from .ocr_unified import doc_dir, boxes_path, meta_path, page_text_path

router = APIRouter(prefix="/lasso", tags=["docai-import"])

# ---------- helpers ----------
def _num(x, dflt=0.0):
    try:
        return float(x)
    except Exception:
        return float(dflt)

def _kv(path: str, val):
    # val can be scalar or object-with-value
    if isinstance(val, dict) and "value" in val:
        v = val.get("value")
    else:
        v = val
    try:
        return str(v) if v is not None else ""
    except Exception:
        return ""

_BBOX_KEYS = ("x","y","width","height","x0","y0","x1","y1")

def _rect_from_bbox(b):
    if not isinstance(b, dict):
        return None
    # Google Doc AI-like: x,y,width,height
    if all(k in b for k in ("x","y","width","height")):
        x0 = _num(b["x"]); y0 = _num(b["y"])
        x1 = x0 + _num(b["width"]); y1 = y0 + _num(b["height"])
        return {"x0":x0,"y0":y0,"x1":x1,"y1":y1}
    # generic x0,y0,x1,y1
    if all(k in b for k in ("x0","y0","x1","y1")):
        return {"x0":_num(b["x0"]),"y0":_num(b["y0"]),"x1":_num(b["x1"]),"y1":_num(b["y1"])}
    return None

def _flatten(obj, prefix=""):
    rows = []
    if isinstance(obj, dict):
        for k,v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            # capture rects if present
            rects = None
            if isinstance(v, dict):
                cand = v.get("boundingBox") or v.get("bbox") or v.get("rect") or v.get("box")
                if cand:
                    r = _rect_from_bbox(cand)
                    pg = int(v.get("page") or v.get("pageNumber") or v.get("page_index") or 1)
                    if r:
                        rects = [{"page": pg, **r}]
            if isinstance(v, (dict,list)):
                # if structured “value” present, treat as leaf
                if isinstance(v, dict) and "value" in v:
                    rows.append({"key": p, "value": _kv(p, v), "rects": rects})
                else:
                    rows.extend(_flatten(v, p))
            else:
                rows.append({"key": p, "value": _kv(p, v), "rects": rects})
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            p = f"{prefix}[{i}]" if prefix else f"[{i}]"
            rows.extend(_flatten(v, p))
    else:
        rows.append({"key": prefix or "(value)", "value": _kv(prefix, obj)})
    return rows

# ---------- endpoint ----------
@router.post("/docai/import")
async def import_docai_json(docai_json: UploadFile = File(...)):
    """
    Accepts a Google DocAI-like JSON structure (your sample):
    { "documents": [ { "properties": { "metadata": { "parser": "...", "metadataMap": {...}, "pages":[ { "elements":[ { "elementType":"paragraph", "content":"...", "boundingBox":{x,y,width,height}, "page":1 }, ... ] } ] } } } ] }
    We will:
      - Create a new doc_id folder
      - Write boxes.json from all (elements[].content, boundingBox,page)
      - Write meta.json pages sizes if available (fallback to 2550x3300)
      - Return flattened rows from metadataMap for the left table
    """
    try:
        payload = json.loads((await docai_json.read()).decode("utf-8", errors="ignore"))
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON: {e}")

    # locate root
    # tolerate: whole object holds metadata/pages directly or nested under documents[0].properties.metadata
    meta_root = payload
    if isinstance(payload, dict) and "documents" in payload and payload["documents"]:
        d0 = payload["documents"][0]
        meta_root = d0.get("properties", {}).get("metadata", d0.get("metadata", d0))
    metadata_map = meta_root.get("metadataMap") or meta_root.get("metadata_map") or {}
    pages = meta_root.get("pages") or []

    # build tokens from pages[].elements[*]
    all_boxes = []
    page_meta = []
    default_w, default_h = 2550.0, 3300.0
    for pi, pg in enumerate(pages, start=1):
        pw = _num(pg.get("width"), default_w)
        ph = _num(pg.get("height"), default_h)
        page_meta.append({"page": pi, "width": pw, "height": ph})
        elements = pg.get("elements") or []
        for el in elements:
            txt = (el.get("content") or "").strip()
            bbox = el.get("boundingBox") or el.get("bbox") or el.get("rect") or {}
            rect = _rect_from_bbox(bbox)
            if not txt or not rect:
                continue
            all_boxes.append({"page": int(el.get("page") or pi), **rect, "text": txt})

    # if no pages at all, fallback single page & try top-level elements
    if not pages:
        elements = meta_root.get("elements") or []
        if elements:
            page_meta.append({"page":1,"width":default_w,"height":default_h})
            for el in elements:
                txt = (el.get("content") or "").strip()
                rect = _rect_from_bbox(el.get("boundingBox") or {})
                if txt and rect:
                    all_boxes.append({"page": int(el.get("page") or 1), **rect, "text": txt})

    # persist
    doc_id = uuid4().hex[:12]
    dd = doc_dir(doc_id)  # ensure
    boxes_path(doc_id).write_text(json.dumps(all_boxes, ensure_ascii=False))
    meta_path(doc_id).write_text(json.dumps({"pages": page_meta or [{"page":1,"width":default_w,"height":default_h}]}, indent=2))
    # dump “pages text” (optional)
    page_text = {}
    for b in all_boxes:
        page_text.setdefault(b["page"], []).append(b.get("text") or "")
    page_text_path(doc_id).write_text("\n\n".join(" ".join(v) for _,v in sorted(page_text.items())))

    # flatten metadataMap for left panel
    rows = _flatten(metadata_map)

    return {
        "doc_id": doc_id,
        "pages": page_meta or [{"page":1,"width":default_w,"height":default_h}],
        "boxes": all_boxes,
        "rows": rows,
    }
