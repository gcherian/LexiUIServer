# ocr.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

@dataclass
class OCRToken:
    text: str
    bbox: List[float]   # [x0,y0,x1,y1] in page pixel coords (origin top-left)
    page: int           # 1-based page index
    angle: int          # 0,90,180,270 — rotation that yielded the token
    conf: float         # tesseract conf (0..100)

def _image_to_tokens(img: Image.Image, angle: int, page: int, psm: int = 6) -> List[OCRToken]:
    """
    Run Tesseract on a PIL image already rotated to `angle`.
    Then rotate all bboxes back to the original (angle 0) page space.
    """
    # run OCR on the rotated image
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    W, H = img.size
    tokens: List[OCRToken] = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        conf_str = data["conf"][i]
        try:
            conf = float(conf_str)
        except Exception:
            conf = -1.0
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        rx0, ry0, rx1, ry1 = float(x), float(y), float(x + w), float(y + h)

        # rotate bbox back to page coords (inverse rotation)
        bx0, by0, bx1, by1 = _rotate_box_back((rx0, ry0, rx1, ry1), angle, W, H)

        tokens.append(OCRToken(
            text=txt,
            bbox=[bx0, by0, bx1, by1],
            page=page,
            angle=angle,
            conf=conf
        ))
    return tokens

def _rotate_box_back(b: Tuple[float,float,float,float], angle: int, W: int, H: int) -> Tuple[float,float,float,float]:
    """
    Map a box from rotated image space back to unrotated page space.
    Coordinates are top-left origin.
    """
    x0, y0, x1, y1 = b
    if angle % 360 == 0:
        return x0, y0, x1, y1
    # four corners
    pts = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
    if angle == 90:
        # rotated img has size HxW; inverse map (x,y)_page = (y, W - x)
        inv = np.stack([pts[:,1], W - pts[:,0]], axis=1)
    elif angle == 180:
        inv = np.stack([W - pts[:,0], H - pts[:,1]], axis=1)
    elif angle == 270:
        inv = np.stack([H - pts[:,1], pts[:,0]], axis=1)
    else:
        # fallback: compute around center
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
    # clamp
    nx0, ny0 = max(0, nx0), max(0, ny0)
    nx1, ny1 = min(W, nx1), min(H, ny1)
    return float(nx0), float(ny0), float(nx1), float(ny1)

def _rotate_image(img: Image.Image, angle: int) -> Image.Image:
    if angle % 360 == 0:
        return img
    # expand=False keeps size WxH; we want same dims for easy back-rotation math
    return img.rotate(angle, expand=False, resample=Image.BICUBIC)

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

def _dedup_tokens(tokens: List[OCRToken], iou_thr: float = 0.6) -> List[OCRToken]:
    """
    Deduplicate overlapping boxes from different rotations.
    Keep the one with the highest conf; if tie, prefer angle=0.
    """
    if not tokens:
        return []
    boxes = np.array([t.bbox for t in tokens], dtype=np.float32)
    confs = np.array([t.conf for t in tokens], dtype=np.float32)
    # simple NMS
    keep = []
    idxs = list(range(len(tokens)))
    # sort by conf desc, angle==0 slight bonus
    order = sorted(idxs, key=lambda i: (confs[i] + (0.1 if tokens[i].angle==0 else 0.0)), reverse=True)
    suppressed = set()
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
    """
    OCR a PDF into per-token dicts with vertical text support via multi-rotation.
    Returns tokens as list of {text, bbox:[x0,y0,x1,y1], page:int, angle:int, conf:float}
    """
    pdf_path = str(pdf_path)
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    all_tokens: List[OCRToken] = []
    for pnum, img in enumerate(pages, start=1):
        for angle in (0, 90, 180, 270):
            rimg = _rotate_image(img, angle)
            toks = _image_to_tokens(rimg, angle=angle, page=pnum, psm=psm)
            all_tokens.extend(toks)
    dedup = _dedup_tokens(all_tokens, iou_thr=0.6)
    # jsonify
    out = []
    for t in dedup:
        out.append({
            "text": t.text,
            "bbox": [float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])],
            "page": int(t.page),
            "angle": int(t.angle),
            "conf": float(t.conf)
        })
    return out

def image_to_tokens(img_path: str | Path, psm: int = 6) -> List[Dict[str, Any]]:
    """
    OCR a single image file (PNG/JPG) with multi-rotation like above.
    """
    img = Image.open(str(img_path)).convert("RGB")
    W, H = img.size
    all_tokens: List[OCRToken] = []
    for angle in (0, 90, 180, 270):
        rimg = _rotate_image(img, angle)
        toks = _image_to_tokens(rimg, angle=angle, page=1, psm=psm)
        all_tokens.extend(toks)
    dedup = _dedup_tokens(all_tokens, iou_thr=0.6)
    return [{
        "text": t.text,
        "bbox": [float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])],
        "page": int(t.page),
        "angle": int(t.angle),
        "conf": float(t.conf)
    } for t in dedup]