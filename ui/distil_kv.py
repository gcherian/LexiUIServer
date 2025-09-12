# File: src/distil_kv.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json, math
import numpy as np

import pytesseract
import pypdfium2 as pdfium
from PIL import Image, ImageOps

from transformers import DistilBertTokenizerFast, DistilBertModel
import torch

router = APIRouter(prefix="/distil", tags=["distil-kv"])

PROJECT_ROOT: Path = Path(".")
DATA: Path = Path(".")

def init_paths(project_root: Path):
    global PROJECT_ROOT, DATA
    PROJECT_ROOT = project_root.resolve()
    DATA = (PROJECT_ROOT / "data").resolve()

# ---------------- Models ----------------

class FieldSpec(BaseModel):
    key: str
    label: str
    type: Optional[str] = None  # "text" | "zip" | "date" | "amount" | "email" | ...

class ExtractReq(BaseModel):
    doc_id: str
    fields: List[FieldSpec]
    max_window: int = 8
    dpi: int = 260

# ---------------- Utils ----------------

def pdf_path(doc_id: str) -> Path:    return DATA / doc_id / "original.pdf"
def meta_path(doc_id: str) -> Path:   return DATA / doc_id / "meta.json"
def boxes_path(doc_id: str) -> Path:  return DATA / doc_id / "boxes.json"

def render_pdf_pages(pdf_file: Path, dpi: int):
    pdf = pdfium.PdfDocument(str(pdf_file))
    for i in range(len(pdf)):
        page = pdf[i]
        zoom = dpi / 72
        pil = page.render(scale=zoom).to_pil()
        yield i+1, pil, pil.width, pil.height

def preprocess(img: Image.Image) -> Image.Image:
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray

def tesseract_image_to_data(img: Image.Image, lang="eng", psm=6, oem=1):
    cfg = f"--oem {oem} --psm {psm}"
    return pytesseract.image_to_data(img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)

def union_rect(boxes: List[Dict[str, float]]) -> Dict[str, float]:
    x0 = min(b["x0"] for b in boxes); y0 = min(b["y0"] for b in boxes)
    x1 = max(b["x1"] for b in boxes); y1 = max(b["y1"] for b in boxes)
    return {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}

def norm(s: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in (s or ""))
        .split()
    )

def is_zip(s: str) -> bool:
    s2 = s.strip()
    if len(s2) >= 10 and s2[:5].isdigit() and (s2[5] in "- "):
        return True
    return s2.isdigit() and (len(s2) in (5, 9))

def is_amount(s: str) -> bool:
    s = s.strip()
    return bool(s and any(ch.isdigit() for ch in s) and all(c in "0123456789.,-$ "() for c in s))

def is_dateish(s: str) -> bool:
    s = s.strip()
    return any(sym in s for sym in "/-") and any(ch.isdigit() for ch in s)

def is_email(s: str) -> bool:
    return "@" in s and "." in s

TYPE_CHECK = {"zip": is_zip, "amount": is_amount, "date": is_dateish, "email": is_email}

# ---------------- DistilBERT ----------------

_tokenizer = None
_encoder = None
_device = "cpu"

def _load_encoder():
    global _tokenizer, _encoder, _device
    if _tokenizer is None:
        _tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    if _encoder is None:
        _encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        _encoder.eval()
        _encoder.to(_device)

@torch.no_grad()
def embed_texts(texts: List[str]) -> np.ndarray:
    _load_encoder()
    toks = _tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    toks = {k: v.to(_device) for k, v in toks.items()}
    out = _encoder(**toks).last_hidden_state[:, 0, :]
    vecs = out.cpu().numpy()
    norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-6, None)
    return vecs / norms

def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

# ---------------- OCR load/run ----------------

def load_or_run_ocr(doc_id: str, dpi: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bp = boxes_path(doc_id); mp = meta_path(doc_id)
    if bp.exists() and mp.exists():
        return json.loads(bp.read_text()), json.loads(mp.read_text())

    pdf = pdf_path(doc_id)
    if not pdf.exists():
        raise HTTPException(404, f"PDF not found for {doc_id}")

    all_boxes: List[Dict[str, Any]] = []
    pages_meta: List[Dict[str, float]] = []
    for page_no, pil, w, h in render_pdf_pages(pdf, dpi):
        img = preprocess(pil)
        d = tesseract_image_to_data(img, psm=6, oem=1)
        toks = []
        for i in range(len(d["text"])):
            txt = (d["text"][i] or "").strip()
            if not txt: continue
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            toks.append({"page": page_no, "x0": float(x), "y0": float(y), "x1": float(x+ww), "y1": float(y+hh), "text": txt})
        toks.sort(key=lambda t: (t["y0"], t["x0"]))
        all_boxes.extend(toks)
        pages_meta.append({"page": page_no, "width": float(w), "height": float(h)})

    mp_out = {"pages": pages_meta, "params": {"dpi": dpi, "psm": 6, "oem": 1, "lang": "eng"}}
    meta_path(doc_id).write_text(json.dumps(mp_out, indent=2))
    boxes_path(doc_id).write_text(json.dumps(all_boxes))
    return all_boxes, mp_out

# ---------------- Line grouping + n-grams ----------------

def group_lines(page_tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group tokens into lines by y-center proximity; keeps reading order.
    """
    if not page_tokens: return []
    # sort by y then x
    toks = sorted(page_tokens, key=lambda t: (t["y0"], t["x0"]))
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = [toks[0]]
    def y_center(t): return (t["y0"] + t["y1"]) / 2
    band = max(3.0, (toks[0]["y1"] - toks[0]["y0"]) * 0.6 + 2)

    for t in toks[1:]:
        prev = cur[-1]
        if abs(y_center(t) - y_center(prev)) <= band:
            cur.append(t)
        else:
            lines.append(cur)
            cur = [t]
            band = max(3.0, (t["y1"] - t["y0"]) * 0.6 + 2)
    if cur: lines.append(cur)
    # ensure each line is left->right
    for ln in lines:
        ln.sort(key=lambda t: t["x0"])
    return lines

def candidate_ngrams_line(line: List[Dict[str, Any]], max_window: int) -> List[Dict[str, Any]]:
    """
    Build n-grams **within a single line only**. This is key to avoid full-width unions.
    """
    out: List[Dict[str, Any]] = []
    n = len(line)
    for i in range(n):
        acc_text = []
        acc_boxes = []
        for w in range(max_window):
            j = i + w
            if j >= n: break
            tj = line[j]
            acc_text.append(tj["text"])
            acc_boxes.append(tj)
            text = " ".join(acc_text)
            ub = union_rect(acc_boxes)
            out.append({"text": text, "boxes": acc_boxes.copy(), "union": ub})
    return out

def candidate_ngrams(tokens: List[Dict[str, Any]], page: int, max_window: int) -> List[Dict[str, Any]]:
    page_toks = [t for t in tokens if t["page"] == page]
    lines = group_lines(page_toks)
    out: List[Dict[str, Any]] = []
    for ln in lines:
        for ng in candidate_ngrams_line(ln, max_window):
            ng["page"] = page
            out.append(ng)
    return out

# ---------------- Scoring & refinement ----------------

def type_score(field_type: Optional[str], text: str) -> float:
    if not field_type: return 0.0
    func = TYPE_CHECK.get(field_type)
    if not func: return 0.0
    return 0.25 if func(text) else -0.25

def compactness_penalty(ngram_boxes: List[Dict[str, float]], union: Dict[str, float], page_w: float) -> float:
    """
    Penalize overly-wide spans, large inter-token gaps, and too many tokens.
    Negative values reduce total score.
    """
    if not ngram_boxes: return 0.0
    # width penalty vs page width
    width_frac = (union["x1"] - union["x0"]) / max(1.0, page_w)
    p_width = -0.6 * max(0.0, width_frac - 0.35)   # allow ~35% width before penalties ramp

    # gap penalty
    boxes = sorted(ngram_boxes, key=lambda b: b["x0"])
    avg_h = sum((b["y1"] - b["y0"]) for b in boxes) / len(boxes)
    gaps = []
    for a, b in zip(boxes, boxes[1:]):
        gaps.append(max(0.0, b["x0"] - a["x1"]))
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
    p_gap = -0.4 * min(1.0, mean_gap / max(6.0, avg_h * 0.8))  # normalize by text height

    # token-count penalty (discourage line-eating)
    p_tok = -0.08 * max(0, len(ngram_boxes) - 4)

    return p_width + p_gap + p_tok

def spatial_bias(union: Dict[str, float], page_w: float, page_h: float) -> float:
    cx = (union["x0"] + union["x1"]) / 2 / max(1.0, page_w)
    cy = (union["y0"] + union["y1"]) / 2 / max(1.0, page_h)
    return 0.04 * (0.5*cx + 0.5*cy)

def score_span(field_vec: np.ndarray,
               span_texts: List[str],
               unions: List[Dict[str, float]],
               box_lists: List[List[Dict[str, float]]],
               page_w: float, page_h: float,
               field_type: Optional[str]) -> Tuple[float, int]:
    """
    Vectorized: compute semantic sims for all candidates, add penalties, pick best index.
    """
    vecs = embed_texts([norm(t) for t in span_texts])  # NxD
    sims = cosine(vecs, field_vec)[:, 0]               # N
    best_score = -1e9
    best_i = 0
    for i, sim in enumerate(sims):
        s_sem = float(sim)
        s_type = type_score(field_type, span_texts[i])
        s_cmp  = compactness_penalty(box_lists[i], unions[i], page_w)
        s_sp   = spatial_bias(unions[i], page_w, page_h)
        s = s_sem + s_type + s_cmp + s_sp
        if s > best_score:
            best_score = s; best_i = i
    return best_score, best_i

def greedy_trim(field_vec: np.ndarray,
                boxes: List[Dict[str, float]],
                page_w: float, page_h: float,
                field_type: Optional[str]) -> Tuple[List[Dict[str, float]], Dict[str, float], str, float]:
    """
    After selecting a span, trim tokens from ends if it does not hurt (or improves) the score.
    This yields tight boxes (no full-line unions).
    """
    if not boxes: return boxes, union_rect(boxes), " ".join(b["text"] for b in boxes), 0.0
    # initial
    text0 = " ".join(b["text"] for b in boxes)
    union0 = union_rect(boxes)
    score0, _ = score_span(field_vec, [text0], [union0], [boxes], page_w, page_h, field_type)

    changed = True
    best_boxes = boxes[:]; best_union = union0; best_text = text0; best_score = score0

    while changed and len(best_boxes) > 1:
        changed = False
        candidates: List[Tuple[List[Dict[str,float]], str, Dict[str,float]]] = []

        # drop left
        left = best_boxes[1:]
        candidates.append((left, " ".join(b["text"] for b in left), union_rect(left)))

        # drop right
        right = best_boxes[:-1]
        candidates.append((right, " ".join(b["text"] for b in right), union_rect(right)))

        # score both and pick if better or equal (<= small epsilon)
        span_texts = [c[1] for c in candidates]
        unions = [c[2] for c in candidates]
        box_lists = [c[0] for c in candidates]
        s, idx = score_span(field_vec, span_texts, unions, box_lists, page_w, page_h, field_type)

        # allow equal or slightly higher to make box tighter
        if s >= best_score - 1e-4:
            best_boxes = candidates[idx][0]
            best_text  = candidates[idx][1]
            best_union = candidates[idx][2]
            best_score = s
            changed = True

    return best_boxes, best_union, best_text, best_score

def best_match_for_field(field: FieldSpec,
                         page: int,
                         page_ngrams: List[Dict[str, Any]],
                         field_vec: np.ndarray,
                         page_w: float,
                         page_h: float) -> Optional[Dict[str, Any]]:
    if not page_ngrams: return None
    texts = [ng["text"] for ng in page_ngrams]
    unions = [ng["union"] for ng in page_ngrams]
    blists = [ng["boxes"] for ng in page_ngrams]
    base_score, base_idx = score_span(field_vec, texts, unions, blists, page_w, page_h, field.type)

    # Greedy end-trim to tighten
    sel_boxes = page_ngrams[base_idx]["boxes"]
    tight_boxes, tight_union, tight_text, tight_score = greedy_trim(field_vec, sel_boxes, page_w, page_h, field.type)

    return {
        "page": page,
        "text": tight_text,
        "boxes": tight_boxes,
        "union": tight_union,
        "score": float(tight_score)
    }

# ---------------- API ----------------

@router.get("/health")
def health():
    return {"ok": True, "svc": "distil-kv", "encoder": "distilbert-base-uncased"}

@router.post("/extract")
def extract(req: ExtractReq):
    if not req.fields:
        raise HTTPException(400, "fields[] required")

    tokens, meta = load_or_run_ocr(req.doc_id, req.dpi)
    pages_meta = {int(p["page"]): p for p in meta.get("pages", [])}
    pages = sorted(pages_meta.keys())

    # Precompute field vectors
    field_texts = [norm(f.label or f.key) for f in req.fields]
    field_vecs = embed_texts(field_texts)  # K x d

    # Build page n-grams once
    ngrams_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for pg in pages:
        ngrams_by_page[pg] = candidate_ngrams(tokens, pg, req.max_window)

    result_fields: List[Dict[str, Any]] = []
    for k, field in enumerate(req.fields):
        field_vec = field_vecs[k:k+1, :]
        best: Optional[Dict[str, Any]] = None

        for pg in pages:
            gn = ngrams_by_page.get(pg, [])
            if not gn: continue
            pm = pages_meta.get(pg, {})
            cand = best_match_for_field(field, pg, gn, field_vec, pm.get("width", 1.0), pm.get("height", 1.0))
            if not cand: continue
            if (best is None) or (cand["score"] > best["score"]):
                best = cand

        if best:
            value_text = best["text"]
            value_union = best["union"]
            value_boxes = [
                {"page": best["page"], "x0": float(b["x0"]), "y0": float(b["y0"]), "x1": float(b["x1"]), "y1": float(b["y1"])}
                for b in best["boxes"]
            ]
            key_box = _approx_key_box_on_left(tokens, best["page"], best["boxes"][0]) if best["boxes"] else None

            result_fields.append({
                "key": field.key,
                "label": field.label,
                "type": field.type,
                "page": best["page"],
                "key_box": key_box,
                "value": value_text,
                "value_boxes": value_boxes,
                "value_union": {"page": best["page"], **value_union},
                "confidence": round(float(best["score"]), 4),
            })
        else:
            result_fields.append({
                "key": field.key, "label": field.label, "type": field.type,
                "page": None, "key_box": None, "value": "",
                "value_boxes": [], "value_union": None, "confidence": 0.0
            })

    return {"doc_id": req.doc_id, "fields": result_fields, "pages": pages, "dpi": req.dpi}

def _approx_key_box_on_left(tokens: List[Dict[str, Any]], page: int, first_val_token: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    vy0, vy1 = first_val_token["y0"], first_val_token["y1"]
    vcy = (vy0 + vy1) / 2
    band = (vy1 - vy0) * 0.7 + 2
    candidates = []
    for t in tokens:
        if t["page"] != page: continue
        cy = (t["y0"] + t["y1"]) / 2
        if abs(cy - vcy) <= band and t["x1"] <= first_val_token["x0"]:
            candidates.append(t)
    if not candidates: return None
    candidates.sort(key=lambda t: abs(first_val_token["x0"] - t["x1"]))
    window = candidates[:3]
    u = union_rect(window)
    return {"page": page, **u}