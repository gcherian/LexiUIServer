# server/routes_match.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json, math, re
from typing import List, Dict, Any, Optional, Tuple

# ---- tiny sentence embeddings (22MB)
try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def emb(s: str):
        return _embed_model.encode([s or ""], normalize_embeddings=True)[0]
except Exception:
    _embed_model = None
    def emb(s: str):
        # fallback: zero vec
        return [0.0] * 384

router = APIRouter(prefix="/match", tags=["kv-match"])

DATA_DIR = Path("data")  # same place where /data/{doc_id}/boxes.json lives

# ---------- Types ----------
class KVReq(BaseModel):
    doc_id: str
    key: str
    label: Optional[str] = None
    value: Optional[str] = None
    page: Optional[int] = None  # optional hint

class Rect(BaseModel):
    page: int
    x0: int
    y0: int
    x1: int
    y1: int

class KVResp(BaseModel):
    rects: List[Rect]
    text: str
    score: float
    panel: Optional[str] = None

# ---------- IO helpers ----------
def load_boxes(doc_id: str) -> List[Dict[str, Any]]:
    p = DATA_DIR / doc_id / "boxes.json"
    if not p.exists():
        raise HTTPException(404, f"boxes.json missing for {doc_id}")
    return json.loads(p.read_text())

# ---------- light utils ----------
_re_zip   = re.compile(r"\b\d{5}(?:-\d{4})?\b")
_re_money = re.compile(r"(?<!\S)[-$€£₹]?\s?\d{1,3}(?:[, ]\d{3})*(?:\.\d{2})?(?!\S)")
_re_date  = re.compile(r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")

ANCHOR_PHRASES = [
    "bill to", "billing", "invoice to", "sold to",
    "ship to", "shipping",
    "remit to", "remittance",
    "customer", "vendor", "supplier",
]

def norm(s: str) -> str:
    return (s or "").lower().strip()

def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / max(1, area)

def cosine(a, b):
    # `sentence_transformers` outputs normalized vectors if requested
    # but fallback above returns zeros; guard against it.
    denom = (sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5)
    if denom == 0: return 0.0
    return sum(x*y for x,y in zip(a,b)) / denom

def group_lines(tokens: List[Dict[str, Any]], y_tol=10):
    """Group tokens into horizontal lines (simple y-centroid clustering)."""
    lines: Dict[int, List[Dict[str, Any]]] = {}
    cur = []
    last_y = None
    for t in sorted(tokens, key=lambda z: (z["page"], (z["y0"]+z["y1"])/2, z["x0"])):
        ymid = (t["y0"] + t["y1"]) / 2
        if last_y is None or abs(ymid - last_y) <= y_tol:
            cur.append(t); last_y = ymid
        else:
            lines.setdefault(cur[0]["page"], []).append(cur)
            cur = [t]; last_y = ymid
    if cur:
        lines.setdefault(cur[0]["page"], []).append(cur)
    return lines

def concat_text(span) -> str:
    return " ".join((t.get("text") or "").strip() for t in span).strip()

def span_rect(span) -> Tuple[int,int,int,int]:
    x0=min(t["x0"] for t in span); y0=min(t["y0"] for t in span)
    x1=max(t["x1"] for t in span); y1=max(t["y1"] for t in span)
    return x0,y0,x1,y1

def detect_anchors(lines) -> Dict[int, List[Tuple[str, Tuple[int,int,int,int]]]]:
    """Find semantic anchors per page and return [(name, rect), ...]."""
    out: Dict[int,List[Tuple[str,Tuple[int,int,int,int]]]] = {}
    if _embed_model is None:
        return out  # no embeddings, skip
    emb_anchors = [(p, emb(p)) for p in ANCHOR_PHRASES]
    for pg, lns in lines.items():
        for ln in lns:
            txt = concat_text(ln)
            if not txt: continue
            e = emb(txt)
            best = max(( (p, cosine(e,ea)) for p,ea in emb_anchors ), key=lambda x:x[1], default=None)
            if best and best[1] >= 0.6:
                r = span_rect(ln)
                out.setdefault(pg, []).append((best[0], r))
    return out

def panel_boxes(anchors: Dict[int, List[Tuple[str,Tuple[int,int,int,int]]]], lines, pad=250):
    """Expand each anchor to a panel box (rect area right/below the anchor)."""
    panels: Dict[int, List[Tuple[str, Tuple[int,int,int,int]]]] = {}
    for pg, items in anchors.items():
        for name, (x0,y0,x1,y1) in items:
            X0, Y0 = x0, y1
            X1 = max(x1, max((max(t["x1"] for t in tok) for tok in lines.get(pg, [])), default=x1))
            Y1 = Y0 + pad
            panels.setdefault(pg, []).append( (name, (X0, Y0, X1, Y1)) )
    return panels

def type_prefilter(tokens, key:str, value:str|None):
    ks = norm(key)
    cands = tokens
    if any(k in ks for k in ["zip", "postal"]):
        cands = [t for t in tokens if _re_zip.search((t.get("text") or ""))]
    elif any(k in ks for k in ["amount","total","balance","price","subtotal","tax"]):
        cands = [t for t in tokens if _re_money.search((t.get("text") or ""))]
    elif any(k in ks for k in ["date"]):
        cands = [t for t in tokens if _re_date.search((t.get("text") or ""))]
    return cands

def inside(rect, panel, margin=30):
    x0,y0,x1,y1 = rect
    px0,py0,px1,py1 = panel
    return (x0>=px0-margin and y0>=py0-margin and x1<=px1+margin and y1<=py1+margin)

# ---------- main matcher ----------
@router.post("/kv", response_model=KVResp)
def match_kv(req: KVReq):
    boxes = load_boxes(req.doc_id)

    # per-page token list
    if req.page:
        tokens = [t for t in boxes if t.get("page")==req.page]
    else:
        tokens = boxes

    # build lines + anchors/panels
    lines = group_lines(tokens)
    anchors = detect_anchors(lines)
    panels = panel_boxes(anchors, lines)

    # candidate tokens (prefilter by type)
    cands = type_prefilter(tokens, req.key, req.value)

    # if we can guess a panel from label text, keep tokens inside that panel
    panel_name = None
    chosen_panel_rect = None
    if _embed_model and req.label:
        pv = emb(norm(req.label))
        # score label vs canonical panel names
        label2panel = [
            ("billing", emb("billing")), ("ship to", emb("ship to")),
            ("remit", emb("remit to")),  ("customer", emb("customer"))
        ]
        best_panel = max(label2panel, key=lambda x: cosine(pv, x[1]))
        if cosine(pv, best_panel[1]) >= 0.45:   # soft threshold
            want = best_panel[0]
            panel_name = want
            # find any page panel whose name cosine~want
            for pg, pans in panels.items():
                for name, rect in pans:
                    if cosine(emb(name), emb(want)) >= 0.7:
                        # stick to same page if page hint present
                        if req.page and pg != req.page: 
                            continue
                        chosen_panel_rect = rect
                        break

    if chosen_panel_rect:
        cands = [t for t in cands if inside((t["x0"],t["y0"],t["x1"],t["y1"]), chosen_panel_rect, margin=40)]
        if not cands:
            # fallback: use all tokens on that page
            pass

    # make simple spans = 1..5 adjacent tokens on the same line
    spans: List[Tuple[int,List[Dict[str,Any]]]] = []  # (page, [tokens])
    for pg, lns in group_lines(cands).items():
        for ln in lns:
            n = len(ln)
            for i in range(n):
                acc = []
                for w in range(5):
                    if i+w < n:
                        acc.append(ln[i+w])
                        spans.append((pg, acc.copy()))

    # score each span
    def val_sim(a: str, b: str) -> float:
        if not a or not b: return 0.0
        if _embed_model is None: return 0.0
        return cosine(emb(norm(a)), emb(norm(b)))

    best = None
    for pg, span in spans:
        text = concat_text(span)
        if not text: 
            continue
        # base score from value similarity (if we have a value)
        s = 0.45 * val_sim(req.value or "", text)

        # type prior
        if any(k in norm(req.key) for k in ["zip","postal"]) and _re_zip.search(text): s += 0.25
        if any(k in norm(req.key) for k in ["amount","total","balance","subtotal","tax"]) and _re_money.search(text): s += 0.20
        if "date" in norm(req.key) and _re_date.search(text): s += 0.15

        # panel proximity bonus
        if chosen_panel_rect:
            xr,yr,Xr,Yr = span_rect(span)
            px0,py0,px1,py1 = chosen_panel_rect
            cx = max(px0, min((xr+Xr)/2, px1))
            cy = max(py0, min((yr+Yr)/2, py1))
            dx = abs((xr+Xr)/2 - cx); dy = abs((yr+Yr)/2 - cy)
            dist = math.hypot(dx,dy)
            s += max(0.0, 0.20 - min(0.20, dist/600.0))  # within ~600px gets up to +0.2

        if not best or s > best[0]:
            best = (s, pg, span, text)

    if not best:
        # final fallback: pick first zip/money/date, else first token
        if cands:
            span = [cands[0]]
            pg = cands[0]["page"]
            text = concat_text(span)
            score = 0.01
        else:
            raise HTTPException(404, "No candidates found")
    else:
        score, pg, span, text = best

    x0,y0,x1,y1 = span_rect(span)
    return KVResp(
        rects=[Rect(page=pg, x0=int(x0), y0=int(y0), x1=int(x1), y1=int(y1))],
        text=text,
        score=float(score),
        panel=panel_name,
    )