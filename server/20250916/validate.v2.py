#!/usr/bin/env python3
# validate_portal.py  — self-contained validation portal with OCR fallback,
#                       and warmup/timeout-based auto-disable of heavy models
from __future__ import annotations
import argparse, json, io, base64, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional local models (no network)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoProcessor, LayoutLMv3Model
except Exception:
    torch = None
    AutoTokenizer = AutoModel = AutoProcessor = LayoutLMv3Model = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# >>> WARMUP/TIMEOUT: global switches that can be turned off after warmup
DISABLE = {"minilm": False, "distilbert": False, "layoutlmv3": False}

# -------------------- basic utils --------------------
def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().replace("\u00A0"," ").split())

def norm_num(s: str) -> str:
    return " ".join((s or "").lower().replace(",","").replace("$"," ").split())

def qratio(a: str, b: str) -> float:
    return float(_rfuzz.QRatio(a, b)) / 100.0

def iou(a: Dict[str,float], b: Dict[str,float]) -> float:
    ax0, ay0, ax1, ay1 = min(a["x0"],a["x1"]), min(a["y0"],a["y1"]), max(a["x0"],a["x1"]), max(a["y0"],a["y1"])
    bx0, by0, bx1, by1 = min(b["x0"],b["x1"]), min(b["y0"],b["y1"]), max(b["x0"],b["x1"]), max(b["y0"],b["y1"])
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, inter_x1 - inter_x0), max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0

def union_rect(span):
    return {
        "x0": float(min(t["x0"] for t in span)),
        "y0": float(min(t["y0"] for t in span)),
        "x1": float(max(t["x1"] for t in span)),
        "y1": float(max(t["y1"] for t in span)),
    }

def context_snippet(tokens_page, span, px_margin=120, py_margin=35):
    R = union_rect(span)
    cx0 = R["x0"] - px_margin; cy0 = R["y0"] - py_margin
    cx1 = R["x1"] + px_margin; cy1 = R["y1"] + py_margin
    bag = [t for t in tokens_page if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return norm(" ".join(t.get("text","") for t in bag if t.get("text")))

def slide_windows(tokens_page, max_w=12):
    n = len(tokens_page)
    for i in range(n):
        acc = []
        for w in range(max_w):
            j = i + w
            if j >= n: break
            txt = (tokens_page[j].get("text") or "").strip()
            if not txt: continue
            acc.append(tokens_page[j])
            yield acc

def text_from_rect(tokens_page, rect) -> str:
    x0, y0, x1, y1 = min(rect["x0"],rect["x1"]), min(rect["y0"],rect["y1"]), max(rect["x0"],rect["x1"]), max(rect["y0"],rect["y1"])
    bag = [t for t in tokens_page if not (t["x1"] < x0 or t["x0"] > x1 or t["y1"] < y0 or t["y0"] > y1)]
    bag.sort(key=lambda r: (r["y0"], r["x0"]))
    return norm(" ".join(t.get("text","") for t in bag if t.get("text")))

def crop_ocr(pdf_path: Path, page_no: int, rect, dpi=260, lang="eng") -> str:
    import pypdfium2 as pdfium
    import pytesseract
    pdf = pdfium.PdfDocument(str(pdf_path))
    if page_no < 1 or page_no > len(pdf): return ""
    pil = pdf[page_no-1].render(scale=(dpi/72)).to_pil()
    gray = ImageOps.autocontrast(pil.convert("L"))
    x0 = int(min(rect["x0"], rect["x1"])); y0 = int(min(rect["y0"], rect["y1"]))
    x1 = int(max(rect["x0"], rect["x1"])); y1 = int(max(rect["y0"], rect["y1"]))
    x0 = max(0, x0-4); y0 = max(0, y0-4)
    x1 = min(gray.width-1, x1+4); y1 = min(gray.height-1, y1+4)
    crop = gray.crop((x0,y0,x1,y1))
    if crop.width < 140 or crop.height < 40:
        scale = 3 if max(crop.width,crop.height) < 60 else 2
        crop = crop.resize((crop.width*scale, crop.height*scale), Image.BICUBIC)
    def ocr(psm:int):
        cfg = f"--oem 1 --psm {psm} -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(crop, lang=lang, config=cfg).strip()
    cands = [(len(s), s) for s in (ocr(6), ocr(7), ocr(11))]
    return (sorted(cands, reverse=True)[0][1] if cands else "").strip()

def b64_png(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# -------------------- OCR fallback --------------------
def ensure_ocr_for_pdf(pdf_path: Path, dpi=260, lang="eng") -> Tuple[Path, Path, Path]:
    """
    Ensure <stem>.meta.json, <stem>.boxes.json, <stem>.pages.txt exist.
    If missing, run OCR locally and write them, plus <stem>.ocr marker.
    Returns (meta_path, boxes_path, pages_path).
    """
    import pypdfium2 as pdfium
    import pytesseract

    stem = pdf_path.with_suffix("")
    meta_p  = stem.with_suffix(".meta.json")
    boxes_p = stem.with_suffix(".boxes.json")
    pages_p = stem.with_suffix(".pages.txt")
    marker  = stem.with_suffix(".ocr")

    if meta_p.exists() and boxes_p.exists():
        return meta_p, boxes_p, pages_p

    pdf = pdfium.PdfDocument(str(pdf_path))
    all_boxes: List[Dict[str,Any]] = []
    pages_meta: List[Dict[str,float]] = []
    all_page_texts: List[str] = []

    for i in range(len(pdf)):
        page_no = i + 1
        pil = pdf[i].render(scale=(dpi/72)).to_pil()
        gray = ImageOps.autocontrast(pil.convert("L"))
        cfg = f"--oem 1 --psm 6"  # token-level boxes
        d = pytesseract.image_to_data(gray, lang=lang, config=cfg, output_type=pytesseract.Output.DICT)
        tokens_this_page: List[str] = []
        for j in range(len(d["text"])):
            txt = (d["text"][j] or "").strip()
            if not txt: continue
            x, y, w, h = d["left"][j], d["top"][j], d["width"][j], d["height"][j]
            all_boxes.append({"page":page_no,"x0":float(x),"y0":float(y),"x1":float(x+w),"y1":float(y+h),"text":txt})
            tokens_this_page.append(txt)
        pages_meta.append({"page":page_no,"width":float(gray.width),"height":float(gray.height)})
        all_page_texts.append(" ".join(tokens_this_page))

    meta_p.write_text(json.dumps({
        "pages": pages_meta,
        "params": {"dpi": dpi, "psm": 6, "oem": 1, "lang": lang},
        "coord_space": {"origin": "top-left", "units": "px@dpi", "dpi": dpi}
    }, indent=2), encoding="utf-8")
    boxes_p.write_text(json.dumps(all_boxes), encoding="utf-8")
    pages_p.write_text("\n\n".join(all_page_texts), encoding="utf-8")
    marker.write_text(json.dumps({"ts": int(time.time()), "dpi": dpi, "lang": lang}), encoding="utf-8")

    print(f"[ocr] wrote {meta_p.name}, {boxes_p.name}, {pages_p.name}, {marker.name}")
    return meta_p, boxes_p, pages_p

# -------------------- model loading --------------------
@dataclass
class DistilLocal:
    tok: Any
    mdl: Any
    device: str

def first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists() and p.is_dir(): return p
    return None

def load_minilm(models_root: Path):
    if SentenceTransformer is None: return None
    p = first_existing([
        models_root / "sentence-transformers" / "all-MiniLM-L6-v2",
        models_root / "sentence-transformers__all-MiniLM-L6-v2",
        models_root / "all-MiniLM-L6-v2",
        models_root / "MiniLML6-v2",
    ])
    if not p: return None
    print(f"[model] MiniLM: {p}")
    return SentenceTransformer(str(p))

def load_distilbert(models_root: Path) -> Optional[DistilLocal]:
    if AutoTokenizer is None or AutoModel is None: return None
    p = first_existing([
        models_root / "distilbert-base-uncased",
        models_root / "DistilBERT" / "distilbert-base-uncased",
    ])
    if not p: return None
    print(f"[model] DistilBERT: {p}")
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    return DistilLocal(tok, mdl, dev)

def load_layoutlmv3(models_root: Path):
    if AutoProcessor is None or LayoutLMv3Model is None: return None
    p = first_existing([
        models_root / "microsoft" / "layoutlmv3-base",
        models_root / "microsoft__layoutlmv3-base",
        models_root / "layoutlmv3-base",
    ])
    if not p: return None
    print(f"[model] LayoutLMv3: {p}")
    proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
    mdl  = LayoutLMv3Model.from_pretrained(str(p), local_files_only=True)
    dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    return {"proc": proc, "mdl": mdl, "device": dev}

def embed_minilm(model, texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def embed_distil(local: DistilLocal, texts):
    with torch.no_grad():
        t = local.tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(local.device)
        out = local.mdl(**t).last_hidden_state
        mask = t["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1)
        emb = torch.nn.functional.normalize(summed / counts, dim=1)
        return emb.cpu().numpy()

# -------------------- data loading --------------------
def tfidf_fit(text: str) -> TfidfVectorizer:
    vec = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
    try: vec.fit([text])
    except ValueError: vec.fit(["placeholder"])
    return vec

def load_tokens(meta_path: Path, boxes_path: Path):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))
    by_page: Dict[int, List[Dict[str,Any]]] = {}
    for b in boxes:
        pg = int(b["page"])
        by_page.setdefault(pg, []).append(b)
    for arr in by_page.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))
    return meta, by_page

def load_extraction(stem: Path) -> Optional[List[Tuple[str,str,Optional[List[Dict[str,Any]]]]]]:
    p = stem.with_suffix(".json")
    if not p.exists(): return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    rows = []
    def norm_rects(x):
        if not x: return None
        arr = x if isinstance(x,list) else [x]
        out = []
        for b in arr:
            try:
                page = int(b.get("page"))
                if "w" in b and "h" in b and "x" in b and "y" in b:
                    out.append({"page": page, "x0": float(b["x"]), "y0": float(b["y"]),
                                "x1": float(b["x"])+float(b["w"]), "y1": float(b["y"])+float(b["h"])})
                else:
                    out.append({"page": page, "x0": float(b["x0"]), "y0": float(b["y0"]),
                                "x1": float(b["x1"]), "y1": float(b["y1"])})
            except Exception:
                pass
        return out or None
    if isinstance(obj, dict) and "fields" in obj and isinstance(obj["fields"], list):
        for f in obj["fields"]:
            key = f.get("key") or f.get("name")
            val = f.get("value")
            rects = f.get("rects") or f.get("bboxes") or f.get("bbox")
            if key and val is not None:
                rows.append((str(key), str(val), norm_rects(rects)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and "value" in v:
                rows.append((k, str(v.get("value","")), norm_rects(v.get("rects") or v.get("bboxes") or v.get("bbox"))))
            elif isinstance(v, (str,int,float)):
                rows.append((k, str(v), None))
    return rows

# -------------------- matchers --------------------
def fuzzy_score(val_text: str, span_text: str, span):
    s = qratio(val_text, span_text)
    ys = sorted([(t["y0"]+t["y1"])*0.5 for t in span])
    spread = (ys[-1]-ys[0]) if len(ys)>1 else 0.0
    avg_h = np.mean([t["y1"]-t["y0"] for t in span]) if span else 1.0
    penalty = max(0.0, (spread - 0.6*avg_h)) / max(1.0, avg_h)
    return float(max(0.0, s - 0.12*penalty))

# >>> WARMUP/TIMEOUT: best_boxes now times heavy models and disables after warmup if needed
def best_boxes(tokens_by_page,
               key, value, max_window,
               minilm_model, distil_local, lv3, tfidf_cache,
               warmup: bool, timeout_sec: float) -> dict:
    """
    Returns dict of method -> hit or None.
    On warmup docs, measures per-method walltime; if a heavy model exceeds timeout or errors,
    mark DISABLE[method]=True so subsequent docs skip it.
    """
    import time as _time

    val_norm = norm(value)
    combo = f"{norm(key)} {val_norm}".strip()

    def pick(scored):
        if not scored: return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    out = {"autolocate": None, "tfidf": None, "minilm": None, "distilbert": None, "layoutlmv3": None}

    # ---- fast (always-on) methods
    for method in ("autolocate", "tfidf"):
        scored = []
        for pg, toks in tokens_by_page.items():
            vec = tfidf_cache[pg]
            # cap spans per page for speed
            span_count = 0
            for span in slide_windows(toks, max_w=max_window):
                span_count += 1
                if span_count > 1500:  # guardrail
                    break
                rect = union_rect(span)
                stext = norm(" ".join((t.get("text") or "") for t in span))
                ctx = context_snippet(toks, span)

                if method == "autolocate":
                    s = fuzzy_score(val_norm, stext, span)
                else:
                    s_span = float(np.clip(cosine_similarity(vec.transform([val_norm]), vec.transform([stext]))[0,0], 0, 1))
                    s_ctx  = float(np.clip(cosine_similarity(vec.transform([val_norm]), vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                    s_combo= float(np.clip(cosine_similarity(vec.transform([combo]),   vec.transform([ctx]))[0,0],   0, 1)) if ctx else 0.0
                    v_toks = val_norm.split()
                    coverage = 0.0
                    if v_toks:
                        covered = 0
                        s_words = stext.split()
                        for w in v_toks:
                            covered += any((_rfuzz.QRatio(w, sw) >= 90) for sw in s_words)
                        coverage = covered / max(1, len(v_toks))
                    s = 0.70*s_span + 0.20*max(s_ctx, s_combo) + 0.10*coverage

                # line penalty
                ys = sorted([(t["y0"]+t["y1"])*0.5 for t in span])
                spread = (ys[-1]-ys[0]) if len(ys)>1 else 0.0
                avg_h = np.mean([t["y1"]-t["y0"] for t in span]) if span else 1.0
                penalty = max(0.0, (spread - 0.6*avg_h)) / max(1.0, avg_h)
                s = float(max(0.0, s - 0.12*penalty))
                scored.append({"page": pg, "rect": rect, "score": s})
        out[method] = pick(scored)

    # ---- heavy models (may be disabled)
    heavy_specs = [
        ("minilm",     minilm_model is not None and not DISABLE["minilm"]),
        ("distilbert", distil_local is not None and not DISABLE["distilbert"]),
        ("layoutlmv3", lv3 is not None and not DISABLE["layoutlmv3"]),
    ]
    for method, can_run in heavy_specs:
        if not can_run:
            continue
        t0 = _time.time()
        ok = True
        scored = []
        try:
            for pg, toks in tokens_by_page.items():
                vec = tfidf_cache[pg]
                span_count = 0
                for span in slide_windows(toks, max_w=max_window):
                    span_count += 1
                    if span_count > 1000:  # guardrail
                        break
                    rect = union_rect(span)
                    ctx = context_snippet(toks, span)

                    if method == "minilm":
                        E = embed_minilm(minilm_model, [combo, ctx])
                        s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    elif method == "distilbert":
                        with torch.no_grad():
                            t = distil_local.tok([combo, ctx], padding=True, truncation=True,
                                                 return_tensors="pt", max_length=256).to(distil_local.device)
                            out_h = distil_local.mdl(**t).last_hidden_state
                            mask = t["attention_mask"].unsqueeze(-1)
                            summed = (out_h * mask).sum(1)
                            counts = mask.sum(1).clamp(min=1)
                            emb = torch.nn.functional.normalize(summed / counts, dim=1)
                            s = float((emb[0] @ emb[1].T).item())
                    else:  # layoutlmv3 proxy score using TF-IDF + local key proximity
                        base = float(np.clip(cosine_similarity(vec.transform([val_norm]), vec.transform([ctx]))[0,0], 0, 1))
                        near = 0
                        x0 = rect["x0"]-80; y0 = rect["y0"]-40; x1 = rect["x1"]+80; y1 = rect["y1"]+40
                        kwords = [w for w in norm(key).split() if len(w)>=2]
                        for tkn in toks:
                            if tkn["x1"] < x0 or tkn["x0"] > x1 or tkn["y1"] < y0 or tkn["y0"] > y1:
                                continue
                            tx = norm(tkn.get("text") or "")
                            if any(w in tx for w in kwords): near += 1
                        s = float(min(1.0, base + 0.05*min(near, 6)))

                    # line penalty
                    ys = sorted([(t["y0"]+t["y1"])*0.5 for t in span])
                    spread = (ys[-1]-ys[0]) if len(ys)>1 else 0.0
                    avg_h = np.mean([t["y1"]-t["y0"] for t in span]) if span else 1.0
                    penalty = max(0.0, (spread - 0.6*avg_h)) / max(1.0, avg_h)
                    s = float(max(0.0, s - 0.12*penalty))
                    scored.append({"page": pg, "rect": rect, "score": s})

                    # soft time budget during warmup
                    if warmup and (_time.time() - t0) > timeout_sec:
                        ok = False
                        raise TimeoutError(f"{method} exceeded {timeout_sec}s on warmup")
        except Exception as exc:
            ok = False
            if warmup:
                print(f"[warn] disabling {method}: {exc}")
                DISABLE[method] = True

        out[method] = pick(scored) if ok else None

    return out

# -------------------- HTML (compact CSS/JS) --------------------
CSS = """/* styles trimmed for brevity */ 
body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
.app { display:grid; grid-template-columns: 320px 1fr; height: 100vh; }
.left { border-right: 1px solid #eee; overflow:auto; }
.right { overflow:auto; padding: 12px 16px; }
.header { padding:10px 12px; border-bottom:1px solid #eee; font-weight:700; }
.doc { padding:10px 12px; cursor:pointer; border-bottom:1px solid #f3f3f3; }
.doc:hover { background:#fafafa; }
.doc .name { font-weight:600; font-size:13px; }
.doc .mini { font-size:12px; color:#666; margin-top:4px; display:flex; gap:6px; flex-wrap:wrap }
.pill { padding:2px 8px; border-radius:999px; background:#f2f2f2; }
.kpi { display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 16px; }
.card { border:1px solid #eee; border-radius:8px; padding:8px 10px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.grid { display:grid; grid-template-columns: 380px 1fr; gap:16px; align-items:start; }
.thumb { border:1px solid #eee; border-radius:8px; overflow:hidden; position:relative; width: 360px; background:#fff; }
.thumb img { display:block; width:100%; }
.canvas { position:absolute; inset:0; }
.legend { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 12px; align-items:center; }
.sw { width:12px; height:12px; border-radius:2px; display:inline-block; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid #eee; padding: 6px 8px; vertical-align: top; }
th { text-align:left; background:#fafafa; position: sticky; top: 0; z-index: 1; }
.small { font-size:12px; color:#666; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
select, button { padding:6px 8px; border:1px solid #ddd; border-radius:6px; background:#fff; }
.topbar { display:flex; gap:12px; align-items:center; margin:8px 0 14px; }
.ok { color:#0a7d2e; font-weight:600; }
.warn { color:#b35c00; font-weight:600; }
.bad { color:#b00020; font-weight:600; }
"""

JS = """/* minimal portal logic */ 
const COLORS = { autolocate:'#ec4899', tfidf:'#f59e0b', minilm:'#10b981', distilbert:'#3b82f6', layoutlmv3:'#a855f7' };
let DATA = null; let STATE = { docIdx: -1, fieldIdx: -1 };
function pct(n,d){ return d ? (100*n/d).toFixed(1)+'%' : '—'; }
function init(){ DATA = window.__DATA__; renderDocList(); renderSummary(); }
function renderDocList(){
  const left = document.getElementById('left'); left.innerHTML=''; 
  const hdr=document.createElement('div'); hdr.className='header'; hdr.textContent='Documents'; left.appendChild(hdr);
  DATA.docs.forEach((d,i)=>{ const div=document.createElement('div'); div.className='doc'; div.onclick=()=>{STATE.docIdx=i;STATE.fieldIdx=-1;renderDoc();};
    const name=document.createElement('div'); name.className='name'; name.textContent=d.name;
    const mini=document.createElement('div'); mini.className='mini';
    for (const k of Object.keys(d.stats)){ const pill=document.createElement('span'); pill.className='pill'; pill.textContent=`${k}: ${d.stats[k]}`; mini.appendChild(pill); }
    div.appendChild(name); div.appendChild(mini); left.appendChild(div); });
}
function renderSummary(){
  const right=document.getElementById('right'); right.innerHTML=''; const title=document.createElement('div'); title.className='header'; title.textContent='Summary'; right.appendChild(title);
  const kpi=document.createElement('div'); kpi.className='kpi';
  for (const [k,v] of Object.entries(DATA.global.kpi)){ const c=document.createElement('div'); c.className='card';
    const t=document.createElement('div'); t.className='small'; t.textContent=k;
    const n=document.createElement('div'); n.style.fontSize='18px'; n.style.fontWeight='700'; n.textContent=v; c.appendChild(t); c.appendChild(n); kpi.appendChild(c); }
  right.appendChild(kpi);
  const tbl=document.createElement('table'); tbl.innerHTML=`<thead><tr><th>Method</th><th>Text ≥ 0.90</th><th>Numeric ≥ 0.98</th><th>IoU ≥ 0.5</th><th>Count</th></tr></thead><tbody></tbody>`;
  const tb=tbl.querySelector('tbody');
  for (const m of DATA.global.methods){ const r=DATA.global.by_method[m]||{text:0,num:0,iou:0,iou_den:0,count:0};
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><span class="pill" style="background:#f6f6f6;border-left:6px solid ${COLORS[m]};padding-left:6px">${m}</span></td>
      <td>${pct(r.text,r.count)}</td><td>${pct(r.num,r.count)}</td><td>${r.iou_den?pct(r.iou,r.iou_den):'—'}</td><td>${r.count}</td>`;
    tb.appendChild(tr); }
  right.appendChild(tbl);
}
function renderDoc(){
  const d=DATA.docs[STATE.docIdx]; const right=document.getElementById('right'); right.innerHTML='';
  const title=document.createElement('div'); title.className='header'; title.textContent=d.name; right.appendChild(title);
  const top=document.createElement('div'); top.className='topbar';
  const sel=document.createElement('select'); sel.onchange=()=>{STATE.fieldIdx=parseInt(sel.value,10); drawField();};
  d.fields.forEach((f,idx)=>{ const opt=document.createElement('option'); opt.value=idx; opt.textContent=f.key; sel.appendChild(opt);});
  if (STATE.fieldIdx<0) STATE.fieldIdx=0; sel.value=String(STATE.fieldIdx); top.appendChild(sel);
  const legend=document.createElement('div'); legend.className='legend';
  for (const [m,color] of Object.entries(COLORS)){ const sw=document.createElement('span'); sw.className='sw'; sw.style.background=color;
    const lab=document.createElement('span'); lab.className='small'; lab.textContent=m; legend.appendChild(sw); legend.appendChild(lab);}
  top.appendChild(legend); right.appendChild(top);
  const grid=document.createElement('div'); grid.className='grid'; const leftBox=document.createElement('div');
  const thumbWrap=document.createElement('div'); thumbWrap.className='thumb';
  const img=document.createElement('img'); img.id='pageImg'; thumbWrap.appendChild(img);
  const overlay=document.createElement('canvas'); overlay.id='ov'; overlay.className='canvas'; thumbWrap.appendChild(overlay);
  leftBox.appendChild(thumbWrap); right.appendChild(grid); grid.appendChild(leftBox);
  const tbl=document.createElement('table'); tbl.id='detailTable';
  tbl.innerHTML=`<thead><tr><th>Method</th><th>Score</th><th>Text (tokens)</th><th>Crop OCR</th><th>Text Sim</th><th>Numeric Sim</th><th>IoU</th></tr></thead><tbody></tbody>`;
  grid.appendChild(tbl);
  window.__pageThumbs=d.page_thumbs; window.__doc=d; drawField();
}
function drawField(){
  const d=window.__doc; const f=d.fields[STATE.fieldIdx]; const methods=['autolocate','tfidf','minilm','distilbert','layoutlmv3'];
  const pages=methods.map(m=>(f[m]&&f[m].page!=null)?f[m].page:null).filter(x=>x!=null); if (pages.length===0) pages.push(1);
  const page=pages.sort((a,b)=> pages.filter(v=>v===a).length - pages.filter(v=>v===b).length).pop();
  const img=document.getElementById('pageImg'); const ov=document.getElementById('ov');
  const thumb=window.__pageThumbs[String(page)] || window.__pageThumbs['1']; img.src=thumb||'';
  img.onload=()=>{ ov.width=img.clientWidth; ov.height=img.clientHeight; const ctx=ov.getContext('2d'); ctx.clearRect(0,0,ov.width,ov.height);
    methods.forEach(m=>{ const h=f[m]; if (!h || h.page!==page || !h.rect) return; const r=h.rect;
      const sw=ov.width / d.meta[String(page)].w; const sh=ov.height / d.meta[String(page)].h;
      const x0=Math.min(r.x0,r.x1)*sw, y0=Math.min(r.y0,r.y1)*sh, x1=Math.max(r.x0,r.x1)*sw, y1=Math.max(r.y0,r.y1)*sh;
      ctx.strokeStyle=COLORS[m]; ctx.lineWidth=2; ctx.fillStyle=COLORS[m]+'33'; ctx.beginPath(); ctx.rect(x0,y0,x1-x0,y1-y0); ctx.fill(); ctx.stroke();
      ctx.fillStyle=COLORS[m]; ctx.font='12px ui-sans-serif'; ctx.fillRect(x0, Math.max(0,y0-16), 8, 16);
      ctx.fillStyle='#000'; ctx.fillText(m, x0+12, Math.max(12,y0-4)); });
    const tb=document.querySelector('#detailTable tbody'); tb.innerHTML='';
    methods.forEach(m=>{ const h=f[m]||{}; const iou=(h.iou==null)?'—':h.iou.toFixed(2); const ts=(h.text_sim==null)?'—':h.text_sim.toFixed(2); const ns=(h.num_sim==null)?'—':h.num_sim.toFixed(2);
      const tr=document.createElement('tr');
      tr.innerHTML=`<td><span class="pill" style="background:#f6f6f6;border-left:6px solid ${COLORS[m]};padding-left:6px">${m}</span></td>
        <td>${h.score==null?'':h.score.toFixed(3)}</td>
        <td class="mono">${(h.text_pred||'')}</td>
        <td class="mono">${(h.text_ocr||'')}</td>
        <td class="${h.text_sim>=0.90?'ok':(h.text_sim>=0.75?'warn':'bad')}">${ts}</td>
        <td class="${h.num_sim>=0.98?'ok':(h.num_sim>=0.90?'warn':'bad')}">${ns}</td>
        <td class="${(h.iou||0)>=0.5?'ok':((h.iou||0)>=0.3?'warn':'bad')}">${iou}</td>`;
      tb.appendChild(tr); }); };
}
window.addEventListener('DOMContentLoaded', init);
"""

def render_html(out_path: Path, payload: dict):
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Validation Portal</title><style>{CSS}</style></head>
<body><div class="app"><div id="left" class="left"></div><div id="right" class="right"></div></div>
<script>window.__DATA__ = {json.dumps(payload)}; {JS}</script></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out_path}")

# -------------------- pipeline --------------------
def build_portal(root: Path, models_root: Path, out: Path, max_window: int, dpi: int, lang: str, thumbs: bool,
                 # >>> WARMUP/TIMEOUT: new params
                 warmup_docs: int = 1, model_timeout: float = 20.0):
    # models
    minilm = load_minilm(models_root)
    distil = load_distilbert(models_root)
    lv3    = load_layoutlmv3(models_root)

    pdfs = [p for p in root.rglob("*.pdf") if p.is_file()]
    print(f"Found {len(pdfs)} PDFs under {root}")

    docs_payload = []
    global_rows = []

    for pdf in pdfs:
        # Ensure OCR artifacts exist (auto-OCR if missing)
        meta_p, boxes_p, _pages_p = ensure_ocr_for_pdf(pdf, dpi=dpi, lang=lang)

        fields = load_extraction(pdf.with_suffix(""))
        meta, tokens_by_page = load_tokens(meta_p, boxes_p)

        # per-page TF-IDF
        tfidf_cache = {}
        for pg, toks in tokens_by_page.items():
            txt = " ".join((t.get("text") or "").strip() for t in toks)
            tfidf_cache[pg] = tfidf_fit(txt)

        # thumbs
        page_thumbs = {}
        if thumbs:
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(str(pdf))
                for i in range(len(doc)):
                    im = doc[i].render(scale=(140/72)).to_pil()
                    page_thumbs[str(i+1)] = b64_png(im)
            except Exception:
                pass

        # >>> WARMUP/TIMEOUT: determine if this is a warmup doc
        doc_idx = len(docs_payload)        # how many docs we have emitted already
        is_warmup = (doc_idx < warmup_docs)

        field_payloads = []
        per_method_counters = {m: {"text":0,"num":0,"iou":0,"iou_den":0,"count":0} for m in
                               ["autolocate","tfidf","minilm","distilbert","layoutlmv3"]}

        if not fields:
            # still render minimal doc card
            docs_payload.append({
                "name": str(pdf.relative_to(root)),
                "page_thumbs": page_thumbs,
                "fields": [],
                "meta": {str(p["page"]): {"w": p["width"], "h": p["height"]} for p in meta["pages"]},
                "stats": {"fields":"0"}
            })
            continue

        for key, value, gt_rects in fields:
            hits = best_boxes(
                tokens_by_page,
                key, value, max_window,
                # skip heavy models if already disabled
                None if DISABLE["minilm"]     else minilm,
                None if DISABLE["distilbert"] else distil,
                None if DISABLE["layoutlmv3"] else lv3,
                tfidf_cache,
                warmup=is_warmup,
                timeout_sec=model_timeout
            )
            row = {"key": key, "gt_value": value}
            for method in ["autolocate","tfidf","minilm","distilbert","layoutlmv3"]:
                hit = hits.get(method)
                rec = None
                if hit:
                    page = hit["page"]; rect = hit["rect"]; score = hit["score"]
                    pred_text = text_from_rect(tokens_by_page[page], rect)
                    crop_text = norm(crop_ocr(pdf, page, rect, dpi=dpi, lang=lang))
                    n_val = norm(value)
                    text_sim = max(qratio(n_val, pred_text), qratio(n_val, crop_text)) if (pred_text or crop_text) else 0.0
                    num_sim  = max(qratio(norm_num(value), norm_num(pred_text)), qratio(norm_num(value), norm_num(crop_text))) if (pred_text or crop_text) else 0.0
                    best_iou = None
                    if gt_rects:
                        cands = [g for g in gt_rects if int(g.get("page",-1)) == page] or gt_rects
                        best_iou = max((iou(rect, g) for g in cands), default=0.0)
                    rec = {
                        "page": page, "rect": rect, "score": score,
                        "text_pred": pred_text, "text_ocr": crop_text,
                        "text_sim": round(text_sim,3), "num_sim": round(num_sim,3),
                        "iou": round(best_iou,3) if best_iou is not None else None
                    }
                    # counters
                    per_method_counters[method]["count"] += 1
                    if text_sim >= 0.90: per_method_counters[method]["text"] += 1
                    if num_sim  >= 0.98: per_method_counters[method]["num"]  += 1
                    if best_iou is not None:
                        per_method_counters[method]["iou_den"] += 1
                        if best_iou >= 0.5: per_method_counters[method]["iou"] += 1
                row[method] = rec
            field_payloads.append(row)

        # left-card stats
        stats = {}
        stats["fields"] = str(len(field_payloads))
        for m, c in per_method_counters.items():
            stats[m] = f"{int(100*(c['text']/c['count'])) if c['count'] else 0}% txt"

        docs_payload.append({
            "name": str(pdf.relative_to(root)),
            "page_thumbs": page_thumbs,
            "fields": field_payloads,
            "meta": {str(p["page"]): {"w": p["width"], "h": p["height"]} for p in meta["pages"]},
            "stats": stats
        })

        # >>> WARMUP/TIMEOUT: after warmup doc(s), log model status
        if is_warmup and (len(docs_payload) == warmup_docs):
            print("[info] heavy model status after warmup:",
                  {k: ("disabled" if v else "enabled") for k, v in DISABLE.items()})

        for f in field_payloads:
            for m in ["autolocate","tfidf","minilm","distilbert","layoutlmv3"]:
                h = f.get(m)
                if not h: continue
                global_rows.append({"method": m, "text_sim": h.get("text_sim") or 0.0,
                                    "num_sim": h.get("num_sim") or 0.0, "iou": h.get("iou")})

    # global KPIs
    by_m = {m: {"text":0,"num":0,"iou":0,"iou_den":0,"count":0} for m in ["autolocate","tfidf","minilm","distilbert","layoutlmv3"]}
    for r in global_rows:
        m = r["method"]; by_m[m]["count"] += 1
        if r["text_sim"] >= 0.90: by_m[m]["text"] += 1
        if r["num_sim"]  >= 0.98: by_m[m]["num"]  += 1
        if r["iou"] is not None:
            by_m[m]["iou_den"] += 1
            if r["iou"] >= 0.5: by_m[m]["iou"] += 1

    def pct(n,d): return f"{(100*n/d):.1f}%" if d else "—"
    kpi = {}
    total_docs = len(docs_payload)
    total_fields = sum(len(d["fields"]) for d in docs_payload)
    kpi["Docs"] = str(total_docs)
    kpi["Fields"] = str(total_fields)
    for m,c in by_m.items():
        kpi[f"{m} · text≥0.90"] = pct(c["text"], c["count"])
        kpi[f"{m} · num≥0.98"]  = pct(c["num"],  c["count"])
        kpi[f"{m} · IoU≥0.5"]   = pct(c["iou"],  c["iou_den"]) if c["iou_den"] else "—"

    payload = {
        "global": {"kpi": kpi, "by_method": by_m, "methods":["autolocate","tfidf","minilm","distilbert","layoutlmv3"]},
        "docs": docs_payload
    }
    render_html(out, payload)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Root folder with PDFs + (optional) *.meta.json + *.boxes.json + *.json")
    ap.add_argument("--models_root", type=Path, default=Path("src/models"))
    ap.add_argument("--out", type=Path, default=Path("validation_portal.html"))
    ap.add_argument("--max_window", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=260)
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--first_page_thumbs", action="store_true")

    # >>> WARMUP/TIMEOUT: new args
    ap.add_argument("--warmup_docs", type=int, default=1,
                    help="Try heavy models on the first N docs only; disable models that error/timeout.")
    ap.add_argument("--model_timeout", type=float, default=20.0,
                    help="Max seconds per heavy model on warmup docs before disabling it.")

    args = ap.parse_args()
    build_portal(
        args.root, args.models_root, args.out,
        args.max_window, args.dpi, args.lang,
        args.first_page_thumbs,
        # >>> WARMUP/TIMEOUT: pass down
        args.warmup_docs, args.model_timeout
    )

if __name__ == "__main__":
    main()