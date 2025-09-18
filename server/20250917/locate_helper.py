# --- Anchors & type rules ----------------------------------------------------

ANCHORS = {
    # key substring -> anchor phrases (lowercased)
    "customer address": ["customer address", "bill to", "sold to", "ship to", "remit to"],
    "billing address":  ["bill to", "billing address"],
    "shipping address": ["ship to", "shipping address"],
    "zip":              ["zip", "postal code", "postcode"],
    "phone":            ["phone", "tel", "telephone"],
    "date":             ["date", "issue date", "invoice date", "policy date", "effective date"],
    "invoice":          ["invoice #", "invoice no", "invoice number"],
    "policy":           ["policy #", "policy no", "policy number"],
    "amount":           ["amount", "total", "balance due", "subtotal", "grand total"],
}

STREET_WORDS = {"st","street","rd","road","ave","avenue","blvd","boulevard","ln","lane","dr","drive","ct","court","hwy","highway"}
import re
RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
RE_DATE = re.compile(r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b")
RE_PHONE = re.compile(r"\b(?:\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})\b")
RE_MONEY = re.compile(r"\b(?:\$)?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")

def _anchor_terms(key: str) -> List[str]:
    k = key.lower()
    bag = []
    for sub, lst in ANCHORS.items():
        if sub in k:
            bag += lst
    # always include bare key tokens
    bag += [w for w in re.split(r"[\s_.:/\-]+", k) if len(w) >= 3]
    # unique
    return sorted(set([t.strip().lower() for t in bag if t.strip()]))

def _find_anchor_positions(tokens: List[Dict[str,Any]], terms: List[str]) -> List[Dict[str,float]]:
    out = []
    for t in tokens:
        txt = (t.get("text") or "").lower()
        if not txt: continue
        if any(term in txt for term in terms):
            out.append({"x0":float(t["x0"]), "y0":float(t["y0"]), "x1":float(t["x1"]), "y1":float(t["y1"])})
    return out

def _center(r): return ((r["x0"]+r["x1"])*0.5, (r["y0"]+r["y1"])*0.5)

def _proximity_boost(rect, anchors, max_px=280.0):
    if not anchors: return 0.0
    cx, cy = _center(rect)
    best = 1e9
    for a in anchors:
        ax, ay = _center(a)
        d = ((cx-ax)**2 + (cy-ay)**2) ** 0.5
        if d < best: best = d
    # 0..1 where 0px->1.0, >=max_px->0
    if best >= max_px: return 0.0
    return float(1.0 - (best / max_px))

def _type_score(key: str, val_text: str) -> float:
    k = key.lower()
    v = val_text.lower()
    # +1.0 perfect; 0 neutral; negative penalty if violated
    if "zip" in k or "postal" in k:
        return 1.0 if RE_ZIP.search(v) else -0.4
    if "date" in k:
        return 1.0 if RE_DATE.search(v) else -0.3
    if "phone" in k or "tel" in k:
        return 1.0 if RE_PHONE.search(v) else -0.4
    if "amount" in k or "total" in k or "balance" in k:
        return 0.8 if RE_MONEY.search(v) else -0.2
    if "address" in k:
        # must have a number + a street word nearby
        has_num = bool(re.search(r"\b\d+\b", v))
        has_st  = any(sw in v for sw in STREET_WORDS)
        return 0.8 if (has_num and has_st) else -0.2
    return 0.0
    
    # Build anchors once per page outside the span loop
# (Place this just after you create tfidf[pg] in locate())
page_anchors = _find_anchor_positions(toks, _anchor_terms(req.key))


# existing: s = ... # your similarity score
# New: apply anchor proximity and type prior
prox = _proximity_boost(rect, page_anchors)          # 0..1
tpri = _type_score(req.key, _norm(" ".join((t.get("text") or "") for t in span)))

s = s + 0.15*prox + 0.10*tpri - 0.12*_line_penalty(span)
    