@router.post("/locate")
def locate(req: LocateReq):
    """
    Return best boxes from 5 methods:
      - autolocate (fuzzy by value)
      - tfidf (value + context)
      - minilm (local MiniLM if present)
      - distilbert (local DistilBERT if present)
      - layoutlmv3 (proxy score: tfidf+key-vicinity)
    Each score is adjusted with:
      + 0.15 * proximity(anchor words near this key)
      + 0.10 * type prior (regex/shape match for zip/date/amount/phone/email/etc.)
      - 0.12 * line penalty (multi-line span discourager)
    """
    # ---------- load tokens / meta ----------
    bp, mp = boxes_path(req.doc_id), meta_path(req.doc_id)
    if not bp.exists() or not mp.exists():
        raise HTTPException(404, "tokens/meta missing; upload/ocr first")

    tokens = json.loads(bp.read_text())
    meta   = json.loads(mp.read_text())
    pages_meta = {int(p["page"]): (float(p["width"]), float(p["height"])) for p in meta.get("pages", [])}

    # group tokens by page and sort reading order
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        pg = int(t["page"])
        if not t.get("text"): 
            continue
        pages.setdefault(pg, []).append(t)
    for arr in pages.values():
        arr.sort(key=lambda r: (r["y0"], r["x0"]))

    # per-page tfidf (for context similarities)
    tfidf: Dict[int, TfidfVectorizer] = {}
    for pg, toks in pages.items():
        txt = " ".join((t.get("text") or "").strip() for t in toks)
        v = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        try:
            v.fit([txt])
        except ValueError:
            v.fit(["placeholder"])
        tfidf[pg] = v

    # ---------- helpers (local closures) ----------
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().lower().replace("\u00A0", " ").split())

    def _union(span: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "x0": float(min(t["x0"] for t in span)),
            "y0": float(min(t["y0"] for t in span)),
            "x1": float(max(t["x1"] for t in span)),
            "y1": float(max(t["y1"] for t in span)),
        }

    def _context(tokens_page: List[Dict[str, Any]], span: List[Dict[str, Any]], px=120, py=35) -> str:
        R = _union(span)
        cx0, cy0, cx1, cy1 = R["x0"] - px, R["y0"] - py, R["x1"] + px, R["y1"] + py
        bag = [t for t in tokens_page if not (t["x1"] < cx0 or t["x0"] > cx1 or t["y1"] < cy0 or t["y0"] > cy1)]
        bag.sort(key=lambda r: (r["y0"], r["x0"]))
        return _norm(" ".join((t.get("text") or "") for t in bag if t.get("text")))

    def _slide(tokens_page: List[Dict[str, Any]], max_w=12):
        n = len(tokens_page)
        for i in range(n):
            acc = []
            for w in range(max_w):
                j = i + w
                if j >= n:
                    break
                txt = (tokens_page[j].get("text") or "").strip()
                if not txt:
                    continue
                acc.append(tokens_page[j])
                yield acc

    def _line_penalty(span: List[Dict[str, Any]]) -> float:
        if len(span) <= 1:
            return 0.0
        ys = sorted([(t["y0"] + t["y1"]) * 0.5 for t in span])
        spread = ys[-1] - ys[0]
        avg_h = float(np.mean([t["y1"] - t["y0"] for t in span]))
        return max(0.0, (spread - 0.6 * avg_h)) / max(1.0, avg_h)

    # --- anchors from key + tiny label lexicon (per page centers) ---
    lex = {
        "name", "address", "addr", "street", "st", "road", "rd", "city", "state", "zip", "postal",
        "phone", "fax", "email", "amount", "total", "date", "invoice", "number", "no", "policy"
    }
    key_words = [w for w in _norm(req.key).split() if len(w) >= 2]
    anchor_words = set(key_words) | lex

    page_anchors: Dict[int, List[tuple]] = {}   # pg -> [(cx, cy), ...]
    for pg, toks in pages.items():
        centers = []
        for t in toks:
            tx = _norm(t.get("text") or "")
            if not tx:
                continue
            if any(w in tx for w in anchor_words):
                cx = 0.5 * (float(t["x0"]) + float(t["x1"]))
                cy = 0.5 * (float(t["y0"]) + float(t["y1"]))
                centers.append((cx, cy))
        page_anchors[pg] = centers

    def _proximity_boost(rect: Dict[str, float], anchors: List[tuple], pg: int) -> float:
        """
        0..1 boost based on min distance to any anchor on this page.
        Normalized by page diagonal * 0.35 (so within ~35% of diagonal -> near).
        """
        if not anchors:
            return 0.0
        cx = 0.5 * (rect["x0"] + rect["x1"])
        cy = 0.5 * (rect["y0"] + rect["y1"])
        dmin = min(((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5 for (ax, ay) in anchors)
        W, H = pages_meta.get(pg, (1000.0, 1000.0))
        diag = (W ** 2 + H ** 2) ** 0.5
        # nearer -> larger boost; clamp to [0,1]
        return float(max(0.0, 1.0 - (dmin / (diag * 0.35))))

    # --- type prior 0..1 (shape/regex) ---
    import re
    re_zip    = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    re_date   = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
    re_amount = re.compile(r"^\$?\s*[-+]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$")
    re_phone  = re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    re_email  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

    def _type_score(key: str, span_text_norm: str) -> float:
        k = _norm(key)
        t = span_text_norm
        # prefer more specific first
        if any(w in k for w in ("zip", "postal")):
            return 1.0 if re_zip.search(t) else 0.0
        if any(w in k for w in ("date", "dob", "issue", "expiry", "expiration")):
            return 1.0 if re_date.search(t) else 0.0
        if any(w in k for w in ("amount", "total", "subtotal", "balance")):
            return 1.0 if re_amount.search(t) else 0.0
        if any(w in k for w in ("phone", "mobile", "fax")):
            return 1.0 if re_phone.search(t) else 0.0
        if any(w in k for w in ("email", "e-mail")):
            return 1.0 if re_email.search(t) else 0.0
        # address-ish prior: presence of numbers + street cue
        if any(w in k for w in ("address", "addr", "street", "st", "road", "rd", "city", "state")):
            cues = (" st", " rd", " ave", " dr", " ln", " blvd", " court", " ct")
            return 0.7 if any(c in t for c in cues) else (0.4 if any(ch.isdigit() for ch in t) else 0.0)
        return 0.0

    # ---------- base strings ----------
    key_n = _norm(req.key)
    val_n = _norm(req.value)
    combo = f"{key_n} {val_n}".strip()
    max_w = max(4, int(req.max_window))

    def _pick(scored: List[Dict[str, Any]]):
        if not scored:
            return None
        scored.sort(key=lambda x: x["score"], reverse=True)
        b = scored[0]
        return {"page": int(b["page"]), "rect": b["rect"], "score": float(b["score"])}

    out: Dict[str, Any] = {
        "autolocate": None,
        "tfidf": None,
        "minilm": None,
        "distilbert": None,
        "layoutlmv3": None,
    }

    # ---------- ALWAYS: autolocate + tfidf ----------
    for method in ("autolocate", "tfidf"):
        scored: List[Dict[str, Any]] = []
        for pg, toks in pages.items():
            vec = tfidf[pg]
            cnt = 0
            for span in _slide(toks, max_w):
                cnt += 1
                if cnt > 1500:
                    break
                rect = _union(span)
                stext = _norm(" ".join((t.get("text") or "") for t in span))
                ctx = _context(toks, span)

                if method == "autolocate":
                    s = float(_rfuzz.QRatio(val_n, stext)) / 100.0
                else:
                    s_span = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([stext]))[0, 0], 0, 1))
                    s_ctx = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0, 0], 0, 1)) if ctx else 0.0
                    s_comb = float(np.clip(cosine_similarity(vec.transform([combo]), vec.transform([ctx]))[0, 0], 0, 1)) if ctx else 0.0
                    s = 0.75 * s_span + 0.25 * max(s_ctx, s_comb)

                # --- NEW scoring adjustments (anchor + type + penalty)
                prox = _proximity_boost(rect, page_anchors.get(pg, []), pg)     # 0..1
                tpri = _type_score(req.key, stext)                               # 0..1
                s = s + 0.15 * prox + 0.10 * tpri - 0.12 * _line_penalty(span)
                s = max(0.0, s)

                scored.append({"page": pg, "rect": rect, "score": s})
        out[method] = _pick(scored)

    # ---------- BEST-EFFORT: MiniLM ----------
    try:
        mroot = Path(req.models_root).resolve() if req.models_root else None
        m = _load_minilm(mroot)
        if m is not None:
            scored: List[Dict[str, Any]] = []
            for pg, toks in pages.items():
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000:
                        break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    E = m.encode([combo, ctx], convert_to_numpy=True, normalize_embeddings=True)
                    s = float(np.clip(np.dot(E[0], E[1]), 0, 1))
                    prox = _proximity_boost(rect, page_anchors.get(pg, []), pg)
                    tpri = _type_score(req.key, _norm(ctx))
                    s = s + 0.15 * prox + 0.10 * tpri - 0.12 * _line_penalty(span)
                    s = max(0.0, s)
                    scored.append({"page": pg, "rect": rect, "score": s})
            out["minilm"] = _pick(scored)
    except Exception:
        out["minilm"] = None

    # ---------- BEST-EFFORT: DistilBERT ----------
    try:
        d = _load_distil(mroot if "mroot" in locals() else None)
        if d is not None and torch is not None:
            scored: List[Dict[str, Any]] = []
            with torch.no_grad():
                for pg, toks in pages.items():
                    cnt = 0
                    for span in _slide(toks, max_w):
                        cnt += 1
                        if cnt > 1000:
                            break
                        rect = _union(span)
                        ctx = _context(toks, span)
                        tok = d.tok([combo, ctx], padding=True, truncation=True, return_tensors="pt", max_length=256).to(d.dev)
                        hs = d.mdl(**tok).last_hidden_state
                        mask = tok["attention_mask"].unsqueeze(-1)
                        summed = (hs * mask).sum(1)
                        counts = mask.sum(1).clamp(min=1)
                        emb = torch.nn.functional.normalize(summed / counts, dim=1)
                        s = float((emb[0] @ emb[1].T).item())
                        prox = _proximity_boost(rect, page_anchors.get(pg, []), pg)
                        tpri = _type_score(req.key, _norm(ctx))
                        s = s + 0.15 * prox + 0.10 * tpri - 0.12 * _line_penalty(span)
                        s = max(0.0, s)
                        scored.append({"page": pg, "rect": rect, "score": s})
            out["distilbert"] = _pick(scored)
    except Exception:
        out["distilbert"] = None

    # ---------- BEST-EFFORT: LayoutLMv3 (proxy: tfidf + key vicinity) ----------
    try:
        L = _load_layout(mroot if "mroot" in locals() else None)
        if L is not None:
            scored: List[Dict[str, Any]] = []
            for pg, toks in pages.items():
                vec = tfidf[pg]
                cnt = 0
                for span in _slide(toks, max_w):
                    cnt += 1
                    if cnt > 1000:
                        break
                    rect = _union(span)
                    ctx = _context(toks, span)
                    base = float(np.clip(cosine_similarity(vec.transform([val_n]), vec.transform([ctx]))[0, 0], 0, 1))

                    # small key proximity boost via lexical hits around rect
                    near = 0
                    x0 = rect["x0"] - 80; y0 = rect["y0"] - 40; x1 = rect["x1"] + 80; y1 = rect["y1"] + 40
                    kwords = [w for w in key_n.split() if len(w) >= 2]
                    for tkn in toks:
                        if tkn["x1"] < x0 or tkn["x0"] > x1 or tkn["y1"] < y0 or tkn["y0"] > y1:
                            continue
                        tx = _norm(tkn.get("text") or "")
                        if any(w in tx for w in kwords):
                            near += 1
                    s = float(min(1.0, base + 0.05 * min(near, 6)))

                    # same scoring adjustments
                    prox = _proximity_boost(rect, page_anchors.get(pg, []), pg)
                    tpri = _type_score(req.key, _norm(ctx))
                    s = s + 0.15 * prox + 0.10 * tpri - 0.12 * _line_penalty(span)
                    s = max(0.0, s)

                    scored.append({"page": pg, "rect": rect, "score": s})
            out["layoutlmv3"] = _pick(scored)
    except Exception:
        out["layoutlmv3"] = None

    return {"hits": out, "pages": meta.get("pages", [])}