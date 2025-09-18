def _load_layout():
    if _loaded["layout"] is not None:
        return _loaded["layout"]
    if AutoProcessor is None or LayoutLMv3Model is None:
        return None

    root = _first_dir(CANDIDATES_LAYOUT)
    if root is None:
        return None  # <- safeguard

    try:
        proc = AutoProcessor.from_pretrained(str(root), local_files_only=True)
        mdl = LayoutLMv3Model.from_pretrained(str(root), local_files_only=True)
    except Exception as e:
        print("[lasso_locate] Failed to load LayoutLMv3:", e)
        return None

    dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    mdl.to(dev).eval()
    _loaded["layout"] = {"proc": proc, "mdl": mdl, "device": dev}
    return _loaded["layout"]
