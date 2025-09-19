#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strictly-local model verifier for:
  - sentence-transformers/all-MiniLM-L6-v2
  - distilbert-base-uncased
  - microsoft/layoutlmv3-base

No internet calls. Works in Jupyter or CLI.
Outputs a JSON summary: verify_local_models.json
"""

import os, sys, json
from pathlib import Path
from typing import Optional, Dict, Any

# ---- Force STRICT offline everywhere -----------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = (ROOT / "src" / "models") if (ROOT / "src" / "models").exists() else (ROOT / "models")

# Candidates based on your tree screenshot
CAND_MINILM = [
    MODELS_ROOT / "sentence-transformers_all-MiniLM-L6-v2",
    MODELS_ROOT / "sentence-transformers__all-MiniLM-L6-v2",
    MODELS_ROOT / "all-MiniLM-L6-v2",
]
CAND_DISTIL = [
    MODELS_ROOT / "distilbert-base-uncased",
    MODELS_ROOT / "DistilBERT" / "distilbert-base-uncased",
]
CAND_LAYOUT = [
    MODELS_ROOT / "microsoft_layoutlmv3-base",
    MODELS_ROOT / "microsoft__layoutlmv3-base",
    MODELS_ROOT / "layoutlmv3-base",
]

def first_dir(paths):
    for p in paths:
        if p.exists() and p.is_dir():
            return p
    return None

def ok(msg):  print(f"[OK] {msg}")
def warn(msg):print(f"[WARN] {msg}")
def err(msg): print(f"[ERR] {msg}")

def test_minilm(summary: Dict[str, Any]) -> None:
    info = {"ok": False, "path": None, "cosine": None, "error": None}
    p = first_dir(CAND_MINILM)
    info["path"] = str(p) if p else None
    if not p:
        warn("MiniLM folder not found in candidates.")
        summary["minilm"] = info
        return
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        info["error"] = f"sentence-transformers not installed: {e}"
        err(info["error"])
        summary["minilm"] = info
        return
    try:
        model = SentenceTransformer(str(p))
        emb = model.encode(["invoice total", "amount due"],
                           normalize_embeddings=True, convert_to_numpy=True)
        cos = float((emb[0] * emb[1]).sum())
        ok(f"MiniLM loaded from {p} (cosine={cos:.3f})")
        info["ok"] = True
        info["cosine"] = cos
    except Exception as e:
        info["error"] = f"load/encode failed: {e}"
        err(info["error"])
    summary["minilm"] = info

def test_distilbert(summary: Dict[str, Any]) -> None:
    info = {"ok": False, "path": None, "cosine": None, "error": None, "device": None}
    p = first_dir(CAND_DISTIL)
    info["path"] = str(p) if p else None
    if not p:
        warn("DistilBERT folder not found in candidates.")
        summary["distilbert"] = info
        return
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        info["error"] = f"transformers/torch not installed: {e}"
        err(info["error"])
        summary["distilbert"] = info
        return
    try:
        tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
        mdl = AutoModel.from_pretrained(str(p), local_files_only=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        info["device"] = dev
        mdl.to(dev).eval()
        with torch.no_grad():
            t = tok(["invoice total", "amount due"], return_tensors="pt",
                    padding=True, truncation=True, max_length=64)
            t = {k: v.to(dev) for k, v in t.items()}
            hs = mdl(**t).last_hidden_state             # [B,T,H]
            mask = t["attention_mask"].unsqueeze(-1)    # [B,T,1]
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            cos = float((pooled[0] @ pooled[1].T).item())
        ok(f"DistilBERT loaded from {p} on {dev} (cosine={cos:.3f})")
        info["ok"] = True
        info["cosine"] = cos
    except Exception as e:
        info["error"] = f"load/encode failed: {e}"
        err(info["error"])
    summary["distilbert"] = info

def test_layoutlmv3(summary: Dict[str, Any]) -> None:
    info = {"ok": False, "path": None, "hidden_size": None, "error": None}
    p = first_dir(CAND_LAYOUT)
    info["path"] = str(p) if p else None
    if not p:
        warn("LayoutLMv3 folder not found in candidates.")
        summary["layoutlmv3"] = info
        return
    try:
        from transformers import AutoProcessor, LayoutLMv3Model
    except Exception as e:
        info["error"] = f"transformers not installed: {e}"
        err(info["error"])
        summary["layoutlmv3"] = info
        return
    try:
        # STRICT local
        proc = AutoProcessor.from_pretrained(str(p), local_files_only=True)
        mdl  = LayoutLMv3Model.from_pretrained(str(p), local_files_only=True)
        hs = int(mdl.config.hidden_size)
        ok(f"LayoutLMv3 loaded from {p} (hidden_size={hs})")
        info["ok"] = True
        info["hidden_size"] = hs
    except Exception as e:
        info["error"] = f"load failed: {e}"
        err(info["error"])
    summary["layoutlmv3"] = info

def main():
    print(f"Models root guess: {MODELS_ROOT}")
    summary = {
        "root": str(MODELS_ROOT),
        "env": {
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        }
    }
    test_minilm(summary)
    test_distilbert(summary)
    test_layoutlmv3(summary)

    out = ROOT / "verify_local_models.json"
    out.write_text(json.dumps(summary, indent=2))
    print("\nSummary written to:", out)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()