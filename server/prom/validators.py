#init.py

from .autolocate import AutoLocateValidator
from .bert import BERTValidator
from .minilm import MiniLMValidator
from .layoutlmv3 import LayoutLMv3Validator

# run_validation.py

#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

from validators import AutoLocateValidator, BERTValidator, MiniLMValidator, LayoutLMv3Validator
from io_schemas import normalize_ocr, normalize_llm
from utils import load_json, write_csv
from metrics import choose_winner

def validate_document(ocr_tokens, llm_fields, validators) -> List[Dict[str, Any]]:
    results_rows = []
    for field_name, llm_value in llm_fields.items():
        method_results: Dict[str, Dict[str, Any]] = {}
        for v in validators:
            try:
                res = v.validate_field(field_name, llm_value, ocr_tokens)
            except Exception as e:
                res = dict(field=field_name, llm_value=llm_value, method=v.name,
                           candidate_value="", confidence=0.0, bbox=None,
                           explanation=f"validator error: {e}")
            method_results[v.name] = res

        winner = choose_winner(method_results)
        chosen = method_results[winner]

        # Flatten one row per field; include all confidences for observability
        row = {
            "field": field_name,
            "llm_value": llm_value,
            "winner_method": winner,
            "winner_candidate": chosen.get("candidate_value",""),
            "winner_confidence": chosen.get("confidence",0.0),
            "winner_bbox": json.dumps(chosen.get("bbox")),
            "autolocate_conf": method_results["autolocate"].get("confidence",0.0),
            "bert_conf":       method_results["bert"].get("confidence",0.0),
            "minilm_conf":     method_results["minilm"].get("confidence",0.0),
            "layoutlmv3_conf": method_results["layoutlmv3"].get("confidence",0.0),
            "autolocate_val":  method_results["autolocate"].get("candidate_value",""),
            "bert_val":        method_results["bert"].get("candidate_value",""),
            "minilm_val":      method_results["minilm"].get("candidate_value",""),
            "layoutlmv3_val":  method_results["layoutlmv3"].get("candidate_value",""),
        }
        results_rows.append(row)
    return results_rows

def main():
    ap = argparse.ArgumentParser(description="Document field validation at scale")
    ap.add_argument("--ocr", required=True, help="Path to OCR tokens JSON")
    ap.add_argument("--llm", required=True, help="Path to LLM extraction JSON")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--device", default=None, help="Override device for LayoutLMv3 (cpu|cuda)")
    ap.add_argument("--model_dir_override", default=None, help="Optional local models dir (uses HF cache otherwise)")
    args = ap.parse_args()

    ocr_json = load_json(args.ocr)
    llm_json = load_json(args.llm)

    ocr_tokens = normalize_ocr(ocr_json)
    llm_fields = normalize_llm(llm_json)

    # Instantiate validators
    autolocate = AutoLocateValidator()
    bert = BERTValidator(model_name="bert-base-uncased")
    minilm = MiniLMValidator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    layout = LayoutLMv3Validator(model_name="microsoft/layoutlmv3-base", device=args.device)

    validators = [autolocate, bert, minilm, layout]

    rows = validate_document(ocr_tokens, llm_fields, validators)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    print(f"✅ Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
    

#utils.py

import json
from pathlib import Path
from typing import Any, Dict

def load_json(p: str) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def write_csv(rows, out_path: str):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    
    
#metrics.py

from typing import Optional, Dict, Any
from rapidfuzz import fuzz

def value_match_score(pred: str, truth: str) -> float:
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    return fuzz.token_set_ratio(pred.lower().strip(), truth.lower().strip()) / 100.0

def choose_winner(candidates: Dict[str, Dict[str, Any]]) -> str:
    """
    candidates: {method_name: ValidationResult}
    Heuristic: pick highest confidence; if tie, prefer layoutlmv3 > minilm > bert > autolocate.
    """
    rank = {"layoutlmv3": 4, "minilm": 3, "bert": 2, "autolocate": 1}
    best_method = None
    best_c = -1
    best_rank = -1
    for m, res in candidates.items():
        c = float(res.get("confidence", 0.0))
        r = rank.get(m, 0)
        if c > best_c or (abs(c-best_c) < 1e-6 and r > best_rank):
            best_c, best_rank, best_method = c, r, m
    return best_method or "autolocate"
    
# io_schemas.py

from typing import List, Dict, Any

# Expected OCR format (per token):
# { "text": "Acme Inc.", "bbox": [x0,y0,x1,y1], "page": 0 }
# A document is a list of tokens possibly across pages.

# Expected LLM extraction format:
# {
#   "fields": [
#       {"name": "VendorName", "value": "Acme Inc."},
#       {"name": "InvoiceDate", "value": "2024-02-01"},
#       ...
#   ]
# }

def normalize_ocr(ocr_json: Any) -> List[Dict[str, Any]]:
    # If your Tesseract JSON differs, adapt here.
    return list(ocr_json)

def normalize_llm(llm_json: Any) -> Dict[str, str]:
    fields = {}
    for f in llm_json.get("fields", []):
        name = f.get("name")
        val  = f.get("value", "")
        if name:
            fields[name] = val
    return fields
    
#layoutlmv3.py

from typing import Dict, Any, List
import torch
import numpy as np
from transformers import AutoProcessor, LayoutLMv3Model
from .base import BaseValidator, ValidationResult

def _norm_bbox(bbox, w=1000, h=1000):
    # If your OCR bboxes are already in 0..1000 space, pass-through.
    # Otherwise normalize to 0..1000.
    x0,y0,x1,y1 = bbox
    return [int(x0),int(y0),int(x1),int(y1)]

class LayoutLMv3Validator(BaseValidator):
    name = "layoutlmv3"

    def __init__(self, model_name: str = "microsoft/layoutlmv3-base", device: str = None):
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3Model.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _encode_tokens(self, tokens: List[Dict[str,Any]], page_w: int = 1000, page_h: int = 1000):
        words = [t.get("text","") for t in tokens]
        boxes = [t.get("bbox",[0,0,0,0]) for t in tokens]
        boxes = [_norm_bbox(b, page_w, page_h) for b in boxes]
        enc = self.processor(text=words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        return enc

    def _encode_query(self, query: str):
        enc = self.processor(text=[query], boxes=[[0,0,0,0]], return_tensors="pt", truncation=True, padding="max_length", max_length=32)
        return enc

    def validate_field(self,
                       field_name: str,
                       llm_value: str,
                       ocr_tokens: List[Dict[str, Any]]) -> ValidationResult:
        if not ocr_tokens:
            return ValidationResult(
                field=field_name, llm_value=llm_value, method=self.name,
                candidate_value="", confidence=0.0, bbox=None,
                explanation="No OCR tokens."
            )

        with torch.no_grad():
            doc_enc = self._encode_tokens(ocr_tokens)
            q_enc = self._encode_query(llm_value)

            for k in doc_enc: doc_enc[k] = doc_enc[k].to(self.device)
            for k in q_enc: q_enc[k] = q_enc[k].to(self.device)

            doc_out = self.model(**doc_enc)
            q_out = self.model(**q_enc)

            # Mean-pool last hidden state as layout-aware embeddings
            doc_emb = doc_out.last_hidden_state.mean(dim=1)    # (1, D)
            q_emb = q_out.last_hidden_state.mean(dim=1)        # (1, D)

            # Token-level embeddings: we’ll also compute sim to each token position
            token_embs = doc_out.last_hidden_state[0]  # (seq_len, D)
            qv = q_emb[0]                              # (D,)

            # Cosine similarity against each token position
            token_norm = token_embs / (token_embs.norm(dim=1, keepdim=True) + 1e-12)
            q_norm = qv / (qv.norm() + 1e-12)
            sims = torch.matmul(token_norm, q_norm)  # (seq_len,)

            best_idx = int(torch.argmax(sims).item())
            best_score = float(sims[best_idx].item())

        # Map best token id back to OCR token (accounting for special tokens)
        # processor created special tokens; for simplicity assume first len(ocr_tokens) map mostly 1:1
        # (Good enough for validation ranking; for production, keep alignment map from processor)
        real_idx = min(best_idx, len(ocr_tokens)-1)
        best_tok = ocr_tokens[real_idx]
        return ValidationResult(
            field=field_name,
            llm_value=llm_value,
            method=self.name,
            candidate_value=best_tok.get("text",""),
            confidence=float(max(0.0, min(1.0, (best_score + 1) / 2))),
            bbox=best_tok.get("bbox"),
            explanation="Layout-aware similarity via LayoutLMv3 pooled/positional embeddings."
        )
        
#minilm.py
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseValidator, ValidationResult

class MiniLMValidator(BaseValidator):
    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def validate_field(self,
                       field_name: str,
                       llm_value: str,
                       ocr_tokens: List[Dict[str, Any]]) -> ValidationResult:
        texts = [llm_value] + [t.get("text","") for t in ocr_tokens]
        embs = self.model.encode(texts, normalize_embeddings=True)
        llm_emb = embs[0]
        token_embs = embs[1:]

        sims = token_embs @ llm_emb  # cosine, since normalized
        best_i = int(np.argmax(sims)) if len(sims) else -1
        best_score = float(sims[best_i]) if best_i >= 0 else -1.0
        best_tok = ocr_tokens[best_i] if best_i >= 0 else None

        return ValidationResult(
            field=field_name,
            llm_value=llm_value,
            method=self.name,
            candidate_value=best_tok.get("text","") if best_tok else "",
            confidence=float(max(0.0, min(1.0, (best_score + 1) / 2))),
            bbox=best_tok.get("bbox") if best_tok else None,
            explanation="Fast semantic similarity (MiniLM)."
        )
        
        
#bert.py

from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseValidator, ValidationResult

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

class BERTValidator(BaseValidator):
    name = "bert"

    def __init__(self, model_name: str = "bert-base-uncased"):
        # Using SentenceTransformer interface for pooling w/ vanilla BERT backbone
        # If you prefer HF feature-extraction: adapt accordingly.
        self.model = SentenceTransformer(model_name)

    def validate_field(self,
                       field_name: str,
                       llm_value: str,
                       ocr_tokens: List[Dict[str, Any]]) -> ValidationResult:
        texts = [llm_value] + [t.get("text","") for t in ocr_tokens]
        embs = self.model.encode(texts, normalize_embeddings=True)  # (N, d)
        llm_emb = embs[0]
        token_embs = embs[1:]

        best_i = -1
        best_score = -1.0
        for i, t in enumerate(ocr_tokens):
            score = float(np.dot(llm_emb, token_embs[i]))
            if score > best_score:
                best_score = score
                best_i = i

        best_tok = ocr_tokens[best_i] if best_i >= 0 else None
        return ValidationResult(
            field=field_name,
            llm_value=llm_value,
            method=self.name,
            candidate_value=best_tok.get("text","") if best_tok else "",
            confidence=float(max(0.0, min(1.0, (best_score + 1) / 2))),  # map cos [-1,1] -> [0,1]
            bbox=best_tok.get("bbox") if best_tok else None,
            explanation="Semantic similarity (BERT pooled embeddings)."
        )
        
#autolocate.py
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseValidator, ValidationResult

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

class BERTValidator(BaseValidator):
    name = "bert"

    def __init__(self, model_name: str = "bert-base-uncased"):
        # Using SentenceTransformer interface for pooling w/ vanilla BERT backbone
        # If you prefer HF feature-extraction: adapt accordingly.
        self.model = SentenceTransformer(model_name)

    def validate_field(self,
                       field_name: str,
                       llm_value: str,
                       ocr_tokens: List[Dict[str, Any]]) -> ValidationResult:
        texts = [llm_value] + [t.get("text","") for t in ocr_tokens]
        embs = self.model.encode(texts, normalize_embeddings=True)  # (N, d)
        llm_emb = embs[0]
        token_embs = embs[1:]

        best_i = -1
        best_score = -1.0
        for i, t in enumerate(ocr_tokens):
            score = float(np.dot(llm_emb, token_embs[i]))
            if score > best_score:
                best_score = score
                best_i = i

        best_tok = ocr_tokens[best_i] if best_i >= 0 else None
        return ValidationResult(
            field=field_name,
            llm_value=llm_value,
            method=self.name,
            candidate_value=best_tok.get("text","") if best_tok else "",
            confidence=float(max(0.0, min(1.0, (best_score + 1) / 2))),  # map cos [-1,1] -> [0,1]
            bbox=best_tok.get("bbox") if best_tok else None,
            explanation="Semantic similarity (BERT pooled embeddings)."
        )
        
#base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ValidationResult(dict):
    """
    Standard result structure per field:
    {
      "field": str,
      "llm_value": str,
      "method": str,             # autolocate|bert|minilm|layoutlmv3
      "candidate_value": str,    # OCR-derived best candidate
      "confidence": float,       # 0..1
      "bbox": [x0,y0,x1,y1] or None,
      "explanation": str
    }
    """

class BaseValidator(ABC):
    name: str = "base"

    @abstractmethod
    def validate_field(self,
                       field_name: str,
                       llm_value: str,
                       ocr_tokens: List[Dict[str, Any]]) -> ValidationRe


    
    