from typing import Dict, Any, List
import os, numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseValidator, ValidationResult

class BERTValidator(BaseValidator):
    name = "bert"

    def __init__(self, model_name: str = "bert-base-uncased", model_dir: str | None = None):
        path = model_dir if model_dir else model_name
        self.model = SentenceTransformer(path)

    def validate_field(self, field_name: str, llm_value: str, ocr_tokens: List[Dict[str, Any]]) -> ValidationResult:
        texts = [llm_value] + [t.get("text","") for t in ocr_tokens]
        embs = self.model.encode(texts, normalize_embeddings=True)
        llm_emb = embs[0]; token_embs = embs[1:]
        sims = token_embs @ llm_emb
        best_i = int(np.argmax(sims)) if len(sims) else -1
        best_score = float(sims[best_i]) if best_i >= 0 else -1.0
        tok = ocr_tokens[best_i] if best_i >= 0 else None
        return ValidationResult(
            field=field_name, llm_value=llm_value, method=self.name,
            candidate_value=tok.get("text","") if tok else "",
            confidence=float(max(0, min(1, (best_score+1)/2))),
            bbox=tok.get("bbox") if tok else None,
            explanation="Semantic similarity (BERT pooled embeddings)."
        )