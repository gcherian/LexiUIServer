from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from validators import (
    AutoLocateValidator, BERTValidator, MiniLMValidator, LayoutLMv3Validator
)
from io_schemas import normalize_ocr, normalize_llm
from metrics import choose_winner

APP = FastAPI(title="EDIP Validation API")

# ---------- Schemas ----------
class OCRToken(BaseModel):
    text: str
    bbox: List[float] = Field(default_factory=list)
    page: int = 0

class ClientAutoLocate(BaseModel):
    field: str
    candidate_value: str
    confidence: float
    bbox: Optional[List[float]] = None

class ValidateRequest(BaseModel):
    # Option A: inline
    ocr_tokens: Optional[List[OCRToken]] = None
    llm_fields: Optional[Dict[str, str]] = None
    # Option B: server paths
    ocr_tokens_path: Optional[str] = None
    llm_json_path: Optional[str] = None

    # Optional: pass client-side autolocate outcome to blend in
    client_autolocate: Optional[List[ClientAutoLocate]] = None

    # Optional model base dir (default: ./models)
    models_dir: Optional[str] = "models"
    device: Optional[str] = None  # "cpu" or "cuda"

class FieldResult(BaseModel):
    field: str
    llm_value: str
    winner_method: str
    winner_candidate: str
    winner_confidence: float
    winner_bbox: Optional[List[float]]
    autolocate_conf: float
    bert_conf: float
    minilm_conf: float
    layoutlmv3_conf: float
    autolocate_val: str
    bert_val: str
    minilm_val: str
    layoutlmv3_val: str

class ValidateResponse(BaseModel):
    results: List[FieldResult]
    model_info: Dict[str, str]

# ---------- Helpers ----------
def _load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _blend_client_autolocate(field: str, client_list: List[ClientAutoLocate]) -> dict | None:
    for c in client_list:
        if c.field == field:
            return {
                "field": field,
                "llm_value": "",
                "method": "autolocate",
                "candidate_value": c.candidate_value,
                "confidence": c.confidence,
                "bbox": c.bbox,
                "explanation": "Client-provided autolocate (FLE)."
            }
    return None

# ---------- Route ----------
@APP.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest):
    # Load payload
    if req.ocr_tokens is not None and req.llm_fields is not None:
        ocr_tokens = [t.model_dump() for t in req.ocr_tokens]
        llm_fields = req.llm_fields
    elif req.ocr_tokens_path and req.llm_json_path:
        ocr_tokens = normalize_ocr(_load_json(req.ocr_tokens_path))
        llm_fields = normalize_llm(_load_json(req.llm_json_path))
    else:
        return ValidateResponse(results=[], model_info={"error": "Provide inline or path-based inputs"})

    # Instantiate validators (point to local ./models)
    mroot = req.models_dir or "models"
    autolocate = AutoLocateValidator()
    bert = BERTValidator(model_dir=str(Path(mroot)/"bert-base-uncased"))
    minilm = MiniLMValidator(model_dir=str(Path(mroot)/"sentence-transformers/all-MiniLM-L6-v2"))
    layout = LayoutLMv3Validator(model_name=str(Path(mroot)/"microsoft/layoutlmv3-base"), device=req.device)

    validators = {
        "autolocate": autolocate,
        "bert": bert,
        "minilm": minilm,
        "layoutlmv3": layout
    }

    results = []
    for field_name, llm_value in llm_fields.items():
        method_results = {}

        # If UI already ran autolocate, blend it; else compute here
        if req.client_autolocate:
            ca = _blend_client_autolocate(field_name, req.client_autolocate)
            if ca:
                method_results["autolocate"] = ca

        for name, v in validators.items():
            if name == "autolocate" and "autolocate" in method_results:
                continue
            try:
                res = v.validate_field(field_name, llm_value, ocr_tokens)
            except Exception as e:
                res = dict(field=field_name, llm_value=llm_value, method=name,
                           candidate_value="", confidence=0.0, bbox=None,
                           explanation=f"validator error: {e}")
            method_results[name] = res

        winner = choose_winner(method_results)
        chosen = method_results[winner]

        row = FieldResult(
            field=field_name,
            llm_value=llm_value,
            winner_method=winner,
            winner_candidate=chosen.get("candidate_value",""),
            winner_confidence=float(chosen.get("confidence",0.0)),
            winner_bbox=chosen.get("bbox"),
            autolocate_conf=float(method_results["autolocate"].get("confidence",0.0)),
            bert_conf=float(method_results["bert"].get("confidence",0.0)),
            minilm_conf=float(method_results["minilm"].get("confidence",0.0)),
            layoutlmv3_conf=float(method_results["layoutlmv3"].get("confidence",0.0)),
            autolocate_val=method_results["autolocate"].get("candidate_value",""),
            bert_val=method_results["bert"].get("candidate_value",""),
            minilm_val=method_results["minilm"].get("candidate_value",""),
            layoutlmv3_val=method_results["layoutlmv3"].get("candidate_value",""),
        )
        results.append(row)

    return ValidateResponse(
        results=results,
        model_info={
            "bert": str(Path(mroot)/"bert-base-uncased"),
            "minilm": str(Path(mroot)/"sentence-transformers/all-MiniLM-L6-v2"),
            "layoutlmv3": str(Path(mroot)/"microsoft/layoutlmv3-base"),
            "device": layout.device
        }
    )