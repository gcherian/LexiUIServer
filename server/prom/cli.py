import argparse, json
from pathlib import Path
import pandas as pd

from validators import AutoLocateValidator, BERTValidator, MiniLMValidator, LayoutLMv3Validator
from io_schemas import normalize_ocr, normalize_llm
from metrics import choose_winner

def validate_one(ocr_path: Path, llm_path: Path, models_dir: Path, device: str | None):
    ocr_tokens = normalize_ocr(json.loads(ocr_path.read_text(encoding="utf-8")))
    llm_fields = normalize_llm(json.loads(llm_path.read_text(encoding="utf-8")))

    autolocate = AutoLocateValidator()
    bert = BERTValidator(model_dir=str(models_dir/"bert-base-uncased"))
    minilm = MiniLMValidator(model_dir=str(models_dir/"sentence-transformers/all-MiniLM-L6-v2"))
    layout = LayoutLMv3Validator(model_name=str(models_dir/"microsoft/layoutlmv3-base"), device=device)

    validators = {"autolocate": autolocate, "bert": bert, "minilm": minilm, "layoutlmv3": layout}
    rows = []
    for field, llm_val in llm_fields.items():
        mres = {}
        for name, v in validators.items():
            try:
                r = v.validate_field(field, llm_val, ocr_tokens)
            except Exception as e:
                r = dict(field=field, llm_value=llm_val, method=name,
                         candidate_value="", confidence=0.0, bbox=None,
                         explanation=f"validator error: {e}")
            mres[name] = r
        winner = choose_winner(mres)
        chosen = mres[winner]
        rows.append({
            "field": field,
            "llm_value": llm_val,
            "winner_method": winner,
            "winner_candidate": chosen.get("candidate_value",""),
            "winner_confidence": float(chosen.get("confidence",0.0)),
            "winner_bbox": json.dumps(chosen.get("bbox")),
            "autolocate_conf": float(mres["autolocate"].get("confidence",0.0)),
            "bert_conf": float(mres["bert"].get("confidence",0.0)),
            "minilm_conf": float(mres["minilm"].get("confidence",0.0)),
            "layoutlmv3_conf": float(mres["layoutlmv3"].get("confidence",0.0)),
            "autolocate_val": mres["autolocate"].get("candidate_value",""),
            "bert_val": mres["bert"].get("candidate_value",""),
            "minilm_val": mres["minilm"].get("candidate_value",""),
            "layoutlmv3_val": mres["layoutlmv3"].get("candidate_value",""),
            "ocr_path": str(ocr_path),
            "llm_path": str(llm_path)
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder of unzipped EDIP-Extraction")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--device", default=None, help="cpu|cuda")
    ap.add_argument("--out_dir", default="validation_out")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    all_rows = []
    for llm_path in root.glob("**/document.json"):
        ocr1 = llm_path.with_name("ocr_tokens.json")
        ocr2 = llm_path.with_suffix(".ocr.json")
        ocr_path = ocr1 if ocr1.exists() else (ocr2 if ocr2.exists() else None)
        if not ocr_path:
            print(f"Skip (no OCR): {llm_path}")
            continue
        rows = validate_one(ocr_path, llm_path, models_dir, args.device)
        out_csv = out_dir / f"{llm_path.parent.name}__{llm_path.parent.parent.name}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
        all_rows.extend(rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(out_dir/"validation_aggregate.parquet", index=False)
        print(f"Aggregate: {out_dir/'validation_aggregate.parquet'}  rows={len(df)}")

if __name__ == "__main__":
    main()