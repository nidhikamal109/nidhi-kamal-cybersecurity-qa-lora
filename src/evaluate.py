"""Evaluate base vs LoRA model on validation samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import re

from peft import PeftModel
from rouge_score import rouge_scorer

from utils import (
    MODEL_ID,
    checkpoints_dir,
    generate_answer,
    load_base_model,
    load_json,
    load_qa_dataset,
    load_tokenizer,
    results_dir,
    save_json,
)


def rouge_l(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(ref, pred)["rougeL"].fmeasure


def categorize_error(pred: str, ref: str) -> str:
    pred_l, ref_l = pred.lower(), ref.lower()
    if not pred.strip():
        return "empty_response"
    if pred_l == ref_l:
        return "exact_match"
    if pred_l in ref_l or ref_l in pred_l:
        return "partial_overlap"
    # Simple hallucination heuristic: many tokens in pred absent from ref
    pred_tokens = set(re.findall(r"\w+", pred_l))
    ref_tokens = set(re.findall(r"\w+", ref_l))
    if pred_tokens and len(pred_tokens - ref_tokens) / len(pred_tokens) > 0.6:
        return "possible_hallucination"
    return "semantic_mismatch"


def run_eval(model, tokenizer, val_ds, n_samples: int):
    scores, rows, errors = [], [], {}
    n = min(n_samples, len(val_ds))

    for i in range(n):
        q = val_ds[i]["question"]
        ref = val_ds[i]["answer"]
        pred = generate_answer(model, tokenizer, q)
        score = rouge_l(pred, ref)
        scores.append(score)
        err = categorize_error(pred, ref)
        errors[err] = errors.get(err, 0) + 1
        rows.append({"question": q, "reference": ref, "prediction": pred, "rouge_l": round(score, 4), "error_type": err})

    return {
        "mean_rouge_l": round(sum(scores) / max(len(scores), 1), 4),
        "samples": rows,
        "error_breakdown": errors,
    }


def write_samples(path: Path, base_res: dict, lora_res: dict) -> None:
    with path.open("w") as f:
        f.write("=== BASE MODEL ===\n")
        f.write(f"mean ROUGE-L: {base_res['mean_rouge_l']}\n")
        f.write(f"errors: {base_res['error_breakdown']}\n\n")
        for i, s in enumerate(base_res["samples"], 1):
            f.write(f"--- Sample {i} ({s['error_type']}, rouge_l={s['rouge_l']}) ---\n")
            f.write(f"Q: {s['question']}\n")
            f.write(f"Expected: {s['reference'][:800]}\n")
            f.write(f"Predicted: {s['prediction'][:800]}\n\n")

        f.write("\n=== FINE-TUNED (LoRA) ===\n")
        f.write(f"mean ROUGE-L: {lora_res['mean_rouge_l']}\n")
        f.write(f"errors: {lora_res['error_breakdown']}\n\n")
        for i, s in enumerate(lora_res["samples"], 1):
            f.write(f"--- Sample {i} ({s['error_type']}, rouge_l={s['rouge_l']}) ---\n")
            f.write(f"Q: {s['question']}\n")
            f.write(f"Expected: {s['reference'][:800]}\n")
            f.write(f"Predicted: {s['prediction'][:800]}\n\n")


def overfitting_note(train_loss: float, val_loss: float) -> str:
    gap = val_loss - train_loss
    if gap > 0.5:
        return "Validation loss notably higher than train — possible overfitting or distribution shift."
    if gap < 0.1:
        return "Train and validation loss are close — model may be underfitting or well-regularized."
    return "Moderate train/val gap — typical for a single epoch LoRA run."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    _, val_ds, stats = load_qa_dataset()
    tokenizer = load_tokenizer()
    adapter = checkpoints_dir() / "lora_adapter"

    print("Evaluating base model...")
    base_model = load_base_model()
    base_res = run_eval(base_model, tokenizer, val_ds, args.samples)
    del base_model

    print("Evaluating LoRA model...")
    base_for_lora = load_base_model()
    if adapter.exists():
        lora_model = PeftModel.from_pretrained(base_for_lora, str(adapter))
    else:
        raise FileNotFoundError(f"No adapter at {adapter}. Run train.py first.")
    lora_res = run_eval(lora_model, tokenizer, val_ds, args.samples)

    summary_path = results_dir() / "training_summary.json"
    train_loss, val_loss = None, None
    if summary_path.exists():
        summary = load_json(summary_path)
        for row in reversed(summary.get("log_history", [])):
            if val_loss is None and "eval_loss" in row:
                val_loss = row["eval_loss"]
            if train_loss is None and "loss" in row and "eval_loss" not in row:
                train_loss = row["loss"]
        if train_loss is None:
            train_loss = summary.get("final_train_loss")

    metrics = {
        "dataset": stats,
        "validation_samples_evaluated": args.samples,
        "base_model": {"model_id": MODEL_ID, **{k: v for k, v in base_res.items() if k != "samples"}},
        "lora_model": {"adapter": str(adapter), **{k: v for k, v in lora_res.items() if k != "samples"}},
        "improvement_rouge_l": round(lora_res["mean_rouge_l"] - base_res["mean_rouge_l"], 4),
        "overfitting_analysis": overfitting_note(train_loss or 0, val_loss or 0) if val_loss else "Run train.py to log eval_loss.",
        "failure_modes": {
            "base": base_res["error_breakdown"],
            "lora": lora_res["error_breakdown"],
        },
    }

    save_json(results_dir() / "metrics.json", metrics)
    write_samples(results_dir() / "sample_predictions.txt", base_res, lora_res)
    print(f"Base ROUGE-L: {base_res['mean_rouge_l']}, LoRA ROUGE-L: {lora_res['mean_rouge_l']}")
    print(f"Results written to {results_dir()}")


if __name__ == "__main__":
    main()
