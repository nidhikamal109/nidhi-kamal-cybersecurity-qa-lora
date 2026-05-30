"""Fine-tune TinyLlama with LoRA on NIST cybersecurity Q&A."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, default_data_collator

from utils import (
    MODEL_ID,
    TRAIN_SAMPLES,
    VAL_SAMPLES,
    apply_lora,
    checkpoints_dir,
    get_device,
    load_base_model,
    load_qa_dataset,
    load_tokenizer,
    results_dir,
    save_json,
    tokenize_dataset,
    use_mps,
)


def plot_learning_curves(log_history: list, out_path: Path) -> None:
    steps, train_loss, eval_loss, eval_steps = [], [], [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            steps.append(entry.get("step", len(steps)))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_loss.append(entry["eval_loss"])

    if not train_loss:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="val loss", marker="o")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} (mps={use_mps()})")

    train_n = args.train_samples or TRAIN_SAMPLES
    val_n = args.val_samples or VAL_SAMPLES
    train_ds, val_ds, stats = load_qa_dataset(train_n=train_n, val_n=val_n)
    save_json(results_dir() / "dataset_stats.json", stats)
    print(f"Dataset: {stats['used_train_rows']} train, {stats['used_val_rows']} val")

    tokenizer = load_tokenizer()
    train_tok = tokenize_dataset(train_ds, tokenizer)
    val_tok = tokenize_dataset(val_ds, tokenizer)

    # 4-bit skipped on Mac — see README bonus section
    model = load_base_model(use_4bit=False)
    model = apply_lora(model, use_4bit=False)
    model.config.use_cache = False

    output_dir = checkpoints_dir() / "lora_adapter"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # MPS fp16 is unstable for this setup
        bf16=False,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    collator = default_data_collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
    )

    start = time.time()
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    elapsed = round((time.time() - start) / 60, 2)
    log_history = trainer.state.log_history

    logs_path = results_dir() / "training_logs.txt"
    with logs_path.open("w") as f:
        f.write(f"model={MODEL_ID}\n")
        f.write(f"device={device}\n")
        f.write(f"train_minutes={elapsed}\n")
        f.write(f"final_train_loss={train_result.training_loss}\n\n")
        for row in log_history:
            f.write(f"{row}\n")

    save_json(
        results_dir() / "training_summary.json",
        {
            "train_runtime_minutes": elapsed,
            "train_samples": stats["used_train_rows"],
            "final_train_loss": train_result.training_loss,
            "log_history": log_history,
        },
    )
    plot_learning_curves(log_history, results_dir() / "learning_curves.png")
    print(f"Done in {elapsed} min. Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
