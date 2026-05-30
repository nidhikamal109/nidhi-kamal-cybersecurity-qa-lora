"""Shared helpers for dataset prep, LoRA setup, and I/O."""

import json
import os
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Defaults tuned for Apple M3 (MPS) and medium training time
DATASET_ID = "AlicanKiraz0/Cybersecurity-Dataset-v1"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
MAX_SEQ_LEN = 512

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def results_dir() -> Path:
    path = project_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir() -> Path:
    path = project_root() / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_token() -> Optional[str]:
    # Set HF_TOKEN in .env or environment when using gated models
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def use_mps() -> bool:
    return torch.backends.mps.is_available()


def messages_to_qa(messages: list) -> Tuple[str, str]:
    """Extract user question and assistant answer from chat messages."""
    question, answer = "", ""
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role == "user" and content:
            question = content
        elif role == "assistant" and content:
            answer = content
    return question, answer


def format_prompt(question: str, answer: Optional[str] = None) -> str:
    """Chat-style prompt used for SFT (TinyLlama chat template simplified)."""
    if answer is None:
        return (
            f"<|system|>\nYou are a cybersecurity expert assistant.</s>\n"
            f"<|user|>\n{question}</s>\n"
            f"<|assistant|>\n"
        )
    return (
        f"<|system|>\nYou are a cybersecurity expert assistant.</s>\n"
        f"<|user|>\n{question}</s>\n"
        f"<|assistant|>\n{answer}</s>"
    )


def _row_to_text(row: dict) -> dict:
    if "messages" in row and row["messages"]:
        q, a = messages_to_qa(row["messages"])
    else:
        q = (row.get("user") or "").strip()
        a = (row.get("assistant") or "").strip()
    if not q or not a:
        return {"text": "", "question": "", "answer": ""}
    return {"text": format_prompt(q, a), "question": q, "answer": a}


def load_qa_dataset(
    train_n: int = TRAIN_SAMPLES,
    val_n: int = VAL_SAMPLES,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, dict]:
    """Load cybersecurity Q&A from Hugging Face (80/20 split)."""
    token = hf_token()
    raw = load_dataset(DATASET_ID, token=token)
    full = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]

    split = full.train_test_split(test_size=0.2, seed=seed)
    train_raw, val_raw = split["train"], split["test"]

    train_ds = train_raw.shuffle(seed=seed).map(
        _row_to_text, remove_columns=train_raw.column_names
    )
    val_ds = val_raw.shuffle(seed=seed).map(
        _row_to_text, remove_columns=val_raw.column_names
    )

    train_ds = train_ds.filter(lambda x: len(x["text"]) > 0)
    val_ds = val_ds.filter(lambda x: len(x["text"]) > 0)
    train_ds = train_ds.select(range(min(train_n, len(train_ds))))
    val_ds = val_ds.select(range(min(val_n, len(val_ds))))

    stats = {
        "dataset_id": DATASET_ID,
        "full_rows": len(full),
        "used_train_rows": len(train_ds),
        "used_val_rows": len(val_ds),
        "split": "80/20 train_test_split",
        "max_seq_len": MAX_SEQ_LEN,
        "avg_answer_chars": round(
            sum(len(r["answer"]) for r in train_ds) / max(len(train_ds), 1)
        ),
    }
    return train_ds, val_ds, stats


def load_tokenizer(model_id: str = MODEL_ID) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        labels = []
        for ids, mask in zip(out["input_ids"], out["attention_mask"]):
            labels.append([t if m else -100 for t, m in zip(ids, mask)])
        out["labels"] = labels
        return out

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def load_base_model(
    model_id: str = MODEL_ID,
    use_4bit: bool = False,
) -> AutoModelForCausalLM:
    """
    Load causal LM. 4-bit quantization is disabled on Mac (bitsandbytes needs CUDA).
    """
    token = hf_token()
    kwargs: dict[str, Any] = {"token": token}

    if use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit quantization requires NVIDIA CUDA; skipped on Apple Silicon."
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        # MPS training is more stable in fp32
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_4bit:
        model.to(get_device())
    return model


def apply_lora(model: AutoModelForCausalLM, use_4bit: bool = False) -> AutoModelForCausalLM:
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    model.eval()
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_device())

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    return text.replace("</s>", "").strip()
