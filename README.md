# Cybersecurity Q&A LoRA (nidhi-kamal)

Fine-tune **TinyLlama 1.1B** with **LoRA** on cybersecurity Q&A data for a domain-specific assistant. Built for **Apple M3 (MPS)**; no NVIDIA GPU required.

## Dataset

| Item | Detail |
|------|--------|
| Source | [AlicanKiraz0/Cybersecurity-Dataset-v1](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1) |
| Format | `user` / `assistant` instruction pairs (~2.5k rows, ~4 MB) |
| Why this set | Assignment-approved, pure Q&A, small enough for medium M3 runs without multi-GB downloads |
| Split | 80/20 `train_test_split` → **2000 train / 500 val** |
| Preprocessing | Map to TinyLlama chat prompts, tokenize (max 512), mask padding in labels |

*NIST dataset (`ethanolivertroy/nist-cybersecurity-training`) is also valid but ~11 GB; this project prioritizes a practical local POC.*

Stats are written to `results/dataset_stats.json` after training starts.

## Model

| Item | Detail |
|------|--------|
| Model | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Params | ~1.1B (under 3B limit) |
| Rationale | Non-gated (no HF token required), fits **18 GB unified memory** on M3, fast enough for medium training |
| Hardware | Apple M3 Pro, MPS backend, ~18 GB RAM |

## LoRA config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` | 16 | Balance capacity vs memory on MPS |
| `lora_alpha` | 32 | Standard 2× rank scaling |
| `lora_dropout` | 0.05 | Light regularization |
| `target_modules` | q, k, v, o projections | Standard attention adapters for causal LM |

Trainable params are printed at train time (~1–2% of base model).

## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1 (default) |
| Batch size | 2 |
| Gradient accumulation | 8 (effective batch 16) |
| Learning rate | 2e-4 |
| Eval every | 100 steps |
| Gradient checkpointing | On |

Expected runtime on M3: **~1–3 hours** (network + first download add extra time).

Challenges on Mac: no CUDA 4-bit, MPS prefers fp32 training here; large HF dataset download on first run.

## Results (after train + evaluate)

- `results/metrics.json` — ROUGE-L, base vs LoRA, error breakdown, overfitting note
- `results/sample_predictions.txt` — 8+ examples with ground truth
- `results/training_logs.txt` — step losses
- `results/learning_curves.png` — train/val loss plot
- `results/training_summary.json` — full log history

## Bonus features

| Feature | Status |
|---------|--------|
| Base vs fine-tuned comparison | Implemented in `evaluate.py` |
| Error categorization | `empty_response`, `partial_overlap`, `possible_hallucination`, etc. |
| Deployment API | `inference.py --serve` (needs `pip install fastapi uvicorn`) |
| 4-bit / 8-bit quantization | **Skipped** — `bitsandbytes` needs NVIDIA CUDA, not available on Apple Silicon |

## Setup

Requires **Python 3.10+** (recommended). System 3.9 may fail with recent PyTorch.

```bash
cd cybersecurity-qa-lora
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional HF token (not required for TinyLlama + NIST dataset):

```bash
cp .env.example .env
# edit HF_TOKEN=...  then: export $(grep -v '^#' .env | xargs)
```

## How to run

```bash
source .venv/bin/activate

# 1) Train LoRA (full run: ~1-3 hours on M3)
python src/train.py

# Quick smoke train (optional)
python src/train.py --train-samples 80 --val-samples 20

# 2) Evaluate (base vs LoRA, writes metrics + samples)
python src/evaluate.py --samples 8

# 3) Inference
python src/inference.py --question "What is zero trust architecture according to NIST?"

# Optional API (bonus)
pip install fastapi uvicorn
python src/inference.py --serve --port 8000
# curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question":"What is MFA?"}'
```

## Project layout

```
cybersecurity-qa-lora/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── results/
├── checkpoints/          # gitignored
├── requirements.txt
└── README.md
```

## Limitations

- Subsampled data — not full 530k NIST corpus
- ROUGE-L is a lexical metric; semantic quality may differ
- 1 epoch LoRA — room for more tuning with longer runs
- MPS training is slower than NVIDIA CUDA at similar batch sizes

## Disqualification checks

- Uses **PEFT LoRA** (not full fine-tune, not Unsloth)
- Validation metrics and learning curves included
- No large weights committed (`checkpoints/` gitignored)
