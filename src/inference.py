"""Run inference on a single question or start a minimal local API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import time

from peft import PeftModel

from utils import (
    checkpoints_dir,
    generate_answer,
    load_base_model,
    load_tokenizer,
)


def load_lora_model():
    base = load_base_model()
    adapter = checkpoints_dir() / "lora_adapter"
    if not adapter.exists():
        raise FileNotFoundError("LoRA adapter not found. Run: python src/train.py")
    return PeftModel.from_pretrained(base, str(adapter))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Cybersecurity question to answer")
    parser.add_argument("--serve", action="store_true", help="Start minimal FastAPI server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        # Bonus deployment: lightweight API (no extra dependency beyond stdlib + uvicorn optional)
        try:
            from fastapi import FastAPI
            import uvicorn
        except ImportError:
            print("Install optional deps for API: pip install fastapi uvicorn")
            return

        app = FastAPI(title="Cybersecurity Q&A LoRA")
        tokenizer = load_tokenizer()
        model = load_lora_model()

        @app.post("/ask")
        def ask(payload: dict):
            q = payload.get("question", "")
            t0 = time.time()
            answer = generate_answer(model, tokenizer, q)
            return {"answer": answer, "latency_sec": round(time.time() - t0, 3)}

        uvicorn.run(app, host="127.0.0.1", port=args.port)
        return

    question = args.question or input("Question: ").strip()
    if not question:
        print("Provide --question or enter one interactively.")
        return

    tokenizer = load_tokenizer()
    model = load_lora_model()
    answer = generate_answer(model, tokenizer, question)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
