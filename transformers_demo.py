#!/usr/bin/env python3
"""Run a small Hugging Face text-generation model on CPU or ROCm GPU.

Usage:
    uv run python transformers_demo.py
    uv run python transformers_demo.py --prompt "The future of AI is"
    uv run python transformers_demo.py --device cpu --max-new-tokens 30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from benchmark_logs import CHILD_ENV, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(
        run_logged(Path(__file__), sys.argv[1:], "transformers.txt")
    )

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="distilgpt2",
        help="Hugging Face model ID (default: distilgpt2)",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time",
        help="Text to continue",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of generated tokens (default: 50)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; 'cuda' also selects ROCm in PyTorch",
    )
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {args.model!r} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    print(f"Input tokens: {input_ids.shape[-1]}")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=encoded["attention_mask"].to(device),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation_ids = output_ids[0, input_ids.shape[-1] :]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generated text: {continuation}")
    print(f"Generated tokens: {continuation_ids.numel()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
