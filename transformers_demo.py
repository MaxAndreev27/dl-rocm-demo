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
import time
from pathlib import Path

from benchmark_logs import CHILD_ENV, report_comparison, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(
        run_logged(Path(__file__), sys.argv[1:], "transformers.txt")
    )

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def run_generation(
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    device: str,
) -> tuple[int, float, str]:
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\nLoading {model_id!r} on {'iGPU' if device == 'cuda' else 'CPU'} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    torch.manual_seed(seed)
    with torch.inference_mode():
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(3, max_new_tokens),
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()

    torch.manual_seed(seed)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    continuation_ids = output_ids[0, input_ids.shape[-1] :]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    generated_tokens = continuation_ids.numel()
    tokens_per_second = generated_tokens / elapsed
    label = "iGPU" if device == "cuda" else "CPU"
    print(f"{label} generated text: {continuation}")
    print(
        f"{label}: tokens={generated_tokens}, time={elapsed:.3f}s, "
        f"throughput={tokens_per_second:.2f} tokens/s"
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return generated_tokens, tokens_per_second, continuation


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
        choices=("both", "cpu", "cuda"),
        default="both",
        help="Compare CPU then iGPU by default; optionally select one device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for generation on each device",
    )
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    gpu_available = torch.cuda.is_available()
    if args.device == "cuda" and not gpu_available:
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")
    if args.device == "both" and not gpu_available:
        print("ROCm is unavailable; only the CPU run will be performed.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    devices = (
        ["cpu", "cuda"]
        if args.device == "both" and gpu_available
        else (["cpu"] if args.device in ("both", "cpu") else ["cuda"])
    )
    results: dict[str, tuple[int, float, str]] = {}
    print(f"Prompt: {args.prompt}")
    print(f"Input tokens: {len(tokenizer(args.prompt)['input_ids'])}")
    for device in devices:
        results[device] = run_generation(
            args.model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.seed,
            device,
        )

    if "cpu" in results and "cuda" in results:
        cpu_tokens, cpu_rate, _ = results["cpu"]
        gpu_tokens, gpu_rate, _ = results["cuda"]
        print(f"Tokens generated: CPU={cpu_tokens}, iGPU={gpu_tokens}")
        report_comparison(
            "text-generation throughput",
            cpu_rate,
            gpu_rate,
            " tokens/s",
            higher_is_better=True,
        )
    elif args.device == "both" and not gpu_available:
        print("CPU result complete; iGPU comparison unavailable (ROCm not detected).")
    else:
        print(f"CPU/iGPU comparison not run (--device {args.device}).")
    if args.device == "both" and not gpu_available:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
