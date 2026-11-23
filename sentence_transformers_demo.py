#!/usr/bin/env python3
"""Batch sentence embeddings and semantic search on CPU or ROCm GPU.

Usage:
    uv run python sentence_transformers_demo.py
    uv run python sentence_transformers_demo.py --query "How do I find similar text?"
    uv run python sentence_transformers_demo.py --device cpu --batch-size 2
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
        run_logged(Path(__file__), sys.argv[1:], "sentence_transformers.txt")
    )

import torch
from sentence_transformers import SentenceTransformer, util

CORPUS = [
    "ROCm enables PyTorch workloads on AMD GPUs.",
    "Sentence Transformers encode sentences as dense vector embeddings.",
    "Text generation models continue writing from a prompt.",
    "Batch inference processes multiple sentences in one model call.",
    "Semantic search finds passages by meaning rather than exact keywords.",
    "PyTorch is an open-source framework for deep learning.",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face sentence embedding model",
    )
    parser.add_argument(
        "--query",
        default="How can I find related documents by meaning?",
        help="Text to search for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of corpus sentences encoded at once (default: 3)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=f"Number of search results (1-{len(CORPUS)}, default: 3)",
    )
    parser.add_argument(
        "--device",
        choices=("both", "cpu", "cuda"),
        default="both",
        help="Compare CPU then iGPU by default; optionally select one device",
    )
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("--query must not be empty")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.top_k <= len(CORPUS):
        parser.error(f"--top-k must be between 1 and {len(CORPUS)}")
    gpu_available = torch.cuda.is_available()
    if args.device == "cuda" and not gpu_available:
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")
    if args.device == "both" and not gpu_available:
        print("ROCm is unavailable; only the CPU run will be performed.")
    devices = (
        ["cpu", "cuda"]
        if args.device == "both" and gpu_available
        else (["cpu"] if args.device in ("both", "cpu") else ["cuda"])
    )

    throughput: dict[str, float] = {}
    print(f"Query: {args.query}")
    for device in devices:
        label = "iGPU" if device == "cuda" else "CPU"
        print(f"\nLoading {args.model!r} on {label}...")
        model = SentenceTransformer(args.model, device=device)
        model.encode(
            CORPUS[:1],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        corpus_embeddings = model.encode(
            CORPUS,
            batch_size=args.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = model.encode(
            [args.query],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        results = util.semantic_search(
            query_embedding, corpus_embeddings, top_k=args.top_k
        )[0]
        print(
            f"{label}: encoded {len(CORPUS) + 1} sentences in {elapsed:.3f}s "
            f"({(len(CORPUS) + 1) / elapsed:.2f} sentences/s); "
            f"embedding size={corpus_embeddings.shape[1]}"
        )
        print("Semantic search results:")
        for rank, result in enumerate(results, start=1):
            score = float(result["score"])
            text = CORPUS[int(result["corpus_id"])]
            print(f"{rank}. score={score:.4f}  {text}")
        throughput[device] = (len(CORPUS) + 1) / elapsed
        del model, corpus_embeddings, query_embedding
        if device == "cuda":
            torch.cuda.empty_cache()

    if "cpu" in throughput and "cuda" in throughput:
        report_comparison(
            "sentence-embedding throughput",
            throughput["cpu"],
            throughput["cuda"],
            " sentences/s",
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
