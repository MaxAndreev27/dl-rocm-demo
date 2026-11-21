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
from pathlib import Path

from benchmark_logs import CHILD_ENV, run_logged

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
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device; 'cuda' also selects ROCm in PyTorch",
    )
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("--query must not be empty")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 1 <= args.top_k <= len(CORPUS):
        parser.error(f"--top-k must be between 1 and {len(CORPUS)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")
    device = (
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    print(f"Loading {args.model!r} on {device}...")
    model = SentenceTransformer(args.model, device=device)
    corpus_embeddings = model.encode(
        CORPUS,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [args.query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(
        f"Encoded {len(CORPUS)} sentences in batches of {args.batch_size}; "
        f"embedding size: {corpus_embeddings.shape[1]}"
    )
    print(f"\nQuery: {args.query}")
    results = util.semantic_search(
        query_embedding, corpus_embeddings, top_k=args.top_k
    )[0]
    print("Semantic search results:")
    for rank, result in enumerate(results, start=1):
        score = float(result["score"])
        text = CORPUS[int(result["corpus_id"])]
        print(f"{rank}. score={score:.4f}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
