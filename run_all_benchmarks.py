#!/usr/bin/env python3
"""Run every CPU/iGPU benchmark sequentially and summarize the overall winner.

Usage:
    uv run python run_all_benchmarks.py
    uv run python run_all_benchmarks.py --device cpu
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from benchmark_logs import CHILD_ENV, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(
        run_logged(Path(__file__), sys.argv[1:], "all_benchmarks.txt")
    )

import torch

CHECK_GPU_SECTIONS = {
    "Test 1: Conv2D": "conv2d.txt",
    "Test 2: Transformer operations": "transformer.txt",
    "Test 3: Matmul": "matmul.txt",
    "Copying 256 MiB": "memory_copy.txt",
    "GPU memory": "gpu_memory.txt",
}

COMPARISON_LINE = re.compile(
    r"^CPU vs iGPU \((?P<metric>.+)\): "
    r"CPU=(?P<cpu>[-+0-9.eE]+)(?P<cpu_unit>.*?), "
    r"iGPU=(?P<gpu>[-+0-9.eE]+)(?P<gpu_unit>.*?); "
    r".*?\(iGPU speedup x(?P<speedup>[-+0-9.eE]+)\)$"
)


@dataclass(frozen=True)
class Comparison:
    script: str
    metric: str
    cpu_value: float
    gpu_value: float
    unit: str
    igpu_speedup: float


def parse_comparison_line(line: str, script: str = "") -> Comparison | None:
    match = COMPARISON_LINE.match(line.strip())
    if match is None:
        return None

    cpu_unit = match.group("cpu_unit")
    gpu_unit = match.group("gpu_unit")
    if cpu_unit != gpu_unit:
        raise ValueError(f"CPU/iGPU units differ in comparison output: {line.strip()}")

    return Comparison(
        script=script,
        metric=match.group("metric"),
        cpu_value=float(match.group("cpu")),
        gpu_value=float(match.group("gpu")),
        unit=cpu_unit,
        igpu_speedup=float(match.group("speedup")),
    )


def geometric_mean_speedup(comparisons: list[Comparison]) -> float | None:
    if not comparisons:
        return None
    return math.exp(
        math.fsum(math.log(result.igpu_speedup) for result in comparisons)
        / len(comparisons)
    )


def print_overall_summary(comparisons: list[Comparison]) -> None:
    overall = geometric_mean_speedup(comparisons)
    if overall is None:
        print("No paired CPU/iGPU results were collected; overall speedup is unavailable.")
        return

    epsilon = 1e-6
    igpu_wins = sum(result.igpu_speedup > 1 + epsilon for result in comparisons)
    cpu_wins = sum(result.igpu_speedup < 1 - epsilon for result in comparisons)
    ties = len(comparisons) - igpu_wins - cpu_wins
    print("Per-comparison winners:")
    for result in comparisons:
        if result.igpu_speedup > 1 + epsilon:
            winner, factor = "iGPU", result.igpu_speedup
        elif result.igpu_speedup < 1 - epsilon:
            winner, factor = "CPU", 1 / result.igpu_speedup
        else:
            winner, factor = "tie", 1.0
        print(f"- {result.script}: {result.metric}: {winner} x{factor:.2f}")

    if overall > 1:
        winner, factor = "iGPU", overall
    elif overall < 1:
        winner, factor = "CPU", 1 / overall
    else:
        winner, factor = "CPU and iGPU are tied", 1.0

    print(
        f"Overall: {winner} is faster by x{factor:.2f} "
        f"(geometric mean across {len(comparisons)} comparisons)."
    )
    print(
        f"Individual wins: iGPU={igpu_wins}, CPU={cpu_wins}, ties={ties}. "
        "Each metric has equal weight."
    )


def run_script(
    script: str,
    title: str,
    args: list[str],
    comparisons: list[Comparison],
) -> int:
    script_path = Path(__file__).resolve().parent / script

    def collect_comparison(line: str, is_stdout: bool) -> None:
        if not is_stdout:
            return
        comparison = parse_comparison_line(line, script)
        if comparison is not None:
            comparisons.append(comparison)

    print(f"\n===== {title} =====")
    if script == "check_gpu.py":
        return run_logged(
            script_path,
            args,
            "gpu_environment.txt",
            CHECK_GPU_SECTIONS,
            full_file="check_gpu.txt",
            on_line=collect_comparison,
        )

    log_files = {
        "computer_vision_demo.py": "computer_vision.txt",
        "resnet18_probe.py": "resnet18_probe.txt",
        "sentence_transformers_demo.py": "sentence_transformers.txt",
        "train_demo.py": "training.txt",
        "transformers_demo.py": "transformers.txt",
    }
    return run_logged(
        script_path,
        args,
        log_files[script],
        on_line=collect_comparison,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("both", "cpu", "cuda"),
        default="both",
        help="Run CPU then iGPU by default; optionally select one device",
    )
    args = parser.parse_args()

    gpu_available = torch.cuda.is_available()
    if args.device in ("both", "cuda") and not gpu_available:
        print("ROCm iGPU is unavailable; no CPU/iGPU comparison can be completed.")
        return 1

    print(f"Device mode: {args.device}")
    if args.device == "both":
        print(f"iGPU: {torch.cuda.get_device_name(0)}")
        print("Each test runs CPU first, then iGPU.")

    benchmarks = [
        ("check_gpu.py", "PyTorch operation benchmarks", []),
        ("train_demo.py", "Small CNN training", []),
    ]
    benchmarks.extend(
        [
            ("computer_vision_demo.py", "Computer vision", []),
            ("transformers_demo.py", "Text generation", []),
            ("sentence_transformers_demo.py", "Sentence embeddings", []),
            ("resnet18_probe.py", "ResNet18 probe", []),
        ]
    )

    comparisons: list[Comparison] = []
    failed: list[str] = []
    for script, title, script_args in benchmarks:
        status = run_script(
            script,
            title,
            [*script_args, "--device", args.device],
            comparisons,
        )
        if status:
            failed.append(f"{title} (exit status {status})")
            print("Stopping after a failed benchmark; remaining GPU tests were not run.")
            break

    print("\n===== Overall CPU vs iGPU result =====")
    if args.device == "both":
        print_overall_summary(comparisons)
    else:
        print("Overall speedup requires --device both.")
    if failed:
        print("Incomplete run: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
