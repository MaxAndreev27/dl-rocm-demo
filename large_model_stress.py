#!/usr/bin/env python3
"""Run a short ResNet18 training test; opt in explicitly to ROCm GPU use."""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
from pathlib import Path

from benchmark_logs import CHILD_ENV, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(run_logged(Path(__file__), sys.argv[1:], "large_model_stress.txt"))

import torch
from torch import nn
from torchvision.models import resnet18

from train_demo import (
    read_kernel_log,
    report_gpu_kernel_faults,
    sample_telemetry,
)

GIB = 2**30


def make_batch(
    batch_size: int, image_size: int, num_classes: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(batch_size, 3, image_size, image_size, generator=generator)
    labels = torch.randint(0, num_classes, (batch_size,), generator=generator)
    return images, labels


def run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images), labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Training loss became non-finite: {loss.item()}")
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
    )
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return float(loss.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--duration-minutes", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=float, default=10.0, help="Seconds")
    parser.add_argument(
        "--target-gtt-gib",
        type=float,
        default=None,
        help="Optional minimum sampled free-memory reduction required for success",
    )
    parser.add_argument("--max-gtt-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA/ROCm is unavailable; the large-model GPU stress test cannot run.")
        return 1
    if not math.isfinite(args.duration_minutes) or args.duration_minutes <= 0:
        parser.error("--duration-minutes must be a positive finite number")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.warmup_steps <= 0:
        parser.error("--warmup-steps must be positive")
    if not math.isfinite(args.log_interval) or args.log_interval <= 0:
        parser.error("--log-interval must be a positive finite number")
    if not math.isfinite(args.max_gtt_gib) or args.max_gtt_gib <= 0:
        parser.error("--max-gtt-gib must be positive and finite")
    if args.target_gtt_gib is not None:
        if (
            not math.isfinite(args.target_gtt_gib)
            or args.target_gtt_gib <= 0
            or args.target_gtt_gib >= args.max_gtt_gib
        ):
            parser.error(
                "--target-gtt-gib must be positive and less than --max-gtt-gib"
            )
        if args.device != "cuda":
            parser.error("--target-gtt-gib requires --device cuda")

    torch.manual_seed(args.seed)
    baseline_free_gib = 0.0
    before = before_error = None
    if args.device == "cuda":
        print(
            "Warning: GPU workloads can hang the system; save your work before running."
        )
        torch.cuda.synchronize()
        baseline_free, total_memory = torch.cuda.mem_get_info()
        baseline_free_gib = baseline_free / GIB
        if baseline_free_gib < 1.0:
            print(
                f"Insufficient free GPU memory: {baseline_free_gib:.2f} GiB (need 1 GiB)."
            )
            return 1
        torch.cuda.reset_peak_memory_stats()
        before, before_error = read_kernel_log()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Baseline free GPU memory: {baseline_free_gib:.2f}/{total_memory / GIB:.2f} GiB"
        )
        print(
            "Free-memory changes include other GPU processes and are sampled between steps."
        )

    num_classes = 10
    images, labels = make_batch(
        args.batch_size, args.image_size, num_classes, args.seed
    )
    images = images.to(args.device)
    labels = labels.to(args.device)
    model = resnet18(weights=None, num_classes=num_classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"ResNet18 training | device={args.device} "
        f"precision=fp32 image={args.image_size}x{args.image_size} "
        f"batch={args.batch_size} params={params:,}"
    )
    print(
        f"Duration={args.duration_minutes:.2f} min after {args.warmup_steps} warm-up steps"
    )
    model.train()
    samples = []
    max_limit_reached = False
    step_times: list[float] = []
    losses: list[float] = []
    elapsed = 0.0
    start = None
    next_report = args.log_interval
    steps = 0
    while start is None or elapsed < args.duration_minutes * 60:
        if steps == args.warmup_steps and start is None:
            start = time.perf_counter()
        step_start = time.perf_counter()
        loss = run_step(model, optimizer, criterion, images, labels, args.device)
        step_time = time.perf_counter() - step_start
        steps += 1
        if start is not None:
            losses.append(loss)
            step_times.append(step_time)
            elapsed = time.perf_counter() - start
        if args.device == "cuda":
            sample = sample_telemetry()
            samples.append(sample)
            used = max(0.0, baseline_free_gib - sample.gpu_free)
            if used >= args.max_gtt_gib:
                max_limit_reached = True
            if max_limit_reached or (start is not None and elapsed >= next_report):
                print(
                    f"elapsed={elapsed:.1f}s steps={len(step_times)} "
                    f"loss={loss:.4f} sampled GTT delta={used:.2f}GiB "
                    f"free={sample.gpu_free:.2f}/{sample.gpu_total:.2f}GiB "
                    f"alloc/res={sample.gpu_allocated:.2f}/"
                    f"{sample.gpu_reserved:.2f}GiB"
                )
                next_report = elapsed + args.log_interval
            if max_limit_reached:
                print("Sampled GTT limit reached; stopping before the next step.")
                break

    target_reached = False
    gpu_fault_check = None
    if args.device == "cuda":
        after, after_error = read_kernel_log()
        gpu_fault_check = report_gpu_kernel_faults(
            before, before_error, after, after_error
        )
        sampled_delta = max(
            (max(0.0, baseline_free_gib - sample.gpu_free) for sample in samples),
            default=0.0,
        )
        print(
            f"Maximum sampled free-memory reduction={sampled_delta:.2f} GiB; "
            f"PyTorch peak allocated/reserved="
            f"{torch.cuda.max_memory_allocated() / GIB:.2f}/"
            f"{torch.cuda.max_memory_reserved() / GIB:.2f} GiB"
        )
        if args.target_gtt_gib is None:
            print("No GTT target specified; reaching 2-4 GiB was not verified.")
        else:
            target_reached = sampled_delta >= args.target_gtt_gib
            print(f"GTT target: {'reached' if target_reached else 'not reached'}")
    if step_times:
        median_step = statistics.median(step_times)
        throughput = args.batch_size * len(step_times) / max(elapsed, 1e-9)
        slowdown = step_times[-1] / step_times[0]
        print(
            f"Training completed={elapsed:.2f}s steps={len(step_times)} "
            f"median step={median_step:.3f}s throughput={throughput:.2f} images/s "
            f"last/first step={slowdown:.2f}x "
            f"mean loss={statistics.fmean(losses):.4f}"
        )
    else:
        print("Timed phase did not start because the sampled GTT limit was reached.")

    if gpu_fault_check is False or max_limit_reached:
        return 2
    if args.target_gtt_gib is not None and not target_reached:
        print("Configured GTT target was not reached.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
