#!/usr/bin/env python3
"""Run one small FP32 ResNet18 training step on synthetic images."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from benchmark_logs import CHILD_ENV, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(run_logged(Path(__file__), sys.argv[1:], "resnet18_probe.txt"))

import torch
from torch import nn
from torchvision.models import resnet18

from train_demo import read_kernel_log, report_gpu_kernel_faults


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to use; cuda selects the ROCm GPU (default: cpu)",
    )
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")

    baseline_free = None
    before = before_error = None
    if device == "cuda":
        print("Warning: even a small GPU workload can hang the system.")
        torch.cuda.synchronize()
        baseline_free, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        before, before_error = read_kernel_log()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Free GPU memory before test: {baseline_free / 2**30:.2f}/{total / 2**30:.2f} GiB")

    torch.manual_seed(42)
    batch_size = 4
    image_size = 64
    print(
        f"ResNet18 probe: device={device}, fp32, one training step, "
        f"batch={batch_size}, image={image_size}x{image_size}"
    )
    model = resnet18(weights=None, num_classes=10).to(device)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(images), labels)
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite training loss: {loss.item()}")
    loss.backward()
    nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf"), error_if_nonfinite=True
    )
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Training step completed: loss={loss.item():.4f}, time={elapsed:.2f}s")

    if device == "cuda":
        assert baseline_free is not None
        free, total = torch.cuda.mem_get_info()
        print(
            f"GPU free after step: {free / 2**30:.2f}/{total / 2**30:.2f} GiB; "
            f"free-memory change: {(baseline_free - free) / 2**30:+.2f} GiB"
        )
        print(
            f"PyTorch peak allocated/reserved: "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f}/"
            f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
        )
        after, after_error = read_kernel_log()
        if report_gpu_kernel_faults(before, before_error, after, after_error) is False:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
