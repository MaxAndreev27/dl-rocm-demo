#!/usr/bin/env python3
"""Compare one small FP32 ResNet18 training step on CPU and ROCm iGPU."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

from benchmark_logs import CHILD_ENV, report_comparison, run_logged

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
        choices=("both", "cpu", "cuda"),
        default="both",
        help="Bench CPU then iGPU by default; optionally select one device",
    )
    args = parser.parse_args()
    gpu_available = torch.cuda.is_available()
    if args.device == "cuda" and not gpu_available:
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")

    if args.device == "both" and not gpu_available:
        print("ROCm is unavailable; only the CPU run will be performed.")

    torch.manual_seed(42)
    batch_size = 4
    image_size = 64
    print(
        "ResNet18 probe: fp32, one training step, "
        f"batch={batch_size}, image={image_size}x{image_size}"
    )
    base_model = resnet18(weights=None, num_classes=10)
    initial_state = copy.deepcopy(base_model.state_dict())
    del base_model
    images = torch.randn(batch_size, 3, image_size, image_size)
    labels = torch.tensor([0, 1, 2, 3])
    devices = (
        ["cpu", "cuda"]
        if args.device == "both" and gpu_available
        else (["cpu"] if args.device in ("both", "cpu") else ["cuda"])
    )
    results: dict[str, float] = {}
    gpu_fault_check: bool | None = True

    for device in devices:
        baseline_free = None
        before = before_error = None
        if device == "cuda":
            print("Warning: even a small GPU workload can hang the system.")
            torch.cuda.synchronize()
            baseline_free, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            before, before_error = read_kernel_log()
            print(f"iGPU: {torch.cuda.get_device_name(0)}")
            print(
                f"Free GPU memory before test: "
                f"{baseline_free / 2**30:.2f}/{total / 2**30:.2f} GiB"
            )

        model = resnet18(weights=None, num_classes=10).to(device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device_images = images.to(device)
        device_labels = labels.to(device)
        model.train()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(device_images), device_labels)
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
        label = "iGPU" if device == "cuda" else "CPU"
        print(f"{label} training step completed: loss={loss.item():.4f}, time={elapsed:.3f}s")
        results[device] = elapsed

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
            gpu_fault_check = report_gpu_kernel_faults(
                before, before_error, after, after_error
            )
        del model, optimizer, device_images, device_labels

    if "cpu" in results and "cuda" in results:
        report_comparison(
            "ResNet18 training step", results["cpu"], results["cuda"], " s"
        )
    elif args.device == "both" and not gpu_available:
        print("CPU result complete; iGPU comparison unavailable (ROCm not detected).")
    else:
        print(f"CPU/iGPU comparison not run (--device {args.device}).")
    if args.device == "both" and not gpu_available:
        return 1
    if gpu_fault_check is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
