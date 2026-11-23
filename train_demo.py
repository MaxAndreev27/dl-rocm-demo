#!/usr/bin/env python3
"""Train on synthetic data: CPU vs iGPU (ROCm).

Data is generated in memory without downloading datasets. Each class has its
own prototype pattern plus noise, so the model can learn to classify it.

Usage:
    uv run python train_demo.py                 # CPU fp32 vs GPU fp16
    uv run python train_demo.py --epochs 3
    uv run python train_demo.py --device cuda
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from benchmark_logs import CHILD_ENV, report_comparison, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(run_logged(Path(__file__), sys.argv[1:], "training.txt"))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def make_prototypes(num_classes: int = 10, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_classes, 3, 8, 8, generator=g)


def make_data(
    prototypes: torch.Tensor, n: int, seed: int = 0, size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, prototypes.shape[0], (n,), generator=g)
    imgs = F.interpolate(prototypes[labels], size=size, mode="bilinear", align_corners=False)
    imgs = imgs + 0.6 * torch.randn(imgs.shape, generator=g)
    return imgs, labels


def read_kernel_log() -> tuple[list[str] | None, str | None]:
    try:
        result = subprocess.run(
            ["dmesg", "--color=never"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return None, "dmesg is not installed"
    except subprocess.TimeoutExpired:
        return None, "dmesg timed out"
    if result.returncode != 0:
        reason = result.stderr.strip() or f"dmesg exited with status {result.returncode}"
        return None, reason
    return result.stdout.splitlines(), None


def report_gpu_kernel_faults(
    before: list[str] | None,
    before_error: str | None,
    after: list[str] | None,
    after_error: str | None,
) -> bool | None:
    if before is None or after is None:
        reason = before_error or after_error or "kernel log is unavailable"
        print(f"GPU page-fault check: unavailable ({reason})")
        return None

    new_lines = (Counter(after) - Counter(before)).elements()
    fault_pattern = re.compile(
        r"(?:amdgpu|kfd|gpu).*(?:page fault|vm fault|protection fault|fault|reset|timeout|hang)",
        re.IGNORECASE,
    )
    faults = [line for line in new_lines if fault_pattern.search(line)]
    if faults:
        print(f"GPU kernel fault check: FAIL ({len(faults)} new matching messages)")
        for line in faults[:5]:
            print(f"  {line}")
        return False
    print("GPU kernel fault check: no new AMD GPU fault/reset messages")
    return True


@dataclass
class Result:
    name: str
    seconds: float
    loss: float
    accuracy: float
    peak_gib: float


def run(
    device: str,
    amp_dtype: torch.dtype | None,
    train_ds: TensorDataset,
    test_ds: TensorDataset,
    init_state: dict[str, torch.Tensor],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Result:
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    model.load_state_dict(init_state)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    use_amp = device == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model.train()

    if device == "cuda":
        # The first convolution call triggers MIOpen autotuning/JIT. Exclude it
        # from the measurement to avoid making the first epoch look much slower.
        torch.backends.cudnn.benchmark = True
        warm_model = copy.deepcopy(model)
        xb, yb = next(iter(train_loader))
        xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=True):
            warm_loss = criterion(warm_model(xb), yb)
        scaler.scale(warm_loss).backward()
        torch.cuda.synchronize()
        del warm_model
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    model.eval()
    correct = total = 0
    last_loss = 0.0
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if device == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(xb)
                last_loss = float(F.cross_entropy(logits, yb))
            correct += int((logits.argmax(1) == yb).sum())
            total += yb.numel()

    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    name = "GPU fp16" if use_amp else ("GPU fp32" if device == "cuda" else "CPU fp32")
    return Result(name, seconds, last_loss, correct / total, peak)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["both", "cpu", "cuda"], default="both")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/ROCm is unavailable; only the CPU will be used.")
        if args.device == "cuda":
            return 1

    prototypes = make_prototypes(seed=args.seed)
    xtr, ytr = make_data(prototypes, args.train_size, seed=args.seed)
    xte, yte = make_data(prototypes, args.test_size, seed=args.seed + 1)
    train_ds = TensorDataset(xtr, ytr)
    test_ds = TensorDataset(xte, yte)

    base = SmallCNN()
    init_state = copy.deepcopy(base.state_dict())
    params = sum(p.numel() for p in base.parameters())
    print(
        f"train={args.train_size} test={args.test_size} epochs={args.epochs} "
        f"batch={args.batch_size} params={params:,}"
    )
    print(f"torch={torch.__version__} hip={torch.version.hip}\n")

    results: list[Result] = []
    if args.device in ("both", "cpu"):
        results.append(
            run("cpu", None, train_ds, test_ds, init_state, args.epochs, args.batch_size, args.lr, args.seed)
        )
    if args.device in ("both", "cuda") and torch.cuda.is_available():
        results.append(
            run("cuda", torch.float16, train_ds, test_ds, init_state, args.epochs, args.batch_size, args.lr, args.seed)
        )

    print(f"{'run':<10} {'time':>8} {'s/epoch':>9} {'loss':>8} {'acc':>7} {'peak GiB':>9}")
    for r in results:
        print(
            f"{r.name:<10} {r.seconds:7.2f}s {r.seconds / args.epochs:8.2f}s "
            f"{r.loss:8.3f} {r.accuracy * 100:6.1f}% {r.peak_gib:9.2f}"
        )

    if len(results) == 2:
        report_comparison(
            "CNN epoch time",
            results[0].seconds / args.epochs,
            results[1].seconds / args.epochs,
            " s/epoch",
        )
    elif args.device == "both" and not torch.cuda.is_available():
        print("\niGPU comparison unavailable (ROCm not detected).")
    else:
        print(f"\nCPU/iGPU comparison not run (--device {args.device}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
