#!/usr/bin/env python3
"""Train and evaluate ResNet18 and MobileNetV3-Small on synthetic images.

Usage:
    uv run python computer_vision_demo.py
    uv run python computer_vision_demo.py --device cuda --epochs 2
    uv run python computer_vision_demo.py --device cpu --train-size 64
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

from benchmark_logs import CHILD_ENV, report_comparison, run_logged

if __name__ == "__main__" and os.environ.get(CHILD_ENV) != "1":
    raise SystemExit(
        run_logged(Path(__file__), sys.argv[1:], "computer_vision.txt")
    )

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models


def make_prototypes(num_classes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_classes, 3, 8, 8, generator=generator)


def make_data(
    prototypes: torch.Tensor, size: int, image_size: int, seed: int
) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, prototypes.shape[0], (size,), generator=generator)
    images = F.interpolate(
        prototypes[labels],
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    images += 0.6 * torch.randn(images.shape, generator=generator)
    return TensorDataset(images, labels)


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    test_size: int,
    epochs: int,
    device: str,
) -> tuple[float, float, float, float, float]:
    model = model.to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loss = 0.0
    train_samples = 0

    model.train()
    if device == "cuda":
        warm_model = copy.deepcopy(model)
        warm_images, warm_labels = next(
            iter(DataLoader(train_loader.dataset, batch_size=train_loader.batch_size))
        )
        warm_images = warm_images.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
        warm_labels = warm_labels.to(device, non_blocking=True)
        criterion(warm_model(warm_images), warm_labels).backward()
        torch.cuda.synchronize()
        del warm_model
        torch.cuda.empty_cache()
    sync(device)
    train_start = time.perf_counter()
    for _ in range(epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device == "cuda":
                images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().float().item() * labels.size(0)
            train_samples += labels.size(0)
    sync(device)
    train_seconds = time.perf_counter() - train_start

    model.eval()
    correct = 0
    test_loss = 0.0
    warm_images, _ = next(iter(test_loader))
    warm_images = warm_images.to(device, non_blocking=True)
    if device == "cuda":
        warm_images = warm_images.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        model(warm_images)
    sync(device)
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device == "cuda":
                images = images.to(memory_format=torch.channels_last)
            logits = model(images)
            loss = criterion(logits, labels)
            test_loss += loss.float().item() * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
    sync(device)
    inference_seconds = time.perf_counter() - inference_start

    return (
        train_seconds,
        inference_seconds * 1000 / test_size,
        train_loss / train_samples,
        test_loss / test_size,
        correct / test_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("both", "cpu", "cuda"),
        default="both",
        help="Bench CPU then iGPU by default; optionally select one device",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but no CUDA/ROCm device is available")
    for name in ("epochs", "train_size", "test_size", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")

    gpu_available = torch.cuda.is_available()
    if args.device == "both" and not gpu_available:
        print("ROCm is unavailable; only the CPU run will be performed.")
    devices = (
        ["cpu", "cuda"]
        if args.device == "both" and gpu_available
        else (["cpu"] if args.device in ("both", "cpu") else ["cuda"])
    )
    num_classes = 10
    prototypes = make_prototypes(num_classes, args.seed)
    train_data = make_data(prototypes, args.train_size, args.image_size, args.seed)
    test_data = make_data(prototypes, args.test_size, args.image_size, args.seed + 1)
    model_factories = (
        ("ResNet18", lambda: models.resnet18(weights=None, num_classes=num_classes)),
        (
            "MobileNetV3-Small",
            lambda: models.mobilenet_v3_small(weights=None, num_classes=num_classes),
        ),
    )
    print("Computer Vision training/inference | precision=fp32")
    print(
        f"images={args.image_size}x{args.image_size} train={args.train_size} "
        f"test={args.test_size} batch={args.batch_size} epochs={args.epochs}"
    )
    print("Dataset: synthetic class prototypes with noise; pretrained weights disabled.")

    for name, create_model in model_factories:
        results: dict[str, tuple[float, float]] = {}
        params = 0
        for device in devices:
            torch.manual_seed(args.seed)
            model = create_model()
            params = sum(parameter.numel() for parameter in model.parameters())
            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            test_loader = DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False
            )
            (
                train_seconds,
                inference_ms,
                train_loss,
                loss,
                accuracy,
            ) = run_model(
                model,
                train_loader,
                test_loader,
                args.test_size,
                args.epochs,
                device,
            )
            label = "iGPU" if device == "cuda" else "CPU"
            print(
                f"{name} {label}: params={params:,} train={train_seconds:.2f}s "
                f"train_loss={train_loss:.4f} "
                f"inference={inference_ms:.2f}ms/image "
                f"test_loss={loss:.4f} accuracy={accuracy * 100:.1f}%"
            )
            results[device] = (train_seconds, inference_ms)
            del model

        if "cpu" in results and "cuda" in results:
            report_comparison(
                f"{name} training time",
                results["cpu"][0],
                results["cuda"][0],
                " s",
            )
            report_comparison(
                f"{name} inference latency",
                results["cpu"][1],
                results["cuda"][1],
                " ms/image",
            )
        else:
            if args.device == "both" and not gpu_available:
                print(f"{name}: iGPU comparison unavailable (ROCm not detected).")
            else:
                print(f"{name}: CPU/iGPU comparison not run (--device {args.device}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
