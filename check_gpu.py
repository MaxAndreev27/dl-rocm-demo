#!/usr/bin/env python3
"""Check a local ROCm stack (gfx1103 / Radeon 780M) and benchmark CPU vs iGPU.

Usage:
    uv run python check_gpu.py

This script does not install software or change system settings. ROCm libraries
are kept in .venv (rocm-sdk-*); the system-wide /opt/rocm is not used.
"""

from __future__ import annotations

import os
import platform
import sys
import time

import torch

N = 4096  # Matrix size for the matmul benchmark.


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_matmul(
    device: str,
    dtype: torch.dtype,
    n: int = N,
    iters: int = 20,
    warmup: int = 5,
) -> tuple[float, float]:
    """Return seconds per iteration and GFLOPS."""
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(warmup):
        a @ b
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(device)

    dt = (time.perf_counter() - t0) / iters
    gflops = 2 * n**3 / dt / 1e9
    del a, b
    if device == "cuda":
        torch.cuda.empty_cache()
    return dt, gflops


def bench_copy(device: str, mib: int = 256, iters: int = 10) -> float:
    """Return copy bandwidth in GB/s (read + write)."""
    n = mib * 1024 * 1024 // 4  # float32
    src = torch.empty(n, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(2):
        dst.copy_(src)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src)
    sync(device)

    dt = time.perf_counter() - t0
    gb = 2 * src.numel() * src.element_size() * iters / 1e9
    del src, dst
    if device == "cuda":
        torch.cuda.empty_cache()
    return gb / dt


def main() -> int:
    line = "=" * 72
    print(line)
    print("ROCm / PyTorch check — local venv, no system-wide ROCm")
    print(line)
    print(f"python       : {platform.python_version()} ({platform.machine()})")
    print(f"torch        : {torch.__version__}")
    print(f"hip runtime  : {torch.version.hip}")
    print(f"cuda available: {torch.cuda.is_available()}")

    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        print(f"HSA_OVERRIDE : {override}  <-- UNNECESSARY for gfx1103")
    else:
        print("HSA_OVERRIDE : not set (correct)")

    if not torch.cuda.is_available():
        print("\nGPU not detected. Check /dev/kfd, render/video groups, and rocm-sdk-* wheels.")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"device       : {torch.cuda.get_device_name(0)}")
    print(f"arch         : {props.gcnArchName}")
    print(f"memory (GTT) : {props.total_memory / 2**30:.2f} GiB")
    print(f"CU (reported): {props.multi_processor_count}")

    # Quickly verify that kernels can actually run.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    sanity = (x @ x).float().mean().item()
    del x
    torch.cuda.empty_cache()
    print(f"matmul sanity: {sanity:.4f}  (ok)")

    print("\n" + line)
    print(f"Matmul {N}x{N}, FP16/BF16 (higher GFLOPS is better)")
    print(line)
    rows: list[tuple[str, float, float]] = []
    rows.append(("CPU fp32", *bench_matmul("cpu", torch.float32)))
    try:
        rows.append(("CPU bf16", *bench_matmul("cpu", torch.bfloat16)))
    except Exception as exc:  # noqa: BLE001
        print(f"CPU bf16 skipped: {exc}")
    try:
        rows.append(("GPU fp16", *bench_matmul("cuda", torch.float16)))
        rows.append(("GPU bf16", *bench_matmul("cuda", torch.bfloat16)))
    except Exception as exc:  # noqa: BLE001
        print(f"\nGPU benchmark failed: {exc}")
        print("Restart the script. If the problem persists, check dmesg and GPU temperature.")
        return 2

    for name, dt, gflops in rows:
        print(f"{name:<10} {dt * 1e3:9.2f} ms   {gflops:9.1f} GFLOPS")

    gpu_fp16 = next(g for n, _, g in rows if n == "GPU fp16")
    cpu_fp32 = next(g for n, _, g in rows if n == "CPU fp32")
    print(f"\nGPU fp16 speedup vs CPU fp32: x{gpu_fp16 / cpu_fp32:.2f}")

    print("\n" + line)
    print("Copying 256 MiB (rough estimate of memory bandwidth)")
    print(line)
    cpu_bw = bench_copy("cpu")
    gpu_bw = bench_copy("cuda")
    print(f"CPU: {cpu_bw:7.1f} GB/s")
    print(f"GPU: {gpu_bw:7.1f} GB/s  (GTT, shared DDR5)")

    print("\n" + line)
    print("GPU memory")
    print(line)
    free, total = torch.cuda.mem_get_info()
    print(f"free/total : {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    print(f"allocated  : {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print(f"reserved   : {torch.cuda.memory_reserved() / 2**30:.2f} GiB")

    print("\nOK: iGPU is available to PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
