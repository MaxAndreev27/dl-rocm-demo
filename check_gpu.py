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
import torch.nn.functional as F

N = 4096  # Matrix size for the matmul benchmark.


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_conv2d(
    device: str,
    dtype: torch.dtype,
    batch_size: int = 16,
    image_size: int = 128,
    in_channels: int = 3,
    out_channels: int = 32,
    kernel_size: int = 3,
    iters: int = 20,
    warmup: int = 5,
) -> tuple[float, float]:
    """Return seconds per forward pass and GFLOPS for a Conv2D."""
    x = torch.randn(
        batch_size, in_channels, image_size, image_size, device=device, dtype=dtype
    )
    weight = torch.randn(
        out_channels, in_channels, kernel_size, kernel_size, device=device, dtype=dtype
    )
    if device == "cuda":
        x = x.to(memory_format=torch.channels_last)
        weight = weight.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    for _ in range(warmup):
        F.conv2d(x, weight, padding=kernel_size // 2)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        F.conv2d(x, weight, padding=kernel_size // 2)
    sync(device)

    dt = (time.perf_counter() - t0) / iters
    output_size = image_size + 2 * (kernel_size // 2) - kernel_size + 1
    flops = 2 * batch_size * output_size**2 * out_channels * in_channels * kernel_size**2
    gflops = flops / dt / 1e9
    del x, weight
    if device == "cuda":
        torch.cuda.empty_cache()
    return dt, gflops


def bench_transformer_ops(
    device: str,
    dtype: torch.dtype,
    batch_size: int = 4,
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 64,
    iters: int = 10,
    warmup: int = 2,
) -> tuple[float, float, float]:
    """Benchmark attention forward/backward and verify softmax and gradients."""
    shape = (batch_size, num_heads, seq_len, head_dim)
    q, k, v = (
        torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
        for _ in range(3)
    )

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q.grad = k.grad = v.grad = None
        scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        loss = output.float().square().mean()
        loss.backward()
        return weights, output, loss

    for _ in range(warmup):
        step()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        weights, output, loss = step()
    sync(device)
    dt = (time.perf_counter() - t0) / iters

    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    if output.shape != expected_shape:
        raise RuntimeError(f"Attention output has unexpected shape: {output.shape}")
    if not torch.isfinite(output).all().item() or not torch.isfinite(weights).all().item():
        raise RuntimeError("Attention output or Softmax contains non-finite values")
    row_sums = weights.float().sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2, rtol=1e-2):
        raise RuntimeError("Softmax rows do not sum to 1")
    grads = (q.grad, k.grad, v.grad)
    if any(grad is None or not torch.isfinite(grad).all().item() for grad in grads):
        raise RuntimeError("Attention backward produced missing or non-finite gradients")
    grad_magnitude = sum(grad.float().abs().sum() for grad in grads if grad is not None)
    if grad_magnitude.item() <= 0:
        raise RuntimeError("Attention backward produced zero gradients")

    flops = 12 * batch_size * num_heads * seq_len**2 * head_dim
    gflops = flops / dt / 1e9
    return dt, gflops, float(loss.item())


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
    print("Test 1: Conv2D forward, 16x3x128x128 -> 32 channels, 3x3")
    print(line)
    conv_rows = [
        ("CPU fp32", *bench_conv2d("cpu", torch.float32)),
        ("GPU fp16", *bench_conv2d("cuda", torch.float16)),
    ]
    for name, dt, gflops in conv_rows:
        print(f"{name:<10} {dt * 1e3:9.2f} ms   {gflops:9.1f} GFLOPS")
    cpu_conv = next(g for n, _, g in conv_rows if n == "CPU fp32")
    gpu_conv = next(g for n, _, g in conv_rows if n == "GPU fp16")
    print(f"GPU fp16 speedup vs CPU fp32: x{gpu_conv / cpu_conv:.2f}")

    print("\n" + line)
    print("Test 2: Transformer operations — attention, Softmax, forward + backward")
    print(line)
    transformer_rows = [
        ("CPU fp32", *bench_transformer_ops("cpu", torch.float32)),
        ("GPU fp16", *bench_transformer_ops("cuda", torch.float16)),
    ]
    for name, dt, gflops, loss in transformer_rows:
        print(
            f"{name:<10} {dt * 1e3:9.2f} ms   {gflops:9.1f} GFLOPS "
            f"loss={loss:.5f} checks=ok"
        )
    cpu_transformer = next(g for n, _, g, _ in transformer_rows if n == "CPU fp32")
    gpu_transformer = next(g for n, _, g, _ in transformer_rows if n == "GPU fp16")
    print(f"GPU fp16 speedup vs CPU fp32: x{gpu_transformer / cpu_transformer:.2f}")

    print("\nTest 3 (long-run stability): uv run python train_demo.py --long-test")
    print("\n" + line)
    print(f"Test 4: Matmul {N}x{N}, FP16/BF16 (higher GFLOPS is better)")
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
