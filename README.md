# dl-rocm-demo

A local PyTorch and ROCm 10.2 demo for the Radeon 780M (`gfx1103`), without a system-wide ROCm installation.

## What this project does

The project keeps the ROCm runtime and PyTorch in its local `.venv`:

- `.venv/` contains Python 3.13, `torch 2.14.0+rocm10.2.0`, `torchvision 0.29.0+rocm10.2.0`, and the `rocm-sdk-*` runtime packages.
- `/opt/rocm` is not used or installed.
- The system uses the standard `amdgpu` kernel driver.

The scripts check the GPU environment and compare CPU and iGPU performance. The notebook provides an interactive version of the demo.

## Requirements

- Linux with an AMD GPU supported by the installed ROCm/PyTorch wheels. This project was tested on a Radeon 780M (`gfx1103`).
- Python 3.13 and [uv](https://docs.astral.sh/uv/).
- Access to `/dev/kfd` and the required GPU device groups (commonly `render` and `video`).

The pinned ROCm/PyTorch versions target this specific setup and may not work on other GPU architectures.

## Setup and usage

```bash
git clone <repository-url>
cd dl-rocm-demo
uv sync

# Check the environment and benchmark the CPU and iGPU.
uv run python check_gpu.py

# Train a small CNN: compare CPU fp32 with GPU fp16.
uv run python train_demo.py
uv run python train_demo.py --epochs 3 --device cuda

# Run Test 3: 100 epochs, about 2.5 minutes on the tested Radeon 780M.
# Actual duration varies with GPU clocks, thermals, and background load.
timeout 30m uv run python train_demo.py --long-test

# Start JupyterLab, if installed in the environment.
uv run jupyter lab
```

Alternatively, open `demo.ipynb` in VS Code or Cursor and select `.venv/bin/python` as the notebook kernel.

Test 3 trains the GPU for 100 epochs by default (select 20–100 with `--long-test-epochs`). On the tested Radeon 780M, the estimate is about 2.5 minutes, extrapolated from the measured 25-epoch run; actual duration varies with GPU clocks, thermals, and background load. It samples temperature, GPU clock, RAM, and GPU memory several times during each epoch and reports the observed peak temperatures/clock plus epoch speed. Each epoch synchronizes with the GPU, so a completed run confirms all epochs finished. New AMD GPU fault/reset messages are checked in `dmesg`; if kernel-log access is restricted, the test reports that check as unavailable rather than claiming no faults occurred. `timeout` provides a process-level guard against a hung run.

## Example results

The following measurements were recorded on a Radeon 780M (`gfx1103`):

| Test | Result |
|---|---|
| `torch.cuda.is_available()` | `True` |
| Device | `AMD Radeon 780M Graphics`, `gfx1103` |
| GTT memory | 7.44 GiB |
| 4096x4096 matmul, CPU fp32 | ~0.76–0.80 TFLOPS |
| 4096x4096 matmul, CPU bf16 | ~1.6–1.7 TFLOPS |
| 4096x4096 matmul, GPU fp16 | ~4.7–6.6 TFLOPS (varies with power and clock speeds) |
| Conv2D `16x3x128x128` → 32 channels, CPU fp32 | ~42 GFLOPS |
| Conv2D `16x3x128x128` → 32 channels, GPU fp16 | ~576 GFLOPS (~13.5x vs CPU fp32) |
| Attention forward/backward (`B=4, H=8, T=512, D=64`), CPU fp32 | ~23 GFLOPS |
| Attention forward/backward (`B=4, H=8, T=512, D=64`), GPU fp16 | ~766 GFLOPS (~32.9x vs CPU fp32) |
| 256 MiB copy, CPU | ~41 GB/s |
| 256 MiB copy, GPU (GTT) | ~75–77 GB/s |
| Small CNN training | CPU 0.87 s/epoch, GPU 0.38 s/epoch (~2.3x) |

Results depend on power limits, cooling, and background load. On an APU, the GPU can be much faster for large matrix multiplications and convolutions, while small models may see a smaller improvement or run faster on the CPU.

`check_gpu.py` runs **Test 1: Conv2D** and **Test 2: Transformer operations** (scaled dot-product attention, Softmax, and backward pass) on CPU fp32 vs GPU fp16. **Test 3** is the optional long-duration GPU training run above. The transformer test checks output shape, Softmax normalization, and finite nonzero gradients. Its GFLOPS estimate counts attention matrix multiplications in forward and backward; Softmax operations are not included. The interactive notebook contains all three tests.

## Notes and troubleshooting

- The first GPU convolution may trigger MIOpen autotuning and JIT compilation and take tens of seconds. The warm-up is excluded from the measurements in `train_demo.py` and the notebook.
- A MIOpen warning about a missing `libMIOpenCKGroupedConv_gfx1103.so` is not necessarily an error. Some algorithms for `gfx1103` are unavailable in the pip build, so MIOpen falls back to another algorithm.
- Do not set `HSA_OVERRIDE_GFX_VERSION` for `gfx1103`; it is not needed and may cause failures.
- GPU memory is GTT (shared DDR5). If another workload such as Ollama is using memory, stop it before training, for example with `sudo systemctl stop ollama`, and restart it afterward with `sudo systemctl start ollama`.
- If memory fragmentation is an issue, try:

  ```bash
  export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
  ```

- For CNN workloads, `channels_last` and AMP (for example, `torch.autocast("cuda", dtype=torch.float16)`) may improve performance.
- If the GPU is not detected, check `/dev/kfd`, your `render`/`video` group membership, and the installed `rocm-sdk-*` wheels.

## Reproducible environment

Run `uv sync` to create the environment. AMD package indexes and the resolver strategy are configured in `pyproject.toml`; package versions are pinned in `uv.lock`. Add standard packages with `uv add`, for example `uv add pandas`.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contribution process and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for community expectations.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
