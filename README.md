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

# Start JupyterLab, if installed in the environment.
uv run jupyter lab
```

Alternatively, open `demo.ipynb` in VS Code or Cursor and select `.venv/bin/python` as the notebook kernel.

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
| 256 MiB copy, CPU | ~41 GB/s |
| 256 MiB copy, GPU (GTT) | ~75–77 GB/s |
| Small CNN training | CPU 0.87 s/epoch, GPU 0.38 s/epoch (~2.3x) |

Results depend on power limits, cooling, and background load. On an APU, the GPU can be much faster for large matrix multiplications and convolutions, while small models may see a smaller improvement or run faster on the CPU.

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
