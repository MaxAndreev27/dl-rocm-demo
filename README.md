# dl-rocm-demo

A local PyTorch and ROCm 10.2 demo for the Radeon 780M (`gfx1103`), without a system-wide ROCm installation.

## What this project does

The project keeps the ROCm runtime and PyTorch in its local `.venv`:

- `.venv/` uses Python 3.13 and contains `torch 2.14.0+rocm10.2.0`, `torchvision 0.29.0+rocm10.2.0`, and the `rocm-sdk-*` runtime packages. Its Python executable may link to an interpreter outside the project.
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
git clone https://github.com/MaxAndreev27/dl-rocm-demo.git
cd dl-rocm-demo
uv sync

# Check the environment and benchmark the CPU and iGPU.
uv run python check_gpu.py

# Run all CPU/iGPU benchmarks sequentially and print the overall result.
uv run python run_all_benchmarks.py
# Train a small CNN: compare CPU first, then iGPU.
uv run python train_demo.py
uv run python train_demo.py --epochs 3
uv run python train_demo.py --device cpu  # Run only on CPU.

# Train and evaluate ResNet18 and MobileNetV3-Small on synthetic images.
uv run python computer_vision_demo.py
uv run python computer_vision_demo.py --epochs 2

# Generate text on CPU first, then iGPU.
uv run python transformers_demo.py
uv run python transformers_demo.py --prompt "The future of AI is" --max-new-tokens 40

# Batch sentence embeddings and search the sample corpus by meaning.
uv run python sentence_transformers_demo.py
uv run python sentence_transformers_demo.py --query "How do I find related text?"

# Start JupyterLab with an additional temporary dependency.
uv run --with jupyterlab jupyter lab
```

Alternatively, open `demo.ipynb` in VS Code or Cursor and select `.venv/bin/python` as the notebook kernel.

By default, benchmark scripts run the CPU first and the ROCm iGPU second on the same workload, then report both measurements and the speedup (including when the CPU is faster). Use `--device cpu` or `--device cuda` to run only one side; `cuda` is PyTorch's device name for ROCm. If ROCm is unavailable, the CPU result is still recorded and the comparison is marked unavailable.

Some throughput comparisons intentionally use CPU fp32 and iGPU fp16; their reported speedup compares those configured modes, not identical precision. Other demos use fp32 on both devices. **GPU runs have previously caused launch failures and system hangs**; save your work before starting the default comparison. `--device cpu` avoids iGPU execution, and a timeout cannot guarantee recovery from a GPU or driver hang.

`run_all_benchmarks.py` runs every benchmark in CPU-then-iGPU order, then reports individual wins and an overall geometric-mean speedup (each measured comparison has equal weight). It stops if a benchmark fails; `--device cpu` or `--device cuda` runs only one side without an overall comparison.

Command-line runs display and save output and comparison results in `benchmark_results/`: the main runner writes `all_benchmarks.txt`; `check_gpu.py` writes `check_gpu.txt` plus separate environment, Conv2D, Transformer, matmul, memory-copy, and GPU-memory logs; `train_demo.py` writes `training.txt`; the other scripts write `computer_vision.txt`, `transformers.txt`, `sentence_transformers.txt`, or `resnet18_probe.txt`. Notebook output, including CPU-vs-iGPU comparisons, is saved to `demo_notebook.txt`. Each file is overwritten when that test starts again; interrupted runs may leave partial logs. The directory is ignored by Git.

`transformers_demo.py` downloads the default `distilgpt2` model from Hugging Face on its first run (internet access is required); later runs use the local model cache. It generates on CPU and iGPU by default and reports token throughput. Use `--device cpu` or `--device cuda` to select one device, `--model MODEL_ID` to test another compatible causal language model, or `--seed` to set the generation seed.

`sentence_transformers_demo.py` downloads `sentence-transformers/all-MiniLM-L6-v2` on its first run and encodes a small sample corpus in batches on CPU and iGPU before ranking it against a query. It reports embedding throughput and semantic-search results. Use `--batch-size`, `--top-k`, `--query`, or `--model` to change the demo parameters.

`computer_vision_demo.py` trains ResNet18 and MobileNetV3-Small from scratch on identical synthetic data on CPU and iGPU, then reports training-time and inference-latency speedups alongside test loss and accuracy. It uses FP32 on both devices to avoid the FP16 MIOpen convolution failure observed with the tested `gfx1103` stack. This smoke test does not download a dataset or pretrained weights; increase `--train-size`, `--test-size`, or `--epochs` for a longer run.

`resnet18_probe.py` compares one FP32 training step with a batch of four 64×64 synthetic images. It reports the timing speedup and GPU memory/fault information. **Even this small GPU workload could hang the system**; save your work before running the default CPU-then-iGPU comparison. Use `--device cpu` to avoid the iGPU, or `--device cuda` to run only the iGPU. A timeout cannot guarantee recovery from a GPU or driver hang.

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

`check_gpu.py` runs Conv2D, Transformer operations (scaled dot-product attention, Softmax, and backward pass), matmul, and memory-copy benchmarks on CPU and iGPU. It reports a speedup for every paired measurement. The transformer test checks output shape, Softmax normalization, and finite nonzero gradients; its GFLOPS estimate counts attention matrix multiplications in forward and backward, not Softmax operations. The notebook invokes all benchmark scripts, so it includes these comparisons along with CNN training, computer vision, text generation, sentence embeddings, and the ResNet18 probe.

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

The environment is stored in `.venv/`, but uv's package cache and managed Python installations are outside the project by default, under `~/.cache/uv/` and `~/.local/share/uv/python/` on Linux. Hugging Face model downloads normally go to `~/.cache/huggingface/`, and MIOpen may write a cache under `~/.cache/miopen/`. These locations can be configured and may be shared with other projects. Clearing caches can require downloads or compilation on the next run. Removing `.venv/` removes the project environment; `uv sync` recreates it using the configured Python and package sources.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contribution process and [CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md) for community expectations.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
