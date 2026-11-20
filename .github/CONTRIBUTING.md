# Contributing

Thank you for considering a contribution to `dl-rocm-demo`.

## Before you start

- Check existing issues and pull requests to avoid duplicating work.
- For a substantial change, open an issue first to discuss the proposed approach.
- Keep contributions focused on the ROCm/PyTorch demo and its documentation.

## Development setup

1. Install Python 3.13 and [uv](https://docs.astral.sh/uv/).
2. Clone the repository and enter its directory.
3. Run `uv sync` to create the project environment.

The pinned ROCm and PyTorch wheels target AMD GPU hardware. CPU-only machines can still check Python syntax, but cannot run the GPU checks.

## Changes and validation

- Keep user-facing documentation, comments, and command output in English.
- Preserve the existing formatting and keep changes focused.
- For Python changes, run `uv run python -m compileall check_gpu.py train_demo.py src`.
- When compatible AMD GPU hardware is available, run `uv run python check_gpu.py` and `uv run python train_demo.py --epochs 1`.
- For notebook changes, verify that the notebook remains valid JSON and that modified code cells are syntactically valid.

## Pull requests

- Explain the problem and the proposed change.
- List the checks you ran and note any hardware or environment limitations.
- Update related documentation when behavior or setup instructions change.
- Keep each pull request focused; avoid bundling unrelated formatting or cleanup.

By submitting a contribution, you agree that it will be distributed under this project's [MIT License](../LICENSE).
