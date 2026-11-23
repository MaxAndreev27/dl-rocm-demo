"""Save the latest command output while keeping it visible in the terminal."""

from __future__ import annotations

import codecs
import math
import os
import selectors
import subprocess
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
CHILD_ENV = "_DL_ROCM_LOG_CHILD"
_notebook_log: tuple[TextIO, TextIO, TextIO] | None = None


class _TeeStream:
    def __init__(self, console: TextIO, log: TextIO) -> None:
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def __getattr__(self, name: str) -> object:
        return getattr(self.console, name)


def report_comparison(
    metric: str,
    cpu_value: float,
    gpu_value: float,
    unit: str,
    *,
    higher_is_better: bool = False,
) -> float:
    if not math.isfinite(cpu_value) or not math.isfinite(gpu_value):
        raise ValueError("Comparison values must be finite")
    if cpu_value <= 0 or gpu_value <= 0:
        raise ValueError("Comparison values must be positive")

    speedup = gpu_value / cpu_value if higher_is_better else cpu_value / gpu_value
    if not math.isfinite(speedup) or speedup <= 0:
        raise ValueError("Comparison speedup must be finite and positive")
    if speedup >= 1:
        faster, factor = "iGPU", speedup
    else:
        faster, factor = "CPU", 1 / speedup
    print(
        f"CPU vs iGPU ({metric}): CPU={cpu_value:.3f}{unit}, "
        f"iGPU={gpu_value:.3f}{unit}; {faster} faster x{factor:.2f} "
        f"(iGPU speedup x{speedup:.8g})"
    )
    return speedup


def start_notebook_logging(filename: str = "demo_notebook.txt") -> Path:
    global _notebook_log
    stop_notebook_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log = (RESULTS_DIR / filename).open("w", encoding="utf-8", buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(stdout, log)
    sys.stderr = _TeeStream(stderr, log)
    _notebook_log = (stdout, stderr, log)
    return RESULTS_DIR / filename


def stop_notebook_logging() -> None:
    global _notebook_log
    if _notebook_log is None:
        return
    stdout, stderr, log = _notebook_log
    sys.stdout, sys.stderr = stdout, stderr
    log.close()
    _notebook_log = None


def run_logged(
    script: Path,
    args: list[str],
    initial_file: str,
    sections: dict[str, str] | None = None,
    results_dir: Path = RESULTS_DIR,
    full_file: str | None = None,
    on_line: Callable[[str, bool], None] | None = None,
) -> int:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    command = [sys.executable, "-u", str(script.resolve()), *args]
    if "-h" in args or "--help" in args:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=False
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    results_dir.mkdir(parents=True, exist_ok=True)
    full_context = (
        (results_dir / full_file).open("w", encoding="utf-8", buffering=1)
        if full_file is not None
        else nullcontext()
    )
    with full_context as full, (results_dir / initial_file).open(
        "w", encoding="utf-8", buffering=1
    ) as initial:
        log = initial
        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            ) as process:
                assert process.stdout is not None and process.stderr is not None

                def write_line(line: str, is_stdout: bool) -> None:
                    nonlocal log
                    if sections and is_stdout:
                        for heading, filename in sections.items():
                            if line.startswith(heading):
                                if log is not initial:
                                    log.close()
                                log = (results_dir / filename).open(
                                    "w", encoding="utf-8", buffering=1
                                )
                                break
                    stream = sys.stdout if is_stdout else sys.stderr
                    stream.write(line)
                    stream.flush()
                    log.write(line)
                    log.flush()
                    if full is not None:
                        full.write(line)
                        full.flush()
                    if on_line is not None:
                        on_line(line, is_stdout)

                try:
                    with selectors.DefaultSelector() as selector:
                        for pipe, is_stdout in ((process.stdout, True), (process.stderr, False)):
                            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                            selector.register(pipe, selectors.EVENT_READ, (is_stdout, decoder))
                        pending = {True: "", False: ""}
                        while selector.get_map():
                            for key, _ in selector.select():
                                is_stdout, decoder = key.data
                                data = os.read(key.fd, 65536)
                                pending[is_stdout] += decoder.decode(data, final=not data)
                                while "\n" in pending[is_stdout]:
                                    line, pending[is_stdout] = pending[is_stdout].split("\n", 1)
                                    write_line(line + "\n", is_stdout)
                                if not data:
                                    if pending[is_stdout]:
                                        write_line(pending[is_stdout], is_stdout)
                                    selector.unregister(key.fileobj)
                except BaseException:
                    process.terminate()
                    raise
                status = process.wait()
        finally:
            if log is not initial:
                log.close()
    return status if status >= 0 else 128 - status
