"""Save the latest command output while keeping it visible in the terminal."""

from __future__ import annotations

import codecs
import os
import selectors
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "benchmark_results"
CHILD_ENV = "_DL_ROCM_LOG_CHILD"


def run_logged(
    script: Path,
    args: list[str],
    initial_file: str,
    sections: dict[str, str] | None = None,
    results_dir: Path = RESULTS_DIR,
    full_file: str | None = None,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
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
                [sys.executable, "-u", str(script.resolve()), *args],
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
