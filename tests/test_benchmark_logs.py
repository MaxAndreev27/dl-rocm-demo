import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_logs import (
    report_comparison,
    run_logged,
    start_notebook_logging,
    stop_notebook_logging,
)


class BenchmarkLogTests(unittest.TestCase):
    def test_run_logged_forwards_each_output_line_to_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "example.py"
            script.write_text(
                "print('CPU vs iGPU (timing): CPU=2.000 s, iGPU=1.000 s; "
                "iGPU faster x2.00 (iGPU speedup x2.000000)')\n",
                encoding="utf-8",
            )
            lines: list[tuple[str, bool]] = []
            with contextlib.redirect_stdout(io.StringIO()):
                status = run_logged(
                    script,
                    [],
                    "training.txt",
                    results_dir=root,
                    on_line=lambda line, is_stdout: lines.append((line, is_stdout)),
                )
            self.assertEqual(status, 0)
            self.assertIn(
                (
                    "CPU vs iGPU (timing): CPU=2.000 s, iGPU=1.000 s; "
                    "iGPU faster x2.00 (iGPU speedup x2.000000)\n",
                    True,
                ),
                lines,
            )

    def test_help_output_does_not_replace_benchmark_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "example.py"
            script.write_text("print('usage: example.py --help')\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = run_logged(
                    script, ["--help"], "training.txt", results_dir=root
                )
            self.assertEqual(status, 0)
            self.assertIn("usage: example.py --help", output.getvalue())
            self.assertFalse((root / "training.txt").exists())

    def test_notebook_output_is_tee_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            errors = io.StringIO()
            with patch("benchmark_logs.RESULTS_DIR", Path(tmp)):
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    path = start_notebook_logging("notebook.txt")
                    print("comparison result")
                    print("warning", file=sys.stderr)
                    stop_notebook_logging()
            self.assertIn("comparison result", output.getvalue())
            self.assertIn("warning", errors.getvalue())
            saved = path.read_text(encoding="utf-8")
            self.assertIn("comparison result", saved)
            self.assertIn("warning", saved)

    def test_comparison_reports_the_winner_and_speedup(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            speedup = report_comparison(
                "training time", 4.0, 2.0, " s", higher_is_better=False
            )
        self.assertEqual(speedup, 2.0)
        self.assertIn("iGPU faster x2.00", output.getvalue())

        with contextlib.redirect_stdout(output):
            speedup = report_comparison(
                "memory bandwidth", 4.0, 2.0, " GB/s", higher_is_better=True
            )
        self.assertEqual(speedup, 0.5)
        self.assertIn("CPU faster x2.00", output.getvalue())

    def test_sections_are_overwritten_independently_and_include_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "example.py"
            script.write_text(
                "import sys\n"
                "print('environment')\n"
                "print('Test 1: Conv2D')\n"
                "print(sys.argv[1])\n"
                "print('warning', file=sys.stderr)\n"
                "if len(sys.argv) > 2:\n"
                "    print('Test 2: Transformer')\n"
                "    print('attention')\n",
                encoding="utf-8",
            )
            sections = {"Test 1:": "conv.txt", "Test 2:": "attention.txt"}
            output = io.StringIO()
            errors = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                status = run_logged(
                    script, ["first", "both"], "env.txt", sections, root, "full.txt"
                )
                self.assertEqual(status, 0)
                previous_attention = (root / "attention.txt").read_text()
                status = run_logged(
                    script, ["second"], "env.txt", sections, root, "full.txt"
                )
                self.assertEqual(status, 0)

            self.assertNotIn("warning", output.getvalue())
            self.assertIn("warning", errors.getvalue())
            self.assertIn("warning", (root / "full.txt").read_text())
            self.assertEqual((root / "env.txt").read_text(), "environment\n")
            self.assertIn("Test 1: Conv2D\nsecond\n", (root / "conv.txt").read_text())
            self.assertIn("Test 2: Transformer\nattention\n", previous_attention)
            self.assertEqual((root / "attention.txt").read_text(), previous_attention)

    def test_failure_is_recorded_and_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "failure.py"
            script.write_text("raise RuntimeError('failure marker')\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = run_logged(script, [], "training.txt", results_dir=root)
            self.assertNotEqual(status, 0)
            self.assertIn("RuntimeError: failure marker", (root / "training.txt").read_text())


if __name__ == "__main__":
    unittest.main()
