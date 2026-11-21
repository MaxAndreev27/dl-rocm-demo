import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from benchmark_logs import run_logged


class BenchmarkLogTests(unittest.TestCase):
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
