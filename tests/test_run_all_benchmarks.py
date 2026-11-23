import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from run_all_benchmarks import (
    Comparison,
    geometric_mean_speedup,
    main,
    parse_comparison_line,
)


class RunAllBenchmarksTests(unittest.TestCase):
    def test_parse_comparison_line_retains_metric_and_normalized_speedup(self) -> None:
        result = parse_comparison_line(
            "CPU vs iGPU (matmul throughput (fp32 vs fp16)): "
            "CPU=664.900 GFLOPS, iGPU=5910.100 GFLOPS; "
            "iGPU faster x8.89 (iGPU speedup x8.889999)",
            "check_gpu.py",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.script, "check_gpu.py")
        self.assertEqual(result.metric, "matmul throughput (fp32 vs fp16)")
        self.assertEqual(result.cpu_value, 664.9)
        self.assertEqual(result.gpu_value, 5910.1)
        self.assertEqual(result.unit, " GFLOPS")
        self.assertEqual(result.igpu_speedup, 8.889999)

    def test_aggregate_uses_geometric_mean_of_normalized_speedups(self) -> None:
        cpu_wins = parse_comparison_line(
            "CPU vs iGPU (training time): CPU=4.000 s, iGPU=8.000 s; "
            "CPU faster x2.00 (iGPU speedup x0.500000)"
        )
        gpu_wins = parse_comparison_line(
            "CPU vs iGPU (throughput): CPU=2.000 tokens/s, iGPU=8.000 tokens/s; "
            "iGPU faster x4.00 (iGPU speedup x4.000000)"
        )
        assert cpu_wins is not None
        assert gpu_wins is not None

        self.assertAlmostEqual(
            geometric_mean_speedup([cpu_wins, gpu_wins]) or 0,
            2**0.5,
        )
        self.assertIsNone(geometric_mean_speedup([]))

    def test_runner_reports_an_overall_result_after_all_benchmarks(self) -> None:
        calls: list[str] = []

        def fake_run_script(script, title, args, comparisons):
            calls.append(title)
            comparisons.append(
                Comparison(script, title, 2.0, 4.0, " units", 2.0)
            )
            return 0

        output = io.StringIO()
        with (
            patch("run_all_benchmarks.torch.cuda.is_available", return_value=True),
            patch("run_all_benchmarks.torch.cuda.get_device_name", return_value="test iGPU"),
            patch("run_all_benchmarks.run_script", side_effect=fake_run_script),
            patch.object(sys, "argv", ["run_all_benchmarks.py"]),
            contextlib.redirect_stdout(output),
        ):
            status = main()

        self.assertEqual(status, 0)
        self.assertEqual(
            calls,
            [
                "PyTorch operation benchmarks",
                "Small CNN training",
                "Computer vision",
                "Text generation",
                "Sentence embeddings",
                "ResNet18 probe",
            ],
        )
        self.assertIn("iGPU is faster by x2.00", output.getvalue())

    def test_runner_requires_iGPU_for_default_comparison(self) -> None:
        output = io.StringIO()
        with (
            patch("run_all_benchmarks.torch.cuda.is_available", return_value=False),
            patch.object(sys, "argv", ["run_all_benchmarks.py"]),
            patch("run_all_benchmarks.run_script") as run_script,
            contextlib.redirect_stdout(output),
        ):
            status = main()

        self.assertEqual(status, 1)
        run_script.assert_not_called()
        self.assertIn("ROCm iGPU is unavailable", output.getvalue())


if __name__ == "__main__":
    unittest.main()
