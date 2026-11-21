import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import large_model_stress as stress


class StressTestChecks(unittest.TestCase):
    def run_gpu_case(self, free_gib: float, extra_args: list[str]) -> tuple[int, str]:
        sample = SimpleNamespace(
            gpu_free=free_gib,
            gpu_total=8.0,
            gpu_allocated=0.3,
            gpu_reserved=0.4,
        )
        tensor = MagicMock()
        tensor.to.return_value = tensor
        model = MagicMock()
        model.to.return_value = model
        model.parameters.return_value = []
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["stress", "--device", "cuda", "--duration-minutes", "0.00000001", *extra_args]),
            patch.object(stress.torch, "manual_seed"),
            patch.object(stress.torch.cuda, "is_available", return_value=True),
            patch.object(stress.torch.cuda, "synchronize"),
            patch.object(stress.torch.cuda, "mem_get_info", return_value=(8 * stress.GIB, 8 * stress.GIB)),
            patch.object(stress.torch.cuda, "reset_peak_memory_stats"),
            patch.object(stress.torch.cuda, "get_device_name", return_value="GPU"),
            patch.object(stress.torch.cuda, "max_memory_allocated", return_value=0),
            patch.object(stress.torch.cuda, "max_memory_reserved", return_value=0),
            patch.object(stress, "make_batch", return_value=(tensor, tensor)),
            patch.object(stress, "resnet18", return_value=model),
            patch.object(stress.torch.optim, "AdamW"),
            patch.object(stress, "run_step", return_value=1.0) as run_step,
            patch.object(stress, "sample_telemetry", return_value=sample),
            patch.object(stress, "read_kernel_log", return_value=(None, "unavailable")),
            patch.object(stress, "report_gpu_kernel_faults", return_value=None),
            contextlib.redirect_stdout(output),
        ):
            status = stress.main()
            if free_gib <= 4.0:
                self.assertEqual(run_step.call_count, 1)
        return status, output.getvalue()

    def test_explicit_target_is_not_reported_as_reached_below_threshold(self) -> None:
        status, output = self.run_gpu_case(7.5, ["--target-gtt-gib", "2"])
        self.assertEqual(status, 1)
        self.assertIn("GTT target: not reached", output)

    def test_sampled_limit_stops_during_warmup(self) -> None:
        status, output = self.run_gpu_case(4.0, [])
        self.assertEqual(status, 2)
        self.assertIn("Sampled GTT limit reached", output)
        self.assertIn("Timed phase did not start", output)


if __name__ == "__main__":
    unittest.main()
