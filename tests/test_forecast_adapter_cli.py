from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.calibration import file_sha256
from chimera.cli import main
from chimera.training import load_checkpoint, save_checkpoint
from tools.make_forecast_data import main as make_data
from tools.benchmark_calibration import main as benchmark_calibration


class ForecastAdapterCliTests(unittest.TestCase):
    def test_calibration_benchmark_uses_disjoint_short_masked_histories(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            checkpoint, output = root / "model.pt", root / "study"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")))
            args = ["--checkpoint", str(checkpoint), "--output-dir", str(output),
                    "--case", "history8_missing30", "--modalities", "series", "text+image+series",
                    "--calibration-batches", "2", "--test-batches", "2", "--batch-size", "8", "--threads", "2"]
            benchmark_calibration(args)
            report = json.loads((output / "benchmark.json").read_text())
            self.assertEqual(set(report["reports"]), {"series", "text+image+series"})
            self.assertEqual(report["test_examples"], 16)
            self.assertEqual(report["settings"]["case"], "history8_missing30")
            for entry in report["reports"].values():
                metrics = entry["metrics"]
                self.assertEqual(metrics["forecast_observations"], 16 * 12 * 3)
                self.assertEqual(entry["calibration"]["checkpoint_sha256"], file_sha256(checkpoint))
                self.assertTrue(0 <= metrics["calibrated_interval_coverage"] <= 1)
            for options in (("--test-seed", "5000001"), ("--test-mask-seed", "5100001")):
                with self.assertRaises(SystemExit):
                    benchmark_calibration([*args, *options])

    def test_data_fit_roundtrip_provenance_and_input_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            source, data, output, report = [root / name for name in ("base.pt", "data.jsonl", "fit.pt", "report.json")]
            config = ChimeraConfig.preset("tiny")
            torch.manual_seed(62)
            original = Chimera(config)
            save_checkpoint(source, original, step=7, trainer={"validation_state": {"best_step": None,
                                                                                   "best_loss": float("inf")}})
            source_sha = file_sha256(source)
            make_data(["--checkpoint", str(source), "--output", str(data), "--examples", "96",
                       "--batch-size", "32", "--seed", "7023", "--augmentation-seed", "1007023"])
            rows = [json.loads(line) for line in data.read_text().splitlines()]
            self.assertEqual(len(rows), 96)
            self.assertTrue(any(len(row["series"]) < 32 for row in rows))
            self.assertTrue(any(value is None for row in rows for step in row["series"] for value in step))
            self.assertTrue(all(set(row) == {"series", "future"} for row in rows))
            args = ["--threads", "2", "refine-forecast", "--checkpoint", str(source), "--data", str(data),
                    "--batch-size", "16", "--output", str(output), "--report", str(report), "--relative-features"]
            main(args)
            fitted, payload = load_checkpoint(output)
            fit = json.loads(report.read_text())
            self.assertEqual(fit, payload["trainer"]["forecast_adapter_fit"])
            self.assertEqual(fit["parent_sha256"], source_sha)
            self.assertEqual(file_sha256(source), source_sha)
            self.assertEqual(payload["step"], 7)
            self.assertNotIn("optimizer", payload)
            self.assertNotIn("validation_state", payload["trainer"])
            self.assertEqual(fit["examples"], 96)
            self.assertGreater(fit["active_examples"], 0)
            self.assertEqual(fit["updated_parameters"], 396)
            self.assertEqual(fit["features"], "relative")
            self.assertTrue(fitted.forecast_adapter.relative)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(fitted.state_dict()[name], value, rtol=0, atol=0)
            predictions = root / "predictions.jsonl"
            main(["--threads", "2", "predict", "--checkpoint", str(output), "--data", str(data),
                  "--output", str(predictions)])
            self.assertEqual(len(predictions.read_text().splitlines()), 96)
            main(args)  # Même entrée, même ajustement, sans entraînement cumulatif implicite.
            refitted, _ = load_checkpoint(output)
            torch.testing.assert_close(refitted.forecast_adapter.proj.weight, fitted.forecast_adapter.proj.weight,
                                       rtol=0, atol=0)
            previous_data, previous_report = data.read_bytes(), report.read_bytes()
            for option, path in (("--output", source), ("--output", data), ("--report", source),
                                 ("--report", data), ("--report", output)):
                with self.assertRaises(SystemExit):
                    main([*args, option, str(path)])
            self.assertEqual(file_sha256(source), source_sha)
            self.assertEqual(data.read_bytes(), previous_data)
            self.assertEqual(report.read_bytes(), previous_report)
            invalid = root / "invalid.jsonl"
            invalid.write_text(json.dumps(dict(series=rows[0]["series"])))
            with self.assertRaisesRegex(ValueError, "forecast_targets"):
                main([*args, "--data", str(invalid)])
            self.assertEqual(report.read_bytes(), previous_report)


if __name__ == "__main__":
    unittest.main()
