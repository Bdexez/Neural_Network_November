from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.calibration import ForecastCalibration
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import save_checkpoint


class CalibrationCliTests(unittest.TestCase):
    def test_fit_evaluate_predict_and_reject_incompatible_or_destructive_outputs(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=2, series_features=1)
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, Chimera(config))
            calibration_data, test_data = root / "calibration.jsonl", root / "test.jsonl"
            for path, count, seed in ((calibration_data, 12, 1000), (test_data, 7, 1001)):
                batch = synthetic_batch(config, batch_size=count, seed=seed)
                path.write_text("\n".join(json.dumps(dict(series=batch["series"][i].tolist(),
                                                         future=batch["forecast_targets"][i].tolist()))
                                         for i in range(count)))
            calibration_path = root / "calibration.json"
            main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                  "--batch-size", "4", "--output", str(calibration_path)])
            fitted = ForecastCalibration.load(calibration_path)
            self.assertEqual(fitted.modalities, ("series",))
            self.assertEqual(fitted.counts.tolist(), [[12], [12]])
            report_path, predictions = root / "report.json", root / "predictions.jsonl"
            common = ["--checkpoint", str(checkpoint), "--data", str(test_data),
                      "--calibration", str(calibration_path)]
            main(["evaluate", *common, "--output", str(report_path)])
            report = json.loads(report_path.read_text())
            self.assertFalse(report["calibration_data_reused"])
            self.assertIn("calibrated_interval_score", report)
            main(["predict", *common, "--output", str(predictions)])
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            for row in rows:
                self.assertEqual(row["calibrated_nominal_coverage"], 0.8)
                for horizon, interval in zip(row["forecast"], row["calibrated_interval"]):
                    self.assertLessEqual(interval[0][0], horizon[0][1])
                    self.assertLessEqual(horizon[0][1], interval[0][1])
            other = root / "other.pt"
            save_checkpoint(other, Chimera(config))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                main(["predict", "--checkpoint", str(other), "--data", str(test_data),
                      "--calibration", str(calibration_path)])
            original = calibration_data.read_bytes()
            with self.assertRaises(SystemExit):
                main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                      "--output", str(calibration_data)])
            self.assertEqual(calibration_data.read_bytes(), original)
            changed = root / "changed.jsonl"
            changed.write_text(json.dumps(dict(series=batch["series"][0].tolist(), text="extra modality")))
            predictions.write_text("previous predictions\n")
            with self.assertRaisesRegex(ValueError, "modalités"):
                main(["predict", "--checkpoint", str(checkpoint), "--data", str(changed),
                      "--calibration", str(calibration_path), "--output", str(predictions)])
            self.assertEqual(predictions.read_text(), "previous predictions\n")


if __name__ == "__main__":
    unittest.main()
