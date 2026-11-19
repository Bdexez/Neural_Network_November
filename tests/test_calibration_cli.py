from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.calibration import ForecastCalibration
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import save_checkpoint


class CalibrationCliTests(unittest.TestCase):
    def test_joint_benchmark_rejects_overlapping_seeds_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=2, series_features=1)
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, Chimera(config))
            original = checkpoint.read_bytes()
            report = root / "report.json"
            common = [sys.executable, "tools/benchmark_joint_calibration.py", "--checkpoint", str(checkpoint),
                      "--batch-size", "2", "--calibration-batches", "5", "--test-batches", "2"]
            result = subprocess.run([*common, "--output", str(report)], text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(report.read_text())
            for condition in ("series", "all"):
                joint = data["measurements"][condition]["joint"]
                self.assertEqual(joint["calibration"]["scope"], "joint")
                self.assertEqual(joint["evaluation"]["forecast_complete_examples"], 4)
                scaled = data["measurements"][condition]["joint_width"]
                self.assertEqual(scaled["calibration"]["score_scale"], "width")
                self.assertEqual(scaled["evaluation"]["forecast_complete_examples"], 4)
            for options in (("--test-seed", "4100001", "--output", str(report)),
                            ("--feature-scales", "0", "--output", str(report)),
                            ("--feature-scales", "1", "2", "--output", str(report)),
                            ("--output", str(checkpoint))):
                result = subprocess.run([*common, *options], text=True, capture_output=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(), original)

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
            self.assertEqual(fitted.scope, "marginal")
            self.assertEqual(fitted.to_dict()["format_version"], 1)
            self.assertEqual(fitted.counts.tolist(), [[12], [12]])
            report_path, predictions = root / "report.json", root / "predictions.jsonl"
            common = ["--checkpoint", str(checkpoint), "--data", str(test_data),
                      "--calibration", str(calibration_path)]
            main(["evaluate", *common, "--output", str(report_path)])
            report = json.loads(report_path.read_text())
            self.assertIsNone(report["calibration_data_reused"])
            self.assertFalse(report["calibration_jsonl_identical"])
            self.assertTrue(report["calibration_external_images_match"])
            self.assertIn("calibrated_interval_score", report)
            joint_path = root / "joint.json"
            main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                  "--batch-size", "4", "--scope", "joint", "--output", str(joint_path)])
            joint = ForecastCalibration.load(joint_path)
            self.assertEqual(joint.scope, "joint")
            self.assertEqual(joint.counts.tolist(), [[12], [12]])
            main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(test_data),
                  "--calibration", str(joint_path), "--output", str(report_path)])
            joint_report = json.loads(report_path.read_text())
            self.assertEqual(joint_report["forecast_complete_examples"], 7)
            self.assertEqual(joint_report["calibrated_coverage_scope"], "joint")
            self.assertIsNotNone(joint_report["calibrated_joint_interval_coverage"])
            self.assertIsNone(joint_report["calibrated_interval_score"])
            main(["predict", "--checkpoint", str(checkpoint), "--data", str(test_data),
                  "--calibration", str(joint_path), "--output", str(predictions)])
            self.assertTrue(all(json.loads(line)["calibrated_coverage_scope"] == "joint"
                                for line in predictions.read_text().splitlines()))
            scaled_path = root / "joint-scaled.json"
            main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                  "--scope", "joint", "--score-scale", "width", "--scale-floor", "0.001",
                  "--output", str(scaled_path)])
            scaled = ForecastCalibration.load(scaled_path)
            self.assertEqual(scaled.score_scale, "width")
            self.assertEqual(scaled.scale_floor, 0.001)
            main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(test_data),
                  "--calibration", str(scaled_path), "--output", str(report_path)])
            self.assertEqual(json.loads(report_path.read_text())["calibrated_score_scale"], "width")
            self.assertEqual(json.loads(report_path.read_text())["calibrated_scale_floor"], 0.001)
            main(["predict", "--checkpoint", str(checkpoint), "--data", str(test_data),
                  "--calibration", str(scaled_path), "--output", str(predictions)])
            self.assertTrue(all(json.loads(line)["calibrated_score_scale"] == "width"
                                for line in predictions.read_text().splitlines()))
            self.assertTrue(all(json.loads(line)["calibrated_scale_floor"] == 0.001
                                for line in predictions.read_text().splitlines()))
            prior = scaled_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "nécessite scope joint"):
                main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                      "--score-scale", "width", "--output", str(scaled_path)])
            self.assertEqual(scaled_path.read_bytes(), prior)
            main(["predict", *common, "--output", str(predictions)])
            rows = [json.loads(line) for line in predictions.read_text().splitlines()]
            self.assertEqual(len(rows), 7)
            with self.assertRaisesRegex(ValueError, "précision"):
                main(["predict", *common, "--amp", "bf16"])
            for row in rows:
                self.assertEqual(row["calibrated_nominal_coverage"], 0.8)
                for horizon, interval in zip(row["forecast"], row["calibrated_interval"]):
                    self.assertLessEqual(interval[0][0], horizon[0][1])
                    self.assertLessEqual(horizon[0][1], interval[0][1])
            bf16_calibration = root / "calibration-bf16.json"
            main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(calibration_data),
                  "--amp", "bf16", "--output", str(bf16_calibration)])
            main(["predict", "--checkpoint", str(checkpoint), "--data", str(test_data),
                  "--calibration", str(bf16_calibration), "--amp", "bf16", "--output", str(predictions)])
            self.assertEqual(len(predictions.read_text().splitlines()), 7)
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

    def test_data_reuse_diagnostics_track_external_images_and_legacy_metadata(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow requis pour les images externes")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=2, series_features=1)
            checkpoint, calibration_path = root / "model.pt", root / "calibration.json"
            save_checkpoint(checkpoint, Chimera(config))
            original, copy = root / "original", root / "copy"
            original.mkdir()
            copy.mkdir()
            for path, color in ((original, "black"), (copy, "white")):
                Image.new("RGB", (config.image_size, config.image_size), color).save(path / "image.png")
            batch = synthetic_batch(config, batch_size=10, seed=1901)
            rows = [dict(image_path="image.png", series=batch["series"][i].tolist(),
                         future=batch["forecast_targets"][i].tolist()) for i in range(10)]
            text = "\n".join(json.dumps(row) for row in rows) + "\n"
            (original / "data.jsonl").write_text(text)
            (copy / "data.jsonl").write_text(text)
            main(["calibrate", "--checkpoint", str(checkpoint), "--data", str(original / "data.jsonl"),
                  "--output", str(calibration_path)])
            fitted = ForecastCalibration.load(calibration_path)
            self.assertTrue(fitted.dataset_images_recorded)
            self.assertEqual(len(fitted.dataset_images_sha256), 64)
            report = root / "report.json"

            def evaluate_on(path):
                main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(path),
                      "--calibration", str(calibration_path), "--output", str(report)])
                return json.loads(report.read_text())

            same = evaluate_on(original / "data.jsonl")
            self.assertTrue(same["calibration_jsonl_identical"])
            self.assertTrue(same["calibration_data_reused"])
            self.assertTrue(same["calibration_external_images_match"])
            changed_image = evaluate_on(copy / "data.jsonl")
            self.assertTrue(changed_image["calibration_data_reused"])
            self.assertFalse(changed_image["calibration_external_images_match"])
            # Même contenu logique et mêmes cibles, mais octets JSONL différents.
            (original / "data.jsonl").write_text(text + "\n")
            different_bytes = evaluate_on(original / "data.jsonl")
            self.assertFalse(different_bytes["calibration_jsonl_identical"])
            self.assertIsNone(different_bytes["calibration_data_reused"])
            self.assertTrue(different_bytes["calibration_external_images_match"])
            legacy = json.loads(calibration_path.read_text())
            del legacy["dataset_images_sha256"]
            calibration_path.write_text(json.dumps(legacy))
            self.assertFalse(ForecastCalibration.load(calibration_path).dataset_images_recorded)
            self.assertIsNone(evaluate_on(original / "data.jsonl")["calibration_external_images_match"])
            legacy["dataset_sha256"] = None
            calibration_path.write_text(json.dumps(legacy))
            unknown = evaluate_on(original / "data.jsonl")
            self.assertIsNone(unknown["calibration_jsonl_identical"])
            self.assertIsNone(unknown["calibration_data_reused"])


if __name__ == "__main__":
    unittest.main()
