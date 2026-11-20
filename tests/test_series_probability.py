from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.probability_metrics import ProbabilityMetrics
from chimera.series_classifier import SeriesClassifier, fit_series_classifier
from chimera.series_probability import (SeriesProbabilityCalibration, fit_series_probability, log_probabilities)
from chimera.training import save_checkpoint


class SeriesProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_match_hand_calculation_and_empty_bins(self):
        logp = torch.tensor([[.8, .2], [.6, .4]], dtype=torch.float64).log()
        labels = torch.tensor([0, 1])
        metrics = ProbabilityMetrics(bins=5)
        metrics.update(logp[:1], labels[:1])
        metrics.update(logp[1:], labels[1:])
        report = metrics.result()
        self.assertEqual(report["examples"], 2)
        self.assertAlmostEqual(report["nll"], -(math.log(.8) + math.log(.4)) / 2)
        self.assertAlmostEqual(report["brier"], .4)
        self.assertAlmostEqual(report["ece"], .4)
        self.assertEqual(sum(row["count"] for row in report["bins"]), 2)
        self.assertIsNone(report["bins"][0]["accuracy"])
        self.assertIsNone(ProbabilityMetrics().result()["nll"])
        with self.assertRaises(ValueError):
            metrics.update(torch.zeros(1, 2), torch.tensor([0]))

    def test_convex_temperature_fit_and_boundary_cases(self):
        class StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), num_classes=2, series_features=2)

        class StubClassifier:
            amp = "none"
            num_classes = 2
            checkpoint_sha256 = "a" * 64
            dataset_sha256 = "b" * 64
            def validate(self, config, *, amp=None):
                if amp != self.amp:
                    raise ValueError("précision")
            def scores(self, model, batch):
                self.assert_mode = model.training
                return batch["series"][:, 0].double()

        model, classifier = StubModel().train(), StubClassifier()
        scores = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 0, 1])
        batches = [dict(series=scores[:2, None], class_labels=labels[:2]),
                   dict(series=scores[2:, None], class_labels=labels[2:])]
        fitted = fit_series_probability(model, classifier, batches, checkpoint_sha256="a" * 64,
                                        classifier_sha256="c" * 64, dataset_sha256="d" * 64)
        self.assertTrue(model.training)
        self.assertFalse(classifier.assert_mode)
        self.assertAlmostEqual(fitted.inverse_temperature, math.log(3) / 2, places=8)
        self.assertLess(fitted.nll_after, fitted.nll_before)
        self.assertEqual(fitted.examples, 4)
        self.assertEqual(fitted.probabilities(scores).argmax(-1).tolist(), scores.argmax(-1).tolist())
        self.assertTrue(torch.allclose(fitted.probabilities(scores).sum(-1), torch.ones(4, dtype=torch.float64)))
        torch.testing.assert_close(fitted.probabilities(scores + 1000), fitted.probabilities(scores), rtol=0, atol=0)
        restored = SeriesProbabilityCalibration.from_dict(json.loads(json.dumps(fitted.to_dict())))
        self.assertEqual(restored.inverse_temperature, fitted.inverse_temperature)
        for data in (dict(fitted.to_dict(), inverse_temperature=0), dict(fitted.to_dict(), num_classes=1),
                     dict(fitted.to_dict(), classifier_sha256="invalid"), dict(fitted.to_dict(), format_version=4)):
            with self.assertRaises(ValueError):
                SeriesProbabilityCalibration.from_dict(data)
        all_correct = [dict(series=scores[:, None], class_labels=torch.zeros(4, dtype=torch.long))]
        all_wrong = [dict(series=scores[:, None], class_labels=torch.ones(4, dtype=torch.long))]
        kwargs = dict(checkpoint_sha256="a" * 64, classifier_sha256="c" * 64, dataset_sha256="d" * 64)
        self.assertEqual(fit_series_probability(model, classifier, all_correct, **kwargs).inverse_temperature, 1e4)
        self.assertEqual(fit_series_probability(model, classifier, all_wrong, **kwargs).inverse_temperature, 1e-4)
        constant = [dict(series=torch.ones(4, 1, 2), class_labels=labels)]
        self.assertEqual(fit_series_probability(model, classifier, constant, **kwargs).inverse_temperature, 1.)
        with self.assertRaisesRegex(ValueError, "identique"):
            fit_series_probability(model, classifier, batches, checkpoint_sha256="a" * 64,
                                   classifier_sha256="c" * 64, dataset_sha256="b" * 64)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "exemples étiquetés"):
            fit_series_probability(model, classifier,
                                   [dict(series=scores[:, None], class_labels=torch.full((4,), -100))], **kwargs)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            log_probabilities(torch.tensor([[float("nan"), 1.]]))

    def test_model_fit_preserves_state_and_ignores_unlabeled_batches(self):
        config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True)
        torch.manual_seed(328)
        model = Chimera(config).train()
        training = synthetic_batch(config, batch_size=128, seed=503)
        training = {k: v for k, v in training.items() if k in ("series", "series_mask", "class_labels")}
        classifier = fit_series_classifier(model, [training], checkpoint_sha256="a" * 64, dataset_sha256="b" * 64)
        batch = synthetic_batch(config, batch_size=64, seed=504)
        batch = {k: v for k, v in batch.items() if k in ("series", "series_mask", "class_labels")}
        batch["class_labels"][:5] = -100
        original = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        options = dict(checkpoint_sha256="a" * 64, classifier_sha256="c" * 64, dataset_sha256="d" * 64)
        fitted = fit_series_probability(model, classifier, [batch, {"series": batch["series"][:3]}], **options)
        self.assertEqual(fitted.examples, 59)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0, msg=name)
        chunks = [{key: value[section] for key, value in batch.items()} for section in (slice(0, 24), slice(24, 64))]
        other = fit_series_probability(model, classifier, chunks, **options)
        self.assertAlmostEqual(fitted.inverse_temperature, other.inverse_temperature, delta=1e-4)

    def test_bf16_fit_and_evaluation_use_same_precision(self):
        config = ChimeraConfig.preset("tiny")
        model = Chimera(config)
        train = synthetic_batch(config, batch_size=128, seed=508)
        train = {k: v for k, v in train.items() if k in ("series", "series_mask", "class_labels")}
        head = fit_series_classifier(model, [train], amp="bf16", checkpoint_sha256="a" * 64)
        test = synthetic_batch(config, batch_size=32, seed=509)
        test = {k: v for k, v in test.items() if k in ("series", "series_mask", "class_labels")}
        calibrated = fit_series_probability(model, head, [test], amp="bf16", checkpoint_sha256="a" * 64,
                                             classifier_sha256="c" * 64)
        result = evaluate(model, [test], amp="bf16", series_classifier=head, series_probability=calibrated)
        self.assertEqual(result["series_classifier_probability_metrics"]["calibrated"]["examples"], 32)
        missing = dict(test, class_labels=torch.full_like(test["class_labels"], -100))
        empty = evaluate(model, [missing], amp="bf16", series_classifier=head, series_probability=calibrated)
        self.assertIsNone(empty["series_classifier_probability_metrics"]["calibrated"]["nll"])
        with self.assertRaisesRegex(ValueError, "précision"):
            evaluate(model, [test], series_classifier=head, series_probability=calibrated)

    def test_cli_separates_data_binds_files_and_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True)
            checkpoint, head, calibration = root / "model.pt", root / "head.json", root / "temperature.json"
            train, tuning, test = root / "train.jsonl", root / "calibration.jsonl", root / "test.jsonl"
            report, predictions = root / "report.json", root / "predictions.jsonl"
            save_checkpoint(checkpoint, Chimera(config))
            for path, count, seed in ((train, 128, 505), (tuning, 64, 506), (test, 24, 507)):
                batch = synthetic_batch(config, batch_size=count, seed=seed)
                path.write_text("\n".join(json.dumps(dict(series=batch["series"][i].tolist(),
                                                          class_label=batch["class_labels"][i].item(),
                                                          future=batch["forecast_targets"][i].tolist()))
                                          for i in range(count)))
            main(["--threads", "2", "refine-series-classifier", "--checkpoint", str(checkpoint),
                  "--data", str(train), "--output", str(head)])
            calibration_args = ["--threads", "2", "calibrate-series-probability", "--checkpoint", str(checkpoint),
                                "--series-classifier", str(head), "--data", str(tuning), "--output", str(calibration)]
            main(calibration_args)
            fitted = SeriesProbabilityCalibration.load(calibration)
            self.assertEqual(fitted.examples, 64)
            self.assertLessEqual(fitted.nll_after, fitted.nll_before)
            common = ["--checkpoint", str(checkpoint), "--data", str(test), "--series-classifier", str(head)]
            main(["predict", *common, "--output", str(predictions)])
            before = [json.loads(line) for line in predictions.read_text().splitlines()]
            main(["predict", *common, "--series-probability", str(calibration), "--output", str(predictions)])
            after = [json.loads(line) for line in predictions.read_text().splitlines()]
            for left, right in zip(before, after):
                self.assertEqual(left["series_classifier_class_id"], right["series_classifier_class_id"])
                self.assertEqual(left["forecast"], right["forecast"])
                self.assertAlmostEqual(sum(right["series_classifier_probabilities"]), 1.)
            with self.assertRaisesRegex(ValueError, "précision"):
                main(["predict", *common, "--series-probability", str(calibration), "--amp", "bf16"])
            main(["evaluate", *common, "--series-probability", str(calibration), "--output", str(report)])
            measured = json.loads(report.read_text())
            self.assertEqual(measured["series_classifier_probability_metrics"]["calibrated"]["examples"], 24)
            self.assertIsNone(measured["series_probability_data_reused"])
            main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(tuning),
                  "--series-classifier", str(head), "--series-probability", str(calibration), "--output", str(report)])
            self.assertTrue(json.loads(report.read_text())["series_probability_data_reused"])
            saved = calibration.read_bytes()
            with self.assertRaisesRegex(ValueError, "identique"):
                main([*calibration_args, "--data", str(train)])
            self.assertEqual(calibration.read_bytes(), saved)
            with self.assertRaises(SystemExit):
                main([*calibration_args, "--output", str(head)])
            with self.assertRaises(SystemExit):
                main(["predict", "--checkpoint", str(checkpoint), "--data", str(test),
                      "--series-probability", str(calibration)])
            altered = json.loads(head.read_text())
            altered["bias"][0] += 1
            head.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "checkpoint ou à la tête"):
                main(["predict", *common, "--series-probability", str(calibration)])

    def test_benchmark_uses_disjoint_seed_ranges_and_protects_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / "model.pt", root / "report.json"
            save_checkpoint(checkpoint, Chimera(ChimeraConfig.preset("tiny")))
            original = checkpoint.read_bytes()
            common = [sys.executable, "tools/benchmark_series_probability.py", "--checkpoint", str(checkpoint),
                      "--train-batches", "12", "--calibration-batches", "2", "--test-batches", "1",
                      "--batch-size", "16", "--shape-context", "4"]
            run = subprocess.run([*common, "--output", str(report)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["calibration"]["examples"], 32)
            self.assertEqual(result["measurements"]["identity"]["probability_metrics"]["calibrated"]["examples"], 16)
            for args in (("--calibration-seed", "52000001", "--output", str(report)),
                         ("--test-mask-seed", "53000001", "--output", str(report)),
                         ("--output", str(checkpoint))):
                run = subprocess.run([*common, *args], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(run.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
