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
from chimera.augmentation import SeriesAugmentation
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.series_classifier import SeriesClassifier, fit_series_classifier, normalized_temporal_features
from chimera.training import save_checkpoint


class SeriesClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = replace(ChimeraConfig.preset("tiny"), temporal_statistics=True)

    def setUp(self):
        torch.manual_seed(125)

    def test_standardized_features_respect_affine_changes_and_padding(self):
        model = Chimera(self.config).eval()
        batch = synthetic_batch(self.config, batch_size=4, seed=491)
        series = batch["series"][:, -12:].double()
        series[0, :, 1] = float("nan")
        series[1, 3:5, 0] = float("nan")
        valid = torch.ones(series.shape[:2], dtype=torch.bool)
        valid[2, 2] = False
        gain = torch.tensor([.4, 10., 3.], dtype=torch.float64)
        shift = torch.tensor([50., -20., 13.], dtype=torch.float64)
        transformed = series * gain + shift
        with torch.inference_mode():
            before = normalized_temporal_features(model, dict(series=series, series_mask=valid), 1e-4)
            after = normalized_temporal_features(model, dict(series=transformed, series_mask=valid), 1e-4)
            padded = torch.nn.functional.pad(series, (0, 0, 3, 4), value=1e6)
            padded_valid = torch.nn.functional.pad(valid, (3, 4), value=False)
            other = normalized_temporal_features(model, dict(series=padded, series_mask=padded_valid), 1e-4)
        torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(before, other, rtol=2e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, "uniquement"):
            normalized_temporal_features(model, dict(series=series, images=batch["images"]), 1e-4)

    def test_shape_features_have_expected_values_and_ignore_missing_positions(self):
        model = Chimera(self.config).eval()
        series = torch.tensor([[[1., 2., float("nan")], [3., 4., float("nan")]]], dtype=torch.float64)
        with torch.inference_mode():
            base = normalized_temporal_features(model, dict(series=series), 1e-4)
            expanded = normalized_temporal_features(model, dict(series=series), 1e-4, shape_context=2)
            transformed = normalized_temporal_features(model, dict(series=series * 3 + 20), 1e-4, shape_context=2)
            padded = torch.nn.functional.pad(series, (0, 0, 2, 3), value=1e5)
            mask = torch.tensor([[0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
            padding_result = normalized_temporal_features(model, dict(series=padded, series_mask=mask), 1e-4, 2)
        torch.testing.assert_close(expanded[:, :self.config.d_model], base.double(), rtol=0, atol=0)
        torch.testing.assert_close(expanded, transformed, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(expanded, padding_result, rtol=2e-5, atol=2e-6)
        extra = expanded[0, self.config.d_model:]
        expected = torch.tensor([-1., 1., -1., 1., 0., 0., 1., 1., 1., 1., 0., 0.,
                                 2., 2., 0., 4., 4., 0., -1., -1., 0.] + [0.] * 18, dtype=torch.float64)
        torch.testing.assert_close(extra, expected, rtol=0, atol=0)

    def test_augmented_shape_fit_is_deterministic_and_versioned(self):
        model = Chimera(self.config).train()
        batch = synthetic_batch(self.config, batch_size=96, seed=498)
        batch = {key: value for key, value in batch.items() if key in ("series", "series_mask", "class_labels")}
        augmentation = SeriesAugmentation(missing_rate=.3, mask_probability=.5)
        rng = torch.get_rng_state().clone()
        first = fit_series_classifier(model, [batch], shape_context=8, augmentation=augmentation, augmentation_seed=42)
        second = fit_series_classifier(model, [batch], shape_context=8, augmentation=augmentation, augmentation_seed=42)
        changed = fit_series_classifier(model, [batch], shape_context=8, augmentation=augmentation, augmentation_seed=43)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(first.weight.shape, (self.config.d_model + 3 * (2 * 8 + 9), self.config.num_classes))
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        self.assertFalse(torch.equal(first.weight, changed.weight))
        payload = first.to_dict()
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["augmentation_seed"], 42)
        self.assertEqual(payload["augmentation"]["missing_rate"], .3)
        loaded = SeriesClassifier.from_dict(json.loads(json.dumps(payload)))
        self.assertEqual(loaded.shape_context, 8)
        augmented_only = fit_series_classifier(model, [batch], augmentation=augmentation, augmentation_seed=42)
        self.assertEqual(augmented_only.to_dict()["format_version"], 2)
        self.assertEqual(SeriesClassifier.from_dict(augmented_only.to_dict()).shape_context, 0)
        model.eval()
        with torch.inference_mode():
            original = loaded.scores(model, batch)
            transformed = dict(batch, series=batch["series"].double() * 4 - 20)
            torch.testing.assert_close(original, loaded.scores(model, transformed), rtol=3e-5, atol=3e-6)
        for corrupted in (dict(payload, format_version=1), dict(payload, shape_context=-1),
                          dict(payload, augmentation={"unknown": 1}), dict(payload, augmentation_seed=None),
                          dict(payload, augmentation_batch_size=0), dict(payload, augmentation_device="")):
            with self.subTest(corrupted=corrupted), self.assertRaises(ValueError):
                SeriesClassifier.from_dict(corrupted)
        with self.assertRaises(ValueError):
            fit_series_classifier(model, [batch], shape_context=self.config.max_series_len + 1)
        with self.assertRaisesRegex(ValueError, "taille des lots"):
            fit_series_classifier(model, [batch], augmentation=augmentation, augmentation_batch_size=10)

    def test_fit_is_batch_invariant_restores_mode_and_rejects_missing_class(self):
        model = Chimera(self.config).train()
        batch = synthetic_batch(self.config, batch_size=128, seed=492)
        batch = {key: value for key, value in batch.items() if key in ("series", "series_mask", "class_labels")}
        batch["class_labels"][0] = -100
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        fitted = fit_series_classifier(model, [batch])
        with_unlabeled = fit_series_classifier(model, [batch, {"series": batch["series"][:3]}])
        split = [{key: value[section] for key, value in batch.items()} for section in (slice(0, 48), slice(48, 128))]
        another = fit_series_classifier(model, split)
        self.assertTrue(model.training)
        self.assertEqual(sum(fitted.class_counts), 127)
        torch.testing.assert_close(fitted.weight, another.weight, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(fitted.bias, another.bias, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(fitted.weight, with_unlabeled.weight, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
        bad = dict(batch, class_labels=torch.zeros(128, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "Chaque classe"):
            fit_series_classifier(model, [bad])
        self.assertTrue(model.training)
        restored = SeriesClassifier.from_dict(json.loads(json.dumps(fitted.to_dict())))
        torch.testing.assert_close(restored.weight, fitted.weight, rtol=0, atol=0)
        for data in (dict(fitted.to_dict(), scale_floor=0), dict(fitted.to_dict(), class_counts=[0] * 6),
                     dict(fitted.to_dict(), weight=[[float("nan")]]), dict(fitted.to_dict(), format_version=99)):
            with self.assertRaises(ValueError):
                SeriesClassifier.from_dict(data)

    def test_bf16_and_unobserved_labels(self):
        model = Chimera(self.config).train()
        batch = synthetic_batch(self.config, batch_size=128, seed=497)
        batch = {key: value for key, value in batch.items() if key in ("series", "series_mask", "class_labels")}
        batch["series"][:, ::4, 0] = float("nan")
        fitted = fit_series_classifier(model, [batch], amp="bf16")
        self.assertTrue(model.training)
        model.eval()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(fitted.scores(model, batch)).all())
        missing = dict(batch, class_labels=torch.full_like(batch["class_labels"], -100))
        report = evaluate(model, [missing], amp="bf16", series_classifier=fitted)
        self.assertEqual(report["series_classifier_examples"], 0)
        self.assertIsNone(report["series_classifier_accuracy"])

    def test_evaluation_reports_classifier_without_changing_model_metrics(self):
        model = Chimera(self.config).eval()
        training = synthetic_batch(self.config, batch_size=128, seed=493)
        train = {key: value for key, value in training.items() if key in ("series", "series_mask", "class_labels")}
        fitted = fit_series_classifier(model, [train])
        validation = synthetic_batch(self.config, batch_size=12, seed=494)
        validation = {key: value for key, value in validation.items()
                      if key in ("series", "series_mask", "class_labels", "forecast_targets")}
        validation["class_labels"][0] = -100
        raw = evaluate(model, [validation])
        result = evaluate(model, [validation], series_classifier=fitted)
        self.assertEqual(result["accuracy"], raw["accuracy"])
        self.assertEqual(result["forecast_mae"], raw["forecast_mae"])
        self.assertEqual(result["series_classifier_examples"], 11)
        self.assertEqual(sum(row["count"] for row in result["series_classifier_confusion"]), 11)
        self.assertTrue(0 <= result["series_classifier_accuracy"] <= 1)
        with torch.inference_mode():
            scores = fitted.scores(model, validation)
            changed_targets = dict(validation, class_labels=torch.zeros_like(validation["class_labels"]),
                                   forecast_targets=torch.full_like(validation["forecast_targets"], 999))
            torch.testing.assert_close(scores, fitted.scores(model, changed_targets), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "précision"):
            evaluate(model, [validation], amp="bf16", series_classifier=fitted)

    def test_cli_roundtrip_and_input_protection(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            checkpoint, data, test = root / "model.pt", root / "train.jsonl", root / "test.jsonl"
            classifier, predictions, report = root / "head.json", root / "predictions.jsonl", root / "report.json"
            save_checkpoint(checkpoint, Chimera(self.config))
            for path, count, seed in ((data, 128, 495), (test, 12, 496)):
                batch = synthetic_batch(self.config, batch_size=count, seed=seed)
                path.write_text("\n".join(json.dumps(dict(series=batch["series"][i].tolist(),
                                                          class_label=batch["class_labels"][i].item(),
                                                          future=batch["forecast_targets"][i].tolist()))
                                          for i in range(count)))
            original = data.read_bytes()
            main(["--threads", "2", "refine-series-classifier", "--checkpoint", str(checkpoint),
                  "--data", str(data), "--output", str(classifier)])
            fitted = SeriesClassifier.load(classifier)
            fitted.validate_checkpoint(checkpoint)
            common = ["--checkpoint", str(checkpoint), "--data", str(test)]
            main(["predict", *common, "--output", str(predictions)])
            raw = [json.loads(row) for row in predictions.read_text().splitlines()]
            main(["predict", *common, "--series-classifier", str(classifier), "--output", str(predictions)])
            rows = [json.loads(row) for row in predictions.read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(len(row["series_classifier_scores"]) == self.config.num_classes for row in rows))
            self.assertEqual([row["forecast"] for row in rows], [row["forecast"] for row in raw])
            main(["evaluate", *common, "--series-classifier", str(classifier), "--output", str(report)])
            metrics = json.loads(report.read_text())
            self.assertEqual(metrics["series_classifier_examples"], 12)
            self.assertIsNone(metrics["series_classifier_data_reused"])
            self.assertEqual(sum(row["count"] for row in metrics["series_classifier_confusion"]), 12)
            shaped_path = root / "shape-head.json"
            main(["--threads", "2", "refine-series-classifier", "--checkpoint", str(checkpoint),
                  "--data", str(data), "--output", str(shaped_path), "--shape-context", "4",
                  "--missing-rate", ".3", "--mask-probability", ".5", "--augmentation-seed", "77"])
            shaped = SeriesClassifier.load(shaped_path)
            self.assertEqual(shaped.shape_context, 4)
            self.assertEqual(shaped.augmentation_seed, 77)
            self.assertEqual(shaped.augmentation_batch_size, 32)
            self.assertEqual(shaped.augmentation_device, "cpu")
            main(["evaluate", *common, "--series-classifier", str(shaped_path), "--output", str(report)])
            self.assertEqual(json.loads(report.read_text())["series_classifier_examples"], 12)
            main(["evaluate", "--checkpoint", str(checkpoint), "--data", str(data),
                  "--series-classifier", str(classifier), "--output", str(report)])
            self.assertTrue(json.loads(report.read_text())["series_classifier_data_reused"])
            with self.assertRaises(SystemExit):
                main(["refine-series-classifier", "--checkpoint", str(checkpoint), "--data", str(data),
                      "--output", str(data)])
            self.assertEqual(data.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "précision"):
                main(["predict", *common, "--series-classifier", str(classifier), "--amp", "bf16"])
            other = root / "other.pt"
            save_checkpoint(other, Chimera(self.config))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                main(["predict", "--checkpoint", str(other), "--data", str(test),
                      "--series-classifier", str(classifier)])
            multimodal = root / "multimodal.jsonl"
            multimodal.write_text(json.dumps(dict(series=[[1., 2., 3.]], text="extra")) + "\n")
            predictions.write_text("previous prediction\n")
            with self.assertRaisesRegex(ValueError, "uniquement"):
                main(["predict", "--checkpoint", str(checkpoint), "--data", str(multimodal),
                      "--series-classifier", str(classifier), "--output", str(predictions)])
            self.assertEqual(predictions.read_text(), "previous prediction\n")
            bad = root / "one-class.jsonl"
            bad.write_text(json.dumps(dict(series=[[1., 2., 3.]], class_label=0)) + "\n")
            prior = classifier.read_bytes()
            with self.assertRaisesRegex(ValueError, "Chaque classe"):
                main(["refine-series-classifier", "--checkpoint", str(checkpoint), "--data", str(bad),
                      "--output", str(classifier)])
            self.assertEqual(classifier.read_bytes(), prior)

    def test_benchmark_protects_checkpoint_and_separates_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, report = root / "model.pt", root / "report.json"
            save_checkpoint(checkpoint, Chimera(self.config))
            before = checkpoint.read_bytes()
            common = [sys.executable, "tools/benchmark_series_classifier.py", "--checkpoint", str(checkpoint),
                      "--train-batches", "12", "--test-batches", "1", "--batch-size", "16",
                      "--shape-context", "4", "--train-missing-rate", ".3"]
            run = subprocess.run([*common, "--output", str(report)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(sum(result["classifier"]["class_counts"]), 192)
            self.assertEqual(result["measurements"]["history8"]["identity"]["examples"], 16)
            self.assertEqual(result["measurements"]["history8"]["identity"]["changed_from_identity"], 0)
            self.assertEqual(result["measurements"]["history8"]["identity"]["shape_changed_from_identity"], 0)
            self.assertEqual(result["shape_classifier"]["shape_context"], 4)
            self.assertEqual(result["classifier"]["augmentation"], result["shape_classifier"]["augmentation"])
            self.assertGreaterEqual(result["measurements"]["history8"]["shift20"]["max_score_difference"], 0)
            for args in (("--test-seed", "43000001", "--output", str(report)),
                         ("--mask-seed", "43000001", "--output", str(report)),
                         ("--train-mask-seed", "43000001", "--output", str(report)),
                         ("--output", str(checkpoint))):
                run = subprocess.run([*common, *args], capture_output=True, text=True, timeout=60)
                self.assertNotEqual(run.returncode, 0)
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
