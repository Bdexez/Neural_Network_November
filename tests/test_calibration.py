from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.calibration import ForecastCalibration, conformity_scores, file_sha256, fit_calibration
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_model_fitting_is_batch_invariant_and_restores_mode_without_changing_weights(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        batch = synthetic_batch(model.config, batch_size=10, seed=128)
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        full = fit_calibration(model, [batch])
        chunks = [{key: value[section] for key, value in batch.items()}
                  for section in (slice(0, 4), slice(4, 10))]
        split = fit_calibration(model, chunks)
        torch.testing.assert_close(full.correction, split.correction, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0, msg=name)
        for batches in ([], [{"series": batch["series"]}], [batch, {"series": batch["series"],
                                                                   "forecast_targets": batch["forecast_targets"]}]):
            with self.assertRaises(ValueError):
                fit_calibration(model, batches)
            self.assertTrue(model.training)

    def test_calibrated_coverage_on_independent_uniform_samples(self):
        generator = torch.Generator().manual_seed(80)
        targets = 2 * torch.rand(10000, 1, 1, generator=generator) - 1
        forecasts = torch.tensor([-3., 0., 3.]).expand(10000, 1, 1, 3)
        fitted = ForecastCalibration.from_scores(conformity_scores(forecasts[:1000], targets[:1000]),
                                                 (0.1, 0.5, 0.9), ("series",))
        interval = fitted.interval(forecasts[1000:])
        covered = ((targets[1000:] >= interval[..., 0]) & (targets[1000:] <= interval[..., 1])).float().mean()
        self.assertAlmostEqual(covered.item(), 0.8, delta=0.025)
        self.assertLess((interval[..., 1] - interval[..., 0]).mean().item(), 2)

    def test_evaluation_preserves_point_metrics_and_weights_missing_targets_correctly(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        fitted = fit_calibration(model, [synthetic_batch(model.config, batch_size=12, seed=190)])
        batch = synthetic_batch(model.config, batch_size=7, seed=191)
        batch["forecast_targets"][1:3] = float("nan")
        batch["forecast_targets"][:, -1] = float("nan")
        raw = evaluate(model, [batch])
        full = evaluate(model, [batch], calibration=fitted)
        split = evaluate(model, [{key: value[section] for key, value in batch.items()}
                                 for section in (slice(0, 2), slice(2, 7))], calibration=fitted)
        for name in ("forecast_mae", "accuracy", "text", "forecast_interval_coverage", "forecast_interval_width"):
            self.assertEqual(raw[name], full[name], name)
        for name in ("calibrated_interval_coverage", "calibrated_interval_width", "calibrated_interval_score",
                     "forecast_interval_score"):
            self.assertAlmostEqual(full[name], split[name], places=5, msg=name)
        self.assertIsNone(full["calibrated_interval_coverage_by_horizon"][-1])
        self.assertIsNone(full["forecast_interval_width_by_horizon"][-1])
        self.assertTrue(model.training)
        bad = {"series": batch["series"], "forecast_targets": batch["forecast_targets"]}
        with self.assertRaisesRegex(ValueError, "modalités"):
            evaluate(model, [bad], calibration=fitted)
        self.assertTrue(model.training)

    def test_interval_score_penalizes_width_and_missed_targets(self):
        class FixedForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=1, series_features=1)

            def forward(self, series, **kwargs):
                forecast = series.new_tensor([0., 1., 2.]).expand(series.size(0), 1, 1, 3)
                return SimpleNamespace(forecast=forecast, loss=None, losses={"forecast": self.anchor * 0})

        model = FixedForecast()
        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[1.]], [[20]], ("series",))
        report = evaluate(model, [dict(series=torch.ones(3, 1, 1),
                                       forecast_targets=torch.tensor([-1., 1., 3.])[:, None, None])],
                          calibration=fitted)
        self.assertAlmostEqual(report["forecast_interval_score"], 26 / 3)
        self.assertAlmostEqual(report["calibrated_interval_score"], 4)
        self.assertEqual(report["calibrated_interval_coverage"], 1)

    def test_exact_finite_sample_rank_instead_of_interpolation_or_clipping(self):
        scores = torch.arange(9, dtype=torch.float64)[:, None, None]
        fitted = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",))
        self.assertEqual(fitted.correction.item(), 7)  # ceil((9+1)*0.8) = 8
        self.assertEqual(fitted.counts.item(), 9)
        with self.assertRaisesRegex(ValueError, "insuffisante"):
            ForecastCalibration.from_scores(scores[:3], (0.1, 0.5, 0.9), ("series",))

    def test_missing_targets_are_counted_separately_for_each_horizon_and_variable(self):
        forecasts = torch.tensor([-1., 0., 1.]).expand(10, 2, 2, 3).clone()
        targets = torch.arange(10, dtype=torch.float32)[:, None, None].expand(10, 2, 2).clone()
        targets[0, 0, 0] = float("nan")
        targets[0:2, 1, 1] = float("inf")
        scores = conformity_scores(forecasts, targets)
        fitted = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",))
        torch.testing.assert_close(fitted.counts, torch.tensor([[9, 10], [10, 8]]))
        torch.testing.assert_close(fitted.correction, torch.tensor([[7., 7.], [7., 8.]], dtype=torch.float64))
        scores[:, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "horizon 1, variable 0"):
            ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",))

    def test_negative_correction_shrinks_and_never_inverts_interval_or_changes_quantiles(self):
        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[-2.]], [[20]], ("series",))
        forecasts = torch.tensor([[[[-1., 0.25, 1.]]]])
        original = forecasts.clone()
        interval = fitted.interval(forecasts)
        torch.testing.assert_close(interval, torch.tensor([[[[0.25, 0.25]]]]))
        torch.testing.assert_close(forecasts, original, rtol=0, atol=0)
        fitted.correction.fill_(0.5)
        torch.testing.assert_close(fitted.interval(forecasts), torch.tensor([[[[-1.5, 1.5]]]]))

    def test_json_roundtrip_binds_checkpoint_and_modalities(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"original weights")
            fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[0.3]], [[20]], ("series",),
                                         checkpoint_sha256=file_sha256(checkpoint))
            restored = ForecastCalibration.from_dict(json.loads(json.dumps(fitted.to_dict())))
            restored.validate_checkpoint(checkpoint)
            config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=1, series_features=1)
            restored.validate(config, {"series": torch.ones(1)})
            with self.assertRaisesRegex(ValueError, "modalités"):
                restored.validate(config, {"series": torch.ones(1), "images": torch.ones(1)})
            with self.assertRaisesRegex(ValueError, "configuration"):
                restored.validate(replace(config, forecast_horizon=2))
            checkpoint.write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                restored.validate_checkpoint(checkpoint)

    def test_invalid_scores_metadata_and_predictions_are_rejected(self):
        valid = dict(quantiles=(0.1, 0.5, 0.9), coverage=0.8, correction=[[0.]], counts=[[20]],
                     modalities=("series",))
        for change in (dict(coverage=1), dict(coverage=float("nan")), dict(counts=[[0]]),
                       dict(counts=[[1.5]]), dict(correction=[[float("inf")]]),
                       dict(modalities=("series", "images")), dict(checkpoint_sha256="invalid"),
                       dict(quantiles=(0.1, 0.9))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ForecastCalibration(**(valid | change))
        for forecast in (torch.tensor([[[[1., 0., -1.]]]]), torch.full((1, 1, 1, 3), float("nan"))):
            with self.assertRaises(ValueError):
                conformity_scores(forecast, torch.zeros(1, 1, 1))
        with self.assertRaises(ValueError):
            ForecastCalibration.from_dict(dict(format_version=999))


if __name__ == "__main__":
    unittest.main()
