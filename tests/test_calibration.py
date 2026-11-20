from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from chimera import Chimera, ChimeraConfig
from chimera.calibration import (ForecastCalibration, conformity_scores, file_sha256, fit_calibration,
                                 tail_conformity_scores)
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

    def test_asymmetric_tail_ranks_and_versioned_roundtrip(self):
        targets = torch.arange(40, dtype=torch.float64).reshape(-1, 1, 1)
        targets[-1] = float("nan")
        forecast = torch.tensor([-1., 0., 2.], dtype=torch.float64).expand(40, 1, 1, 3)
        scores = tail_conformity_scores(forecast, targets)
        fitted = ForecastCalibration.from_tail_scores(scores, (.1, .5, .9), ("series",),
                                                       coverage=.8, lower_tail_fraction=.25)
        self.assertEqual(fitted.counts.tolist(), [[39]])
        self.assertEqual(fitted.correction.tolist(), [[-2.]])
        self.assertEqual(fitted.upper_correction.tolist(), [[31.]])
        self.assertEqual(fitted.interval(forecast[:1]).tolist(), [[[[0., 33.]]]])
        data = fitted.to_dict()
        self.assertEqual(data["format_version"], 4)
        restored = ForecastCalibration.from_dict(json.loads(json.dumps(data)))
        self.assertEqual(restored.lower_tail_fraction, .25)
        torch.testing.assert_close(restored.interval(forecast[:4]), fitted.interval(forecast[:4]), rtol=0, atol=0)
        for corrupted in (dict(data, format_version=1), dict(data, upper_correction=None),
                          dict(data, lower_tail_fraction=0), dict(data, scope="invalid"),
                          dict(data, upper_correction=[[float("inf")]])):
            with self.subTest(corrupted=corrupted), self.assertRaises(ValueError):
                ForecastCalibration.from_dict(corrupted)
        mismatched = scores.clone()
        mismatched[0, 0, 0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "mêmes cibles"):
            ForecastCalibration.from_tail_scores(mismatched, (.1, .5, .9), ("series",))
        with self.assertRaisesRegex(ValueError, "insuffisante"):
            ForecastCalibration.from_tail_scores(scores[:8], (.1, .5, .9), ("series",))

    def test_asymmetric_joint_scores_use_complete_trajectory_maxima(self):
        targets = torch.arange(40, dtype=torch.float64)[:, None, None].expand(40, 2, 2).clone()
        targets[:, 1, 1] += 3
        targets[-1, 0, 0] = float("nan")
        forecast = torch.tensor([-1., 0., 2.], dtype=torch.float64).expand(40, 2, 2, 3)
        scores = tail_conformity_scores(forecast, targets, score_scale="width")
        fitted = ForecastCalibration.from_tail_scores(scores, (.1, .5, .9), ("series",), scope="joint",
                                                       score_scale="width", coverage=.8, lower_tail_fraction=.5)
        complete = torch.isfinite(scores).flatten(1).all(-1)
        maxima = scores[complete].flatten(1, 2).amax(1)
        rank = 36  # ceil((39 + 1) * 0.9)
        self.assertEqual(fitted.counts.tolist(), [[39, 39], [39, 39]])
        self.assertAlmostEqual(fitted.correction[0, 0].item(), maxima[:, 0].kthvalue(rank).values.item())
        self.assertAlmostEqual(fitted.upper_correction[0, 0].item(), maxima[:, 1].kthvalue(rank).values.item())
        self.assertTrue((fitted.upper_correction == fitted.upper_correction[0, 0]).all())
        restored = ForecastCalibration.from_dict(fitted.to_dict())
        self.assertEqual(restored.scope, "joint")
        self.assertEqual(restored.score_scale, "width")
        torch.testing.assert_close(restored.interval(forecast[:2]), fitted.interval(forecast[:2]))

    def test_asymmetric_calibration_balances_tails_on_independent_skewed_samples(self):
        generator = torch.Generator().manual_seed(9234)
        targets = -torch.log(torch.rand(14000, 1, 1, generator=generator).clamp_min(1e-8))
        forecasts = torch.tensor([.5, 1., 1.5]).expand(14000, 1, 1, 3)
        fitted = ForecastCalibration.from_tail_scores(tail_conformity_scores(forecasts[:2000], targets[:2000]),
                                                       (.1, .5, .9), ("series",))
        interval = fitted.interval(forecasts[2000:])
        below = (targets[2000:] < interval[..., 0]).float().mean().item()
        above = (targets[2000:] > interval[..., 1]).float().mean().item()
        self.assertAlmostEqual(below, .1, delta=.025)
        self.assertAlmostEqual(above, .1, delta=.025)
        self.assertNotAlmostEqual(fitted.correction.item(), fitted.upper_correction.item(), places=2)

    def test_asymmetric_joint_tail_events_on_independent_trajectories(self):
        generator = torch.Generator().manual_seed(9235)
        targets = -torch.log(torch.rand(12000, 3, 2, generator=generator).clamp_min(1e-8))
        forecasts = torch.tensor([.5, 1., 1.5]).expand(12000, 3, 2, 3)
        scores = tail_conformity_scores(forecasts[:2000], targets[:2000], score_scale="width")
        fitted = ForecastCalibration.from_tail_scores(scores, (.1, .5, .9), ("series",), scope="joint",
                                                       score_scale="width")
        interval = fitted.interval(forecasts[2000:])
        below = (targets[2000:] < interval[..., 0]).flatten(1).any(-1).float().mean().item()
        above = (targets[2000:] > interval[..., 1]).flatten(1).any(-1).float().mean().item()
        self.assertAlmostEqual(below, .1, delta=.025)
        self.assertAlmostEqual(above, .1, delta=.025)

    def test_mixed_precision_calibration_cannot_be_reused_with_another_precision(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        batch = synthetic_batch(model.config, batch_size=8, seed=195)
        fitted = fit_calibration(model, [batch], amp="bf16")
        restored = ForecastCalibration.from_dict(json.loads(json.dumps(fitted.to_dict())))
        self.assertEqual(restored.amp, "bf16")
        with self.assertRaisesRegex(ValueError, "précision"):
            evaluate(model, [batch], calibration=restored)
        self.assertTrue(model.training)
        report = evaluate(model, [batch], amp="bf16", calibration=restored)
        self.assertIsNotNone(report["calibrated_interval_coverage"])

    def test_evaluation_preserves_point_metrics_and_weights_missing_targets_correctly(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        fitted = fit_calibration(model, [synthetic_batch(model.config, batch_size=12, seed=190)])
        batch = synthetic_batch(model.config, batch_size=7, seed=191)
        batch["forecast_targets"][1:3] = float("nan")
        batch["forecast_targets"][:, -1] = float("nan")
        batch["forecast_targets"][:, :, 1] = float("nan")
        raw = evaluate(model, [batch])
        full = evaluate(model, [batch], calibration=fitted)
        split = evaluate(model, [{key: value[section] for key, value in batch.items()}
                                 for section in (slice(0, 2), slice(2, 7))], calibration=fitted)
        for name in ("forecast_mae", "accuracy", "text", "forecast_interval_coverage", "forecast_interval_width"):
            self.assertEqual(raw[name], full[name], name)
        self.assertAlmostEqual(full["forecast"], sum(full["forecast_pinball_by_quantile"]) / 3, places=6)
        for name in ("calibrated_interval_coverage", "calibrated_interval_width", "calibrated_interval_score",
                     "forecast_interval_score"):
            self.assertAlmostEqual(full[name], split[name], places=5, msg=name)
        self.assertIsNone(full["calibrated_interval_coverage_by_horizon"][-1])
        self.assertIsNone(full["forecast_interval_width_by_horizon"][-1])
        self.assertIsNone(full["forecast_interval_width_by_feature"][1])
        self.assertIsNone(full["calibrated_interval_width_by_feature"][1])
        for index in (0, 2):
            self.assertAlmostEqual(full["calibrated_interval_width_by_feature"][index],
                                   split["calibrated_interval_width_by_feature"][index], places=5)
        self.assertEqual(full["forecast_complete_examples"], 0)
        self.assertIsNone(full["forecast_joint_interval_coverage"])
        self.assertIsNone(full["calibrated_joint_interval_coverage"])
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
        asymmetric = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[0.]], [[20]], ("series",),
                                         upper_correction=[[0.]], lower_tail_fraction=.25)
        report = evaluate(model, [dict(series=torch.ones(3, 1, 1),
                                       forecast_targets=torch.tensor([-1., 1., 3.])[:, None, None])],
                          calibration=asymmetric)
        self.assertAlmostEqual(report["calibrated_interval_score"], 2 + (20 + 1 / .15) / 3)
        self.assertEqual(report["calibrated_lower_tail_fraction"], .25)

    def test_per_variable_and_coordinate_coverage_exposes_hidden_undercoverage(self):
        class FixedForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=2, series_features=3)

            def forward(self, series, **kwargs):
                forecast = series.new_tensor([-1., 0., 1.]).expand(series.size(0), 2, 3, 3)
                return SimpleNamespace(forecast=forecast, loss=None, losses={"forecast": self.anchor * 0})

        model = FixedForecast()
        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[0., 1., 0.], [0., 1., 0.]],
                                     [[20, 20, 20], [20, 20, 20]], ("series",))
        targets = torch.tensor([[[0., -2., float("nan")], [.2, 3., float("nan")]],
                                [[-1., 2., float("nan")], [0., -3., float("nan")]],
                                [[.5, .5, float("nan")], [float("nan"), float("nan"), float("nan")]]])
        batch = dict(series=torch.ones(3, 1, 3), forecast_targets=targets)
        full = evaluate(model, [batch], calibration=fitted)
        split = evaluate(model, [{key: value[section] for key, value in batch.items()}
                                 for section in (slice(0, 1), slice(1, 3))], calibration=fitted)
        self.assertEqual(full["forecast_observations_by_horizon"], [6, 4])
        self.assertEqual(full["forecast_observations_by_feature"], [5, 5, 0])
        self.assertEqual(full["forecast_observations_by_horizon_feature"], [[3, 3, 0], [2, 2, 0]])
        self.assertEqual(full["forecast_interval_coverage"], 0.6)
        self.assertEqual(full["forecast_interval_coverage_by_feature"], [1., .2, None])
        self.assertEqual(full["calibrated_interval_coverage_by_feature"], [1., .6, None])
        self.assertEqual(full["forecast_interval_coverage_by_horizon_feature"], [[1., 1 / 3, None], [1., 0., None]])
        self.assertEqual(full["calibrated_interval_coverage_by_horizon_feature"], [[1., 1., None], [1., 0., None]])
        self.assertEqual(full["forecast_interval_below_by_feature"], [0., .4, None])
        self.assertEqual(full["forecast_interval_above_by_feature"], [0., .4, None])
        self.assertEqual(full["calibrated_interval_below_by_feature"], [0., .2, None])
        self.assertEqual(full["calibrated_interval_above_by_feature"], [0., .2, None])
        self.assertEqual(full["forecast_quantile_levels"], [0.1, 0.5, 0.9])
        finite_targets = [float(value) for value in targets.flatten() if torch.isfinite(value)]
        for index, (quantile, predicted) in enumerate(zip((.1, .5, .9), (-1., 0., 1.))):
            expected_coverage = sum(value <= predicted for value in finite_targets) / len(finite_targets)
            expected_pinball = sum(max(quantile * (value - predicted), (quantile - 1) * (value - predicted))
                                   for value in finite_targets) / len(finite_targets)
            self.assertAlmostEqual(full["forecast_quantile_coverage"][index], expected_coverage)
            self.assertAlmostEqual(full["forecast_pinball_by_quantile"][index], expected_pinball)
            for feature in (0, 1):
                observed = [float(value) for value in targets[:, :, feature].flatten() if torch.isfinite(value)]
                expected = sum(value <= predicted for value in observed) / len(observed)
                self.assertAlmostEqual(full["forecast_quantile_coverage_by_feature"][feature][index], expected)
                expected_pinball = sum(max(quantile * (value - predicted), (quantile - 1) * (value - predicted))
                                       for value in observed) / len(observed)
                self.assertAlmostEqual(full["forecast_pinball_by_feature_quantile"][feature][index], expected_pinball)
        self.assertEqual(full["forecast_quantile_coverage_by_feature"][2], [None, None, None])
        self.assertEqual(full["forecast_pinball_by_feature_quantile"][2], [None, None, None])

        def assert_nested_close(first, second):
            if isinstance(first, list):
                self.assertEqual(len(first), len(second))
                for a, b in zip(first, second):
                    assert_nested_close(a, b)
            elif first is None:
                self.assertIsNone(second)
            else:
                self.assertAlmostEqual(first, second, places=12)

        for key in ("forecast_interval_coverage_by_feature", "calibrated_interval_coverage_by_feature",
                    "forecast_interval_coverage_by_horizon_feature", "forecast_quantile_coverage",
                    "forecast_pinball_by_quantile", "forecast_mae_by_feature", "forecast_mae_by_horizon_feature",
                    "forecast_quantile_coverage_by_feature", "forecast_pinball_by_feature_quantile"):
            with self.subTest(metric=key):
                assert_nested_close(full[key], split[key])

    def test_quantile_diagnostics_support_median_only_and_missing_targets(self):
        class MedianForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=1, series_features=1,
                                      quantiles=(0.5,))

            def forward(self, series, **kwargs):
                return SimpleNamespace(forecast=series.new_zeros(series.size(0), 1, 1, 1), loss=None,
                                       losses={"forecast": self.anchor * 0})

        model = MedianForecast()
        batch = dict(series=torch.ones(2, 1, 1), forecast_targets=torch.tensor([[[-2.]], [[2.]]]))
        report = evaluate(model, [batch])
        self.assertEqual(report["forecast_quantile_levels"], [.5])
        self.assertEqual(report["forecast_quantile_coverage"], [.5])
        self.assertEqual(report["forecast_pinball_by_quantile"], [1.])
        self.assertNotIn("forecast_interval_coverage", report)
        batch["forecast_targets"].fill_(float("nan"))
        empty = evaluate(model, [batch])
        self.assertEqual(empty["forecast_quantile_coverage"], [None])
        self.assertEqual(empty["forecast_pinball_by_quantile"], [None])
        self.assertEqual(empty["forecast_mae_by_horizon_feature"], [[None]])

    def test_forecast_metrics_do_not_overflow_half_precision_subtraction(self):
        class FixedForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=1, series_features=1)

            def forward(self, series, **kwargs):
                forecast = series.new_tensor([-60000, 60000, 60000]).expand(series.size(0), 1, 1, 3)
                return SimpleNamespace(forecast=forecast, loss=None, losses={"forecast": self.anchor * 0})

        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[0.]], [[20]], ("series",))
        batch = dict(series=torch.ones(1, 1, 1, dtype=torch.float16),
                     forecast_targets=torch.tensor([[[-60000]]], dtype=torch.float16))
        report = evaluate(FixedForecast(), [batch], calibration=fitted)
        for metric in ("forecast_mae", "forecast_rmse", "forecast_interval_width", "forecast_interval_score",
                       "calibrated_interval_width", "calibrated_interval_score"):
            self.assertEqual(report[metric], 120000, metric)
        self.assertEqual(report["forecast_interval_coverage"], 1)

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

    def test_joint_calibration_uses_maximum_score_and_complete_trajectories(self):
        values = torch.arange(10, dtype=torch.float64)[:, None, None]
        scores = values + torch.tensor([[0., 2.], [-1., 1.]], dtype=torch.float64)
        scores[0, 0, 0] = float("nan")
        fitted = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",), scope="joint")
        self.assertEqual(fitted.scope, "joint")
        torch.testing.assert_close(fitted.counts, torch.full((2, 2), 9))
        torch.testing.assert_close(fitted.correction, torch.full((2, 2), 10., dtype=torch.float64))
        restored = ForecastCalibration.from_dict(json.loads(json.dumps(fitted.to_dict())))
        self.assertEqual(restored.to_dict()["format_version"], 2)
        self.assertEqual(restored.scope, "joint")
        torch.testing.assert_close(restored.correction, fitted.correction)
        with self.assertRaisesRegex(ValueError, "trajectoires complètes"):
            ForecastCalibration.from_scores(scores[:3], (0.1, 0.5, 0.9), ("series",), scope="joint")
        broken = fitted.to_dict()
        broken["correction"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "correction et un effectif communs"):
            ForecastCalibration.from_dict(broken)
        broken = fitted.to_dict()
        broken["counts"][0][0] -= 1
        with self.assertRaisesRegex(ValueError, "correction et un effectif communs"):
            ForecastCalibration.from_dict(broken)

    def test_joint_coverage_on_independent_multivariate_samples(self):
        generator = torch.Generator().manual_seed(830)
        targets = torch.randn(7000, 3, 2, generator=generator)
        forecasts = torch.tensor([-0.5, 0., 0.5]).expand(7000, 3, 2, 3)
        scores = conformity_scores(forecasts[:1000], targets[:1000])
        joint = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",), scope="joint")
        marginal = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",))

        def coverage(calibration):
            interval = calibration.interval(forecasts[1000:])
            inside = (targets[1000:] >= interval[..., 0]) & (targets[1000:] <= interval[..., 1])
            return inside.flatten(1).all(-1).float().mean().item(), (interval[..., 1] - interval[..., 0]).mean().item()

        joint_coverage, joint_width = coverage(joint)
        marginal_coverage, marginal_width = coverage(marginal)
        self.assertAlmostEqual(joint_coverage, 0.8, delta=0.045)
        self.assertGreater(joint_coverage, marginal_coverage + 0.25)
        self.assertGreater(joint_width, marginal_width)

    def test_joint_fitting_is_batch_invariant_with_missing_trajectories(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        batch = synthetic_batch(model.config, batch_size=12, seed=1231)
        batch["forecast_targets"][:2, 0, 0] = float("nan")
        chunks = [{key: value[section] for key, value in batch.items()}
                  for section in (slice(0, 4), slice(4, 12))]
        for amp in ("none", "bf16"):
            for scale in ("none", "width"):
                for fraction in (None, .5):
                    with self.subTest(amp=amp, scale=scale, fraction=fraction):
                        full = fit_calibration(model, [batch], scope="joint", score_scale=scale, amp=amp,
                                               lower_tail_fraction=fraction)
                        split = fit_calibration(model, chunks, scope="joint", score_scale=scale, amp=amp,
                                                lower_tail_fraction=fraction)
                        self.assertEqual(full.counts[0, 0].item(), 10)
                        torch.testing.assert_close(full.correction, split.correction, rtol=2e-5, atol=2e-6)
                        if fraction is not None:
                            torch.testing.assert_close(full.upper_correction, split.upper_correction, rtol=2e-5, atol=2e-6)
        self.assertTrue(model.training)

    def test_joint_interval_coverage_uses_only_complete_examples(self):
        class FixedForecast(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.config = replace(ChimeraConfig.preset("tiny"), forecast_horizon=2, series_features=2)

            def forward(self, series, **kwargs):
                forecast = series.new_tensor([-1., 0., 1.]).expand(series.size(0), 2, 2, 3)
                return SimpleNamespace(forecast=forecast, loss=None, losses={"forecast": self.anchor * 0})

        calibration = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[1., 1.], [1., 1.]],
                                          [[20, 20], [20, 20]], ("series",), scope="joint")
        targets = torch.zeros(4, 2, 2)
        targets[1, 1, 1] = 2
        targets[2, 0, 0] = float("nan")
        targets[3] = float("nan")
        batch = dict(series=torch.ones(4, 1, 2), forecast_targets=targets)
        report = evaluate(FixedForecast(), [batch], calibration=calibration)
        self.assertEqual(report["forecast_complete_examples"], 2)
        self.assertEqual(report["forecast_joint_interval_coverage"], 0.5)
        self.assertEqual(report["calibrated_joint_interval_coverage"], 1)
        self.assertEqual(report["calibrated_coverage_scope"], "joint")
        self.assertEqual(report["forecast_observations_by_feature"], [5, 6])
        self.assertEqual(report["forecast_interval_coverage_by_feature"], [1., 5 / 6])
        self.assertEqual(report["calibrated_interval_coverage_by_feature"], [1., 1.])
        self.assertIsNone(report["calibrated_interval_score"])

    def test_width_normalized_joint_calibration_respects_per_variable_units(self):
        forecast = torch.tensor([-1., 0., 1.], dtype=torch.float64).expand(10, 2, 2, 3).clone()
        target = (torch.arange(10, dtype=torch.float64) - 4.5)[:, None, None].expand(10, 2, 2).clone()
        target[0, 0, 0] = float("nan")
        gain = torch.tensor([0.1, 10.], dtype=torch.float64)
        shift = torch.tensor([-7., 20.], dtype=torch.float64)
        changed_forecast = forecast * gain[None, None, :, None] + shift[None, None, :, None]
        changed_target = target * gain[None, None, :] + shift[None, None, :]
        scores = conformity_scores(forecast, target, score_scale="width", scale_floor=1e-8)
        changed_scores = conformity_scores(changed_forecast, changed_target, score_scale="width", scale_floor=1e-8)
        torch.testing.assert_close(scores, changed_scores, rtol=1e-12, atol=1e-12, equal_nan=True)
        base = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",),
                                               scope="joint", score_scale="width", scale_floor=1e-8)
        changed = ForecastCalibration.from_scores(changed_scores, (0.1, 0.5, 0.9), ("series",),
                                                  scope="joint", score_scale="width", scale_floor=1e-8)
        torch.testing.assert_close(base.correction, changed.correction, rtol=1e-12, atol=1e-12)
        expected = base.interval(forecast) * gain[None, None, :, None] + shift[None, None, :, None]
        torch.testing.assert_close(changed.interval(changed_forecast), expected, rtol=1e-12, atol=1e-12)
        restored = ForecastCalibration.from_dict(json.loads(json.dumps(base.to_dict())))
        self.assertEqual(restored.to_dict()["format_version"], 3)
        self.assertEqual(restored.score_scale, "width")
        self.assertEqual(restored.scale_floor, 1e-8)
        torch.testing.assert_close(restored.interval(forecast), base.interval(forecast), rtol=0, atol=0)

    def test_width_floor_handles_degenerate_intervals_and_invalid_metadata(self):
        forecast = torch.zeros(10, 1, 1, 3)
        target = torch.ones(10, 1, 1)
        scores = conformity_scores(forecast, target, score_scale="width", scale_floor=1e-4)
        fitted = ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",), scope="joint",
                                                 score_scale="width", scale_floor=1e-4)
        interval = fitted.interval(forecast)
        self.assertTrue(torch.isfinite(interval).all())
        self.assertTrue((interval[..., 1] >= target).all())
        for changes in (dict(scope="marginal", score_scale="width"), dict(scope="joint", score_scale="invalid"),
                        dict(scope="joint", score_scale="width", scale_floor=0),
                        dict(scope="joint", score_scale="width", scale_floor=float("nan")),
                        dict(scope="joint", score_scale="none", scale_floor=0.1)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ForecastCalibration.from_scores(scores, (0.1, 0.5, 0.9), ("series",), **changes)
        for corrupt in (dict(format_version=3, scope="joint", score_scale="width"),
                        dict(format_version=2, scope="joint", score_scale="width")):
            with self.assertRaisesRegex(ValueError, "Version et portée"):
                ForecastCalibration.from_dict(corrupt)

    def test_negative_correction_shrinks_and_never_inverts_interval_or_changes_quantiles(self):
        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, [[-2.]], [[20]], ("series",))
        forecasts = torch.tensor([[[[-1., 0.25, 1.]]]])
        original = forecasts.clone()
        interval = fitted.interval(forecasts)
        torch.testing.assert_close(interval, torch.tensor([[[[0.25, 0.25]]]]))
        torch.testing.assert_close(forecasts, original, rtol=0, atol=0)
        fitted.correction.fill_(0.5)
        torch.testing.assert_close(fitted.interval(forecasts), torch.tensor([[[[-1.5, 1.5]]]]))

    def test_float32_bounds_round_outward_for_calibration_boundary(self):
        upper = torch.tensor(-0.001368486089631915, dtype=torch.float32)
        target = torch.tensor(-9.519225386611652e-06, dtype=torch.float32)
        correction = target.double() - upper.double()
        forecast = torch.tensor([[[[-0.1, -0.01, upper.item()]]]], dtype=torch.float32)
        fitted = ForecastCalibration((0.1, 0.5, 0.9), 0.8, correction.reshape(1, 1), [[20]], ("series",))
        interval = fitted.interval(forecast)
        self.assertEqual(interval.dtype, torch.float32)
        self.assertGreaterEqual(interval[0, 0, 0, 1].item(), target.item())
        self.assertLess((interval[0, 0, 0, 1].double() - target.double()).item(), 1e-10)

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
                       dict(dataset_images_sha256="a" * 64),
                       dict(dataset_images_sha256="invalid", dataset_images_recorded=True),
                       dict(dataset_images_sha256=123, dataset_images_recorded=True),
                       dict(dataset_images_recorded=1),
                       dict(quantiles=(0.1, 0.9)), dict(scope="invalid")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ForecastCalibration(**(valid | change))
        for forecast in (torch.tensor([[[[1., 0., -1.]]]]), torch.full((1, 1, 1, 3), float("nan"))):
            with self.assertRaises(ValueError):
                conformity_scores(forecast, torch.zeros(1, 1, 1))
        with self.assertRaises(ValueError):
            ForecastCalibration.from_dict(dict(format_version=999))
        with self.assertRaisesRegex(ValueError, "Version et portée"):
            ForecastCalibration.from_dict(dict(format_version=1, scope="joint"))

    def test_image_provenance_is_optional_and_roundtrips_all_formats(self):
        common = dict(quantiles=(0.1, 0.5, 0.9), coverage=0.8, correction=[[0.]], counts=[[20]],
                      modalities=("series",))
        for options in ({}, {"scope": "joint"}, {"scope": "joint", "score_scale": "width"}):
            for digest in (None, "a" * 64):
                with self.subTest(options=options, digest=digest):
                    fitted = ForecastCalibration(**common, **options, dataset_images_recorded=True,
                                                 dataset_images_sha256=digest)
                    loaded = ForecastCalibration.from_dict(fitted.to_dict())
                    self.assertTrue(loaded.dataset_images_recorded)
                    self.assertEqual(loaded.dataset_images_sha256, digest)
                    legacy = fitted.to_dict()
                    del legacy["dataset_images_sha256"]
                    self.assertFalse(ForecastCalibration.from_dict(legacy).dataset_images_recorded)


if __name__ == "__main__":
    unittest.main()
