from types import SimpleNamespace
import math
import unittest

import torch

from chimera.evaluation import evaluate
from chimera.streaming_rmse import StreamingRMSE


class ConstantForecast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(forecast_horizon=1, series_features=1, num_classes=2,
                                      quantiles=(.1, .5, .9), forecast_weight=1)

    def forward(self, forecast_targets):
        return SimpleNamespace(forecast=torch.zeros((*forecast_targets.shape, 3), dtype=torch.float64),
                               loss=torch.tensor(0.), losses={'forecast': torch.tensor(0.)})


class ForecastRmsePrecisionTests(unittest.TestCase):
    def test_representable_rmse_does_not_overflow_squares(self):
        batch = {'forecast_targets': torch.tensor([[[1e200]], [[-1e200]]], dtype=torch.float64)}
        report = evaluate(ConstantForecast(), [batch])
        self.assertEqual(report['forecast_mae'], 1e200)
        self.assertEqual(report['forecast_rmse'], 1e200)

    def test_normal_values_match_reference_and_nonfinite_errors_propagate(self):
        values = torch.tensor([0.0, 3.0, -4.0, 12.0], dtype=torch.float64)
        accumulator = StreamingRMSE()
        accumulator.update(values[:2])
        accumulator.update(values[2:])
        self.assertAlmostEqual(accumulator.result(len(values)), 6.5, places=14)
        accumulator.update(torch.tensor([float('inf')], dtype=torch.float64))
        self.assertTrue(math.isinf(accumulator.result(5)))
        accumulator.update(torch.tensor([float('nan')], dtype=torch.float64))
        self.assertTrue(math.isnan(accumulator.result(6)))
        zero = StreamingRMSE()
        zero.update(torch.zeros(2, dtype=torch.float64))
        self.assertEqual(zero.result(2), 0)
        self.assertIsNone(zero.result(0))

    def test_scaling_across_batches_and_missing_values(self):
        targets = torch.tensor([0., 1e-200, float('nan'), -3e200, 4e200], dtype=torch.float64).view(-1, 1, 1)
        model = ConstantForecast()
        report = evaluate(model, [{'forecast_targets': targets}])
        reverse = evaluate(model, [{'forecast_targets': targets[3:]}, {'forecast_targets': targets[:3]}])
        # sqrt((3² + 4²) * 1e400 / 4) = 2.5e200 ; NaN is ignored.
        self.assertEqual(report['forecast_observations'], 4)
        self.assertAlmostEqual(report['forecast_rmse'] / 1e200, 2.5, places=14)
        self.assertAlmostEqual(reverse['forecast_rmse'] / 1e200, 2.5, places=14)
        tiny = evaluate(model, [{'forecast_targets': targets[:2]}])
        self.assertAlmostEqual(tiny['forecast_rmse'] / 1e-200, 1 / math.sqrt(2), places=14)
        missing = evaluate(model, [{'forecast_targets': targets[2:3]}])
        self.assertIsNone(missing['forecast_rmse'])


if __name__ == '__main__':
    unittest.main()
