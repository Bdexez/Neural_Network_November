import unittest

import torch

from chimera.augmentation import SeriesAffineAugmentation


class AffineAugmentationTests(unittest.TestCase):
    def test_transforms_history_and_future_together_without_mutation_or_global_rng(self):
        series = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
        future = series[:, -2:].clone() + 10
        series[:, 0, 1] = float("nan")
        future[0, -1, 0] = float("nan")
        batch = dict(series=series, forecast_targets=future, series_mask=torch.ones(2, 5, dtype=torch.bool),
                     class_labels=torch.tensor([0, 1]))
        before = {key: value.clone() for key, value in batch.items()}
        rng = torch.get_rng_state().clone()
        augmentation = SeriesAffineAugmentation(probability=1)
        changed = augmentation(batch, index=4, seed=5601)
        repeated = augmentation(batch, index=4, seed=5601)
        gain = (changed["series"][:, -1] - changed["series"][:, -2]) / (series[:, -1] - series[:, -2])
        shift = changed["series"][:, -1] - gain * series[:, -1]
        self.assertTrue((gain > 0).all())
        torch.testing.assert_close(changed["forecast_targets"], future * gain[:, None] + shift[:, None], equal_nan=True)
        for key in batch:
            torch.testing.assert_close(batch[key], before[key], rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(changed[key], repeated[key], rtol=0, atol=0, equal_nan=True)
        self.assertIs(changed["series_mask"], batch["series_mask"])
        self.assertIs(changed["class_labels"], batch["class_labels"])
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertIs(SeriesAffineAugmentation()(batch, index=4, seed=5601), batch)

    def test_rejects_invalid_ranges_mismatched_targets_and_overflows(self):
        for settings in (dict(probability=-0.1), dict(probability=float("nan")), dict(min_scale=0),
                         dict(min_scale=2, max_scale=1), dict(max_shift=-1), dict(max_scale=float("inf"))):
            with self.assertRaises(ValueError):
                SeriesAffineAugmentation(**settings)
        augmentation = SeriesAffineAugmentation(probability=1, min_scale=10, max_scale=10, max_shift=0)
        with self.assertRaisesRegex(ValueError, "forecast_targets"):
            augmentation(dict(series=torch.ones(2, 4, 3), forecast_targets=torch.ones(2, 3, 2)), index=0, seed=1)
        with self.assertRaisesRegex(ValueError, "précision"):
            augmentation(dict(series=torch.full((1, 2, 3), torch.finfo(torch.float32).max)), index=0, seed=1)


if __name__ == "__main__":
    unittest.main()
