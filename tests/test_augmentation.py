import random
import unittest

import torch

from chimera.augmentation import SeriesAugmentation


class AugmentationTests(unittest.TestCase):
    def test_crop_preserves_recent_valid_positions_targets_and_original_inputs(self):
        series = torch.arange(5 * 12 * 2).float().reshape(5, 12, 2)
        valid = torch.ones(5, 12, dtype=torch.bool)
        valid[1, :3] = False
        valid[2, 5:] = False
        valid[3, [1, 5, 8]] = False
        valid[4, 2:] = False
        batch = dict(series=series, series_mask=valid, forecast_targets=torch.randn(5, 4, 2))
        result = SeriesAugmentation(crop_probability=1, min_history=4)(batch, index=3, seed=9)
        self.assertIs(result["forecast_targets"], batch["forecast_targets"])
        torch.testing.assert_close(series, torch.arange(120).float().reshape(5, 12, 2))
        torch.testing.assert_close(batch["series_mask"], valid)
        self.assertTrue(torch.isnan(result["series"][~result["series_mask"]]).all())
        for row in range(5):
            original_indices = valid[row].nonzero().flatten()
            retained = result["series_mask"][row].nonzero().flatten()
            self.assertGreaterEqual(len(retained), min(4, len(original_indices)))
            torch.testing.assert_close(retained, original_indices[-len(retained):])
            torch.testing.assert_close(result["series"][row, retained], series[row, retained])

    def test_missing_values_do_not_modify_masks_or_existing_targets(self):
        batch = dict(series=torch.ones(4, 20, 3), series_mask=torch.ones(4, 20, dtype=torch.bool),
                     forecast_targets=torch.randn(4, 12, 3), input_ids=torch.tensor([[1, 2]] * 4))
        batch["series"][0, 0, 0] = float("nan")
        result = SeriesAugmentation(missing_rate=0.5, mask_probability=1)(batch, index=1, seed=8)
        self.assertIs(result["series_mask"], batch["series_mask"])
        self.assertIs(result["input_ids"], batch["input_ids"])
        self.assertIs(result["forecast_targets"], batch["forecast_targets"])
        self.assertTrue(torch.isnan(result["series"][0, 0, 0]))
        self.assertGreater(torch.isnan(result["series"]).sum().item(), 1)
        self.assertTrue(torch.isfinite(result["series"]).any())
        missing = SeriesAugmentation(missing_rate=1, mask_probability=1)(batch, index=1, seed=8)
        self.assertTrue(torch.isnan(missing["series"]).all())
        self.assertTrue(missing["series_mask"].all())

    def test_determinism_independent_streams_and_global_rng_preserved(self):
        batch = dict(series=torch.ones(16, 32, 3))
        torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
        augmentation = SeriesAugmentation(crop_probability=0.5, missing_rate=0.2, min_history=8)
        first = augmentation(batch, index=4, seed=42)
        repeated = augmentation(batch, index=4, seed=42)
        torch.testing.assert_close(first["series"], repeated["series"], equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(first["series_mask"], repeated["series_mask"], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_state)
        changed = augmentation(batch, index=5, seed=42)
        self.assertFalse(torch.equal(torch.isnan(first["series"]), torch.isnan(changed["series"])))
        missing_only = SeriesAugmentation(missing_rate=0.2)(batch, index=4, seed=42)
        kept = first["series_mask"].unsqueeze(-1).expand_as(first["series"])
        torch.testing.assert_close(first["series"][kept], missing_only["series"][kept], equal_nan=True)

    def test_disabled_augmentation_and_invalid_arguments(self):
        batch = dict(series=torch.ones(2, 10, 1))
        self.assertIs(SeriesAugmentation()(batch, index=0, seed=0), batch)
        text = dict(input_ids=torch.ones(1, 3, dtype=torch.long))
        self.assertIs(SeriesAugmentation(crop_probability=1)(text, index=0, seed=0), text)
        for change in (dict(crop_probability=-1), dict(missing_rate=1.1),
                       dict(missing_rate=float("nan")), dict(mask_probability=-0.1),
                       dict(min_history=0), dict(min_history=True)):
            with self.assertRaises(ValueError):
                SeriesAugmentation(**change)
        with self.assertRaises(ValueError):
            SeriesAugmentation(crop_probability=1)(batch, index=-1, seed=0)


if __name__ == "__main__":
    unittest.main()
