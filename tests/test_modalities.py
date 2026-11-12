import itertools
import random
import unittest

import torch

from chimera.modalities import MODALITIES, balanced_modalities, keep_modalities


class ModalityTests(unittest.TestCase):
    def setUp(self):
        self.batch = {name: torch.tensor([1]) for name in
                      (*MODALITIES, "labels", "attention_mask", "series_mask",
                       "class_labels", "forecast_targets")}

    @staticmethod
    def signature(batch):
        return tuple(name for name in MODALITIES if name in batch)

    def test_every_cycle_covers_all_seven_subsets_without_mutation(self):
        expected = {subset for n in range(1, 4) for subset in itertools.combinations(MODALITIES, n)}
        for cycle in range(3):
            selected = [balanced_modalities(self.batch, cycle * 7 + i, seed=42) for i in range(7)]
            self.assertEqual({self.signature(batch) for batch in selected}, expected)
            for batch in selected:
                self.assertIn("class_labels", batch)
                self.assertIn("forecast_targets", batch)
                self.assertEqual("labels" in batch, "input_ids" in batch)
                self.assertEqual("attention_mask" in batch, "input_ids" in batch)
                self.assertEqual("series_mask" in batch, "series" in batch)
        self.assertEqual(len(self.batch), 8)

    def test_language_only_supervision_always_retains_text(self):
        batch = {key: value for key, value in self.batch.items()
                 if key not in ("class_labels", "forecast_targets")}
        expected = {subset for n in range(1, 4) for subset in itertools.combinations(MODALITIES, n)
                    if "input_ids" in subset}
        self.assertEqual({self.signature(balanced_modalities(batch, i)) for i in range(4)}, expected)

    def test_one_available_modality_and_invalid_batches(self):
        batch = {"series": self.batch["series"], "forecast_targets": self.batch["forecast_targets"]}
        self.assertEqual(self.signature(balanced_modalities(batch, 1000)), ("series",))
        for invalid in ({}, {"class_labels": torch.tensor([1])}, {"series": torch.ones(1)}):
            with self.assertRaises(ValueError):
                balanced_modalities(invalid, 0)
        for index in (-1, 0.5, True):
            with self.assertRaises(ValueError):
                balanced_modalities(batch, index)

    def test_resume_and_distributed_partition_preserve_global_schedule_and_rng(self):
        rng = random.getstate()
        expected = [self.signature(balanced_modalities(self.batch, i, seed=9)) for i in range(28)]
        resumed = [self.signature(balanced_modalities(self.batch, i, seed=9)) for i in range(11, 28)]
        self.assertEqual(resumed, expected[11:])
        self.assertEqual(random.getstate(), rng)
        ranks = [[self.signature(balanced_modalities(self.batch, i * 2 + rank, seed=9))
                  for i in range(14)] for rank in range(2)]
        self.assertEqual([ranks[i % 2][i // 2] for i in range(28)], expected)
        changed = [self.signature(balanced_modalities(self.batch, i, seed=10)) for i in range(28)]
        self.assertNotEqual(expected, changed)

    def test_keep_modalities_preserves_tensor_identity(self):
        kept = keep_modalities(self.batch, ("series",))
        self.assertIs(kept["series"], self.batch["series"])
        self.assertIs(kept["forecast_targets"], self.batch["forecast_targets"])


if __name__ == "__main__":
    unittest.main()
