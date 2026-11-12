from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.losses import symmetric_contrastive_loss


class AlignmentTests(unittest.TestCase):
    def test_paired_and_distinct_classes_match_original_objective(self):
        scores = torch.randn(5, 5, generator=torch.Generator().manual_seed(1))
        ids = torch.arange(5)
        expected = (F.cross_entropy(scores, ids) + F.cross_entropy(scores.T, ids)) / 2
        torch.testing.assert_close(symmetric_contrastive_loss(scores), expected, rtol=0, atol=0)
        torch.testing.assert_close(symmetric_contrastive_loss(scores, ids), expected)

    def test_same_class_matches_are_attracted_and_unknowns_remain_individual(self):
        scores = torch.zeros(4, 4, requires_grad=True)
        symmetric_contrastive_loss(scores, torch.tensor([1, 1, -100, -100])).backward()
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertGreater(scores.grad[0, 2].item(), 0)
        self.assertGreater(scores.grad[2, 3].item(), 0)
        self.assertLess(scores.grad[2, 2].item(), 0)

    def test_permutation_invariance_and_identical_class_stationary_point(self):
        scores = torch.randn(5, 5, generator=torch.Generator().manual_seed(8))
        labels = torch.tensor([2, -100, 2, 3, 3])
        permutation = torch.tensor([4, 0, 2, 1, 3])
        torch.testing.assert_close(symmetric_contrastive_loss(scores, labels),
                                   symmetric_contrastive_loss(scores[permutation][:, permutation],
                                                              labels[permutation]))
        equal = torch.zeros(3, 3, requires_grad=True)
        symmetric_contrastive_loss(equal, torch.zeros(3, dtype=torch.long)).backward()
        torch.testing.assert_close(equal.grad, torch.zeros_like(equal), atol=1e-7, rtol=0)

    def test_extreme_scores_and_single_example_are_finite(self):
        scores = torch.tensor([[10000, -10000], [-10000, 10000]], dtype=torch.float16, requires_grad=True)
        loss = symmetric_contrastive_loss(scores, torch.tensor([1, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(symmetric_contrastive_loss(torch.tensor([[10.0]]), torch.tensor([1])).item(), 0)

    def test_mode_preserves_weight_compatibility_and_never_conditions_predictions_on_labels(self):
        torch.set_num_threads(2)
        config = ChimeraConfig.preset("tiny")
        paired = Chimera(config).eval()
        grouped = Chimera(replace(config, alignment_mode="class")).eval()
        grouped.load_state_dict(paired.state_dict(), strict=True)
        batch = synthetic_batch(config, batch_size=3)
        with torch.no_grad():
            before = grouped(**batch)
            batch["class_labels"] = torch.tensor([0, 0, 0])
            after = grouped(**batch)
        torch.testing.assert_close(before.classification_logits, after.classification_logits, rtol=0, atol=0)
        torch.testing.assert_close(before.forecast, after.forecast, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(after.loss))
        for change in (dict(alignment_mode="invalid"), dict(alignment_temperature=0),
                       dict(alignment_temperature=float("nan"))):
            with self.assertRaises(ValueError):
                replace(config, **change)


if __name__ == "__main__":
    unittest.main()
