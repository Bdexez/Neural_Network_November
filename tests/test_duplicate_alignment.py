from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import distributed as dist

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.distributed_alignment import distributed_alignment_loss
from chimera.losses import symmetric_contrastive_loss


def distributed_worker(rank, group_file, output, partial):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{group_file}', rank=rank, world_size=2)
    try:
        left = torch.tensor([[.8, .2], [.1, .9]], dtype=torch.float32)[rank:rank+1].clone().requires_grad_()
        right = torch.tensor([[.5, .3], [.2, .8]], dtype=torch.float32)[rank:rank+1].clone().requires_grad_()
        ids = None if partial and rank == 1 else torch.tensor([7], dtype=torch.long)
        loss = distributed_alignment_loss({0: left, 1: right}, None, .5, 'paired',
                                          group=dist.group.WORLD, sample_ids=ids)
        loss.backward()
        torch.save({'loss': loss.detach(), 'left': left.grad, 'right': right.grad}, Path(output) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


class DuplicateAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_duplicate_positives_and_unknown_ids(self):
        scores = torch.zeros(5, 5, requires_grad=True)
        ids = torch.tensor([7, 7, 9, -1, -1])
        loss = symmetric_contrastive_loss(scores, sample_ids=ids)
        loss.backward()
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertGreater(scores.grad[0, 2].item(), 0)
        self.assertGreater(scores.grad[3, 4].item(), 0)
        reference = symmetric_contrastive_loss(scores.detach(), torch.tensor([0, 0, 1, -100, -100]))
        torch.testing.assert_close(loss.detach(), reference)

    def test_class_and_identity_positives_are_unioned(self):
        scores = torch.zeros(4, 4, requires_grad=True)
        ids = torch.tensor([1, 1, 2, -1])
        labels = torch.tensor([0, 1, 1, -100])
        symmetric_contrastive_loss(scores, labels, sample_ids=ids).backward()
        self.assertLess(scores.grad[0, 1].item(), 0)
        self.assertLess(scores.grad[1, 2].item(), 0)
        self.assertGreater(scores.grad[0, 2].item(), 0)
        self.assertGreater(scores.grad[2, 3].item(), 0)

    def test_model_predictions_unchanged_and_identity_validation(self):
        config = replace(ChimeraConfig.preset('tiny'), alignment_mode='paired')
        torch.manual_seed(72)
        model = Chimera(config).eval()
        batch = synthetic_batch(config, batch_size=3, seed=73)
        with torch.no_grad():
            baseline = model(**batch)
            grouped = model(**batch, alignment_ids=torch.tensor([4, 4, 9]))
        for name in ('logits', 'classification_logits', 'forecast'):
            torch.testing.assert_close(getattr(baseline, name), getattr(grouped, name), rtol=0, atol=0)
        self.assertNotEqual(baseline.losses['alignment'].item(), grouped.losses['alignment'].item())
        for invalid in (torch.tensor([0, -2, 1]), torch.tensor([0., 0., 1.]), torch.tensor([0, 1])):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'alignment_ids'):
                model(**batch, alignment_ids=invalid)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_distributed_matches_full_reference_and_unknown_rank(self):
        for partial in (False, True):
            with self.subTest(partial=partial), tempfile.TemporaryDirectory() as directory:
                torch.multiprocessing.spawn(distributed_worker,
                    args=(str(Path(directory) / 'group'), directory, partial), nprocs=2, join=True)
                left = torch.tensor([[.8, .2], [.1, .9]], requires_grad=True)
                right = torch.tensor([[.5, .3], [.2, .8]], requires_grad=True)
                ids = torch.tensor([7, -1 if partial else 7])
                expected = symmetric_contrastive_loss(left @ right.T / .5, sample_ids=ids)
                expected.backward()
                for rank in range(2):
                    result = torch.load(Path(directory) / f'{rank}.pt', weights_only=True)
                    torch.testing.assert_close(result['loss'], expected.detach())
                    torch.testing.assert_close(result['left'] / 2, left.grad[rank:rank+1])
                    torch.testing.assert_close(result['right'] / 2, right.grad[rank:rank+1])


if __name__ == '__main__':
    unittest.main()
