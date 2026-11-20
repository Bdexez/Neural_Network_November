from pathlib import Path
import tempfile
import unittest

import torch
from torch import distributed as dist
from torch.nn import functional as F

from chimera.distributed_alignment import distributed_alignment_loss


LEFT = torch.tensor([[.8, .1, .3], [.2, .9, .4], [.1, .3, .9]], dtype=torch.float32)
RIGHT = torch.tensor([[.7, .2, .4], [.3, .8, .1], [.2, .2, .8]], dtype=torch.float32)


def worker(rank, group_file, output, case):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{group_file}', rank=rank, world_size=2)
    try:
        if case in ('rectangular', 'class'):
            values = LEFT.clone() if rank == 0 else RIGHT[:2].clone()
            ids = torch.tensor([7, 7, 9] if rank == 0 else [7, 9])
        elif case == 'unmatched':
            values = LEFT.clone() if rank == 0 else RIGHT.clone()
            ids = torch.tensor([7, 9, -1] if rank == 0 else [7, 9, 11])
        else:
            values = LEFT[:2].clone() if rank == 0 else RIGHT[:2].clone()
            ids = (None if case == 'no_ids' else torch.tensor([11, 12] if case == 'disjoint' and rank == 1 else [7, 9]))
        values.requires_grad_()
        vectors = {0: values} if rank == 0 else {1: values}
        labels = torch.tensor([0, 1, 1] if rank == 0 else [0, 1]) if case == 'class' else None
        loss = distributed_alignment_loss(vectors, labels, .4, 'class' if case == 'class' else 'paired',
                                          group=dist.group.WORLD, sample_ids=ids,
                                          cross_rank_views=case != 'disabled')
        loss.backward()
        torch.save({'loss': loss.detach(), 'grad': values.grad}, Path(output) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


def expected_loss(left, right, positive):
    scores = left @ right.T / .4
    weights_left = positive.float() / positive.sum(-1, keepdim=True)
    weights_right = positive.T.float() / positive.T.sum(-1, keepdim=True)
    return (-(weights_left * F.log_softmax(scores, -1)).sum(-1).mean()
            -(weights_right * F.log_softmax(scores.T, -1)).sum(-1).mean()) / 2


class ComplementaryAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_workers(self, case):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(worker, args=(str(Path(directory) / 'group'), directory, case),
                                        nprocs=2, join=True)
            return [torch.load(Path(directory) / f'{rank}.pt', weights_only=True) for rank in range(2)]

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_complementary_views_match_full_reference(self):
        left = LEFT[:2].clone().requires_grad_()
        right = RIGHT[:2].clone().requires_grad_()
        expected = expected_loss(left, right, torch.eye(2, dtype=torch.bool))
        expected.backward()
        results = self.run_workers('complementary')
        for row in results:
            torch.testing.assert_close(row['loss'], expected.detach())
        torch.testing.assert_close(results[0]['grad'] / 2, left.grad)
        torch.testing.assert_close(results[1]['grad'] / 2, right.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_rectangular_multi_positive_and_padding(self):
        left, right = LEFT.clone().requires_grad_(), RIGHT[:2].clone().requires_grad_()
        positive = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)
        expected = expected_loss(left, right, positive)
        expected.backward()
        result = self.run_workers('rectangular')
        torch.testing.assert_close(result[0]['loss'], expected.detach())
        torch.testing.assert_close(result[0]['grad'] / 2, left.grad)
        torch.testing.assert_close(result[1]['grad'] / 2, right.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_class_positives_combine_with_identity_across_ranks(self):
        left, right = LEFT.clone().requires_grad_(), RIGHT[:2].clone().requires_grad_()
        positive = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        expected = expected_loss(left, right, positive)
        expected.backward()
        result = self.run_workers('class')
        torch.testing.assert_close(result[0]['loss'], expected.detach())
        torch.testing.assert_close(result[0]['grad'] / 2, left.grad)
        torch.testing.assert_close(result[1]['grad'] / 2, right.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_unmatched_and_unknown_views_do_not_become_negatives(self):
        left = LEFT[:2].clone().requires_grad_()
        right = RIGHT[:2].clone().requires_grad_()
        expected = expected_loss(left, right, torch.eye(2, dtype=torch.bool))
        expected.backward()
        result = self.run_workers('unmatched')
        torch.testing.assert_close(result[0]['loss'], expected.detach())
        torch.testing.assert_close(result[0]['grad'][:2] / 2, left.grad)
        torch.testing.assert_close(result[1]['grad'][:2] / 2, right.grad)
        self.assertEqual(result[0]['grad'][2].count_nonzero().item(), 0)
        self.assertEqual(result[1]['grad'][2].count_nonzero().item(), 0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_no_ids_retains_previous_zero_loss(self):
        for case in ('no_ids', 'disjoint', 'disabled'):
            for row in self.run_workers(case):
                self.assertEqual(row['loss'].item(), 0)
                self.assertEqual(row['grad'].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
