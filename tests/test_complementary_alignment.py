from pathlib import Path
import tempfile
import unittest

import torch
from torch import distributed as dist
from torch.nn import functional as F

from chimera.distributed_alignment import _rectangular_alignment, distributed_alignment_loss


LEFT = torch.tensor([[.8, .1, .3], [.2, .9, .4], [.1, .3, .9]], dtype=torch.float32)
RIGHT = torch.tensor([[.7, .2, .4], [.3, .8, .1], [.2, .2, .8]], dtype=torch.float32)


def worker(rank, group_file, output, case):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{group_file}', rank=rank, world_size=2)
    try:
        if case in ('rectangular', 'class', 'cross_class', 'cross_class_unknown', 'cross_class_no_match', 'cross_class_disabled'):
            values = LEFT.clone() if rank == 0 else RIGHT[:2].clone()
            ids = torch.tensor([7, 7, 9] if rank == 0 else [7, 9])
            if case.startswith('cross_class'):
                ids = None
        elif case == 'unmatched':
            values = LEFT.clone() if rank == 0 else RIGHT.clone()
            ids = torch.tensor([7, 9, -1] if rank == 0 else [7, 9, 11])
        else:
            values = LEFT[:2].clone() if rank == 0 else RIGHT[:2].clone()
            ids = (None if case == 'no_ids' else torch.tensor([11, 12] if case == 'disjoint' and rank == 1 else [7, 9]))
        values.requires_grad_()
        vectors = {0: values} if rank == 0 else {1: values}
        if case == 'class':
            labels = torch.tensor([0, 1, 1] if rank == 0 else [0, 1])
        elif case.startswith('cross_class'):
            labels = torch.tensor([0, 0, 1] if rank == 0 else [0, 1])
            if case == 'cross_class_unknown' and rank == 0:
                labels[1] = -100
            if case == 'cross_class_no_match' and rank == 1:
                labels += 2
        else:
            labels = None
        loss = distributed_alignment_loss(vectors, labels, .4, 'class' if labels is not None else 'paired',
                                          group=dist.group.WORLD, sample_ids=ids,
                                          cross_rank_views=case != 'disabled',
                                          cross_class_views=case.startswith('cross_class') and case != 'cross_class_disabled')
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

    def test_class_and_identity_policies_control_eligibility_separately(self):
        gathered = torch.zeros(6, 3, 3)
        gathered[:3, 0], gathered[3:, 1] = LEFT, RIGHT
        metadata = torch.tensor([[1, 0, 0, 0, 7], [1, 0, 0, 1, -1], [1, 0, 0, 3, 11],
                                 [0, 1, 0, 1, 7], [0, 1, 0, 0, -1], [0, 1, 0, 2, 11]])
        for cross_ids in (False, True):
            actual = _rectangular_alignment(gathered, metadata, 0, 1, .4, 'class', cross_ids=cross_ids,
                                             cross_classes=True, has_ids=True)
            if cross_ids:
                expected = expected_loss(LEFT, RIGHT, torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.bool))
            else:
                expected = expected_loss(LEFT[:2], RIGHT[:2], torch.tensor([[1, 1], [1, 0]], dtype=torch.bool))
            torch.testing.assert_close(actual, expected)

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

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_unpaired_classes_match_rectangular_reference_and_gradients(self):
        for case in ('cross_class', 'cross_class_unknown'):
            with self.subTest(case=case):
                left, right = LEFT.clone().requires_grad_(), RIGHT[:2].clone().requires_grad_()
                if case == 'cross_class':
                    selected = left
                    positive = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)
                else:
                    selected = left[[0, 2]]
                    positive = torch.eye(2, dtype=torch.bool)
                expected = expected_loss(selected, right, positive)
                expected.backward()
                result = self.run_workers(case)
                for row in result:
                    torch.testing.assert_close(row['loss'], expected.detach())
                torch.testing.assert_close(result[0]['grad'] / 2, left.grad)
                torch.testing.assert_close(result[1]['grad'] / 2, right.grad)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_unpaired_classes_need_a_known_matching_class_and_opt_in(self):
        for case in ('cross_class_disabled', 'cross_class_no_match'):
            for row in self.run_workers(case):
                self.assertEqual(row['loss'].item(), 0)
                self.assertEqual(row['grad'].count_nonzero().item(), 0)


if __name__ == '__main__':
    unittest.main()
