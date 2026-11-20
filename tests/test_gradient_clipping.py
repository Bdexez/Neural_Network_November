import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from chimera.gradient_clipping import clip_grad_norm
from chimera.training import step_optimizer


def distributed_worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.zero_()
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        wrapped(torch.full((1, 2), (1 + 2 * rank) * 1e20)).sum().backward()
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.)
        torch.save({'weight': model.weight.detach(), 'norm': norm, 'skipped': skipped},
                   Path(output_dir) / f'{rank}.pt')
    finally:
        dist.destroy_process_group()


class GradientClippingTests(unittest.TestCase):
    def test_normal_path_matches_pytorch(self):
        first = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        second = torch.nn.Parameter(first.detach().clone())
        first.grad = torch.tensor([2., -3., 4.])
        second.grad = first.grad.clone()
        expected = torch.nn.utils.clip_grad_norm_([first], 0.7, error_if_nonfinite=True)
        with patch('chimera.gradient_clipping.math.hypot', side_effect=AssertionError('repli inattendu')):
            actual = clip_grad_norm([second], 0.7)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_overflowing_float32_norm_clips_and_steps(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.zero_()
        model.weight.grad = torch.full_like(model.weight, 1e20)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.)
        self.assertFalse(skipped)
        self.assertAlmostEqual(norm / 1e20, math.sqrt(2), places=6)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, -.1 / math.sqrt(2)), rtol=1e-6, atol=0)

    def test_finite_overflow_with_scaler_is_not_silently_discarded(self):
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cpu', init_scale=16)
        scaler.scale(model(torch.full((1, 2), 1e20)).sum()).backward()
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.)
        self.assertFalse(skipped)
        self.assertAlmostEqual(norm / 1e20, math.sqrt(2), places=6)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, -.1 / math.sqrt(2)), rtol=1e-6, atol=0)
        self.assertEqual(scaler.get_scale(), 16)

    def test_multiple_parameter_groups_share_one_global_norm(self):
        first = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        second = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        first.grad = torch.tensor([3e20, 4e20], dtype=torch.float32)
        second.grad = torch.tensor([12e20], dtype=torch.float64)
        norm = clip_grad_norm([first, second], 1.)
        self.assertAlmostEqual(norm.item() / 1e20, 13., places=6)
        torch.testing.assert_close(first.grad, torch.tensor([3 / 13, 4 / 13]), rtol=1e-6, atol=0)
        torch.testing.assert_close(second.grad, torch.tensor([12 / 13], dtype=torch.float64), rtol=1e-6, atol=0)

    def test_fallback_handles_double_overflow_and_tiny_clip(self):
        parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        parameter.grad = torch.tensor([1.7e308, -1.7e308], dtype=torch.float64)
        norm = clip_grad_norm([parameter], 1.)
        self.assertTrue(torch.isinf(norm))
        torch.testing.assert_close(parameter.grad, torch.tensor([1 / math.sqrt(2), -1 / math.sqrt(2)], dtype=torch.float64))
        small = torch.nn.Parameter(torch.zeros(2))
        small.grad = torch.tensor([3e38, -3e38])
        clip_grad_norm([small], 1e-30)
        torch.testing.assert_close(small.grad, torch.tensor([1e-30 / math.sqrt(2), -1e-30 / math.sqrt(2)]), rtol=1e-6, atol=0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_distributed_gradients_clip_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(distributed_worker, args=(str(Path(directory) / 'group'), directory),
                                        nprocs=2, join=True)
            for rank in range(2):
                result = torch.load(Path(directory) / f'{rank}.pt', weights_only=True)
                self.assertFalse(result['skipped'])
                self.assertAlmostEqual(result['norm'] / 1e20, 2 * math.sqrt(2), places=6)
                torch.testing.assert_close(result['weight'], torch.full((1, 2), -.1 / math.sqrt(2)), rtol=1e-6, atol=0)

    def test_nonfinite_gradients_still_fail_or_skip(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        for invalid in (float('inf'), float('nan')):
            parameter.grad = torch.tensor([1., invalid])
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, 'non-finite'):
                clip_grad_norm([parameter], 1.)
            result = clip_grad_norm([parameter], 1., allow_nonfinite=True)
            self.assertFalse(torch.isfinite(result))
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cpu', init_scale=16)
        before = model.weight.detach().clone()
        scaler.scale(model(torch.ones(1, 1)).sum() * float('inf')).backward()
        norm, skipped = step_optimizer(model, optimizer, scaler, 1.)
        self.assertTrue(skipped)
        self.assertIsNone(norm)
        self.assertEqual(scaler.get_scale(), 8)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
