from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import os
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.cli import main
from chimera.calibration import file_sha256
from chimera.modules import RMSNorm
from chimera.training import load_checkpoint, save_checkpoint


class StableRMSNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_large_finite_values_and_gradients_against_fp64(self):
        layer = RMSNorm(2, stable=True)
        x = torch.tensor([[1e20, 2e20], [-3e30, 4e30]], requires_grad=True)
        value = layer(x)
        reference_x = x.detach().double().requires_grad_()
        reference = reference_x * torch.rsqrt(reference_x.square().mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(value.double(), reference, rtol=1e-6, atol=1e-7)
        grad = torch.autograd.grad(value.sum(), x)[0]
        expected = torch.autograd.grad(reference.sum(), reference_x)[0]
        torch.testing.assert_close(grad.double(), expected, rtol=2e-6, atol=0.)
        legacy = RMSNorm(2)(x.detach())
        self.assertTrue(torch.equal(legacy, torch.zeros_like(legacy)))
        self.assertTrue(torch.isfinite(grad).all())
        bf16 = torch.tensor([[1e20, 2e20]], dtype=torch.bfloat16, requires_grad=True)
        out = layer(bf16)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 1.)
        out.sum().backward()
        self.assertTrue(torch.isfinite(bf16.grad).all())

    def test_fp64_extreme_values_and_epsilon_outside_fp32(self):
        layer = RMSNorm(2, stable=True).double()
        x = torch.tensor([[1e300, 2e300]], dtype=torch.float64, requires_grad=True)
        expected = torch.tensor([[2**.5/(5**.5), 2*2**.5/(5**.5)]], dtype=torch.float64)
        torch.testing.assert_close(layer(x), expected, rtol=1e-14, atol=0)
        grad = torch.autograd.grad(layer(x).sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().max().item(),0)
        torch.testing.assert_close(grad * 1e300, torch.tensor([[.2529822128, -.1264911064]], dtype=torch.float64),
                                   rtol=1e-9, atol=0)
        self.assertTrue(torch.autograd.gradcheck(layer, (torch.tensor([[1., 2.]], dtype=torch.float64,
                                                                      requires_grad=True),)))
        for eps in (1e-100, 1e100):
            small = torch.tensor([[1e30, 0.]], requires_grad=True)
            output = RMSNorm(2, eps=eps, stable=True)(small)
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreater(output[0,0].item(),0)
        zero = RMSNorm(2, eps=1e-100, stable=True)(torch.zeros(1,2))
        self.assertTrue(torch.isfinite(zero).all())
        self.assertEqual(zero.sum().item(),0.)

    def test_normal_outputs_gradients_and_mixed_precision(self):
        generator = torch.Generator().manual_seed(900)
        x = (torch.randn(3, 5, 16, generator=generator) * 4).requires_grad_()
        old, stable = RMSNorm(16), RMSNorm(16, stable=True)
        old_out, new_out = old(x), stable(x)
        historical_formula = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + old.eps)
        torch.testing.assert_close(old_out, historical_formula.to(x.dtype) * old.weight, rtol=0, atol=0)
        torch.testing.assert_close(new_out, old_out, rtol=1e-6, atol=1e-6)
        direction = torch.randn(x.shape, generator=generator)
        a = torch.autograd.grad((old_out * direction).sum(), x, retain_graph=True)[0]
        b = torch.autograd.grad((new_out * direction).sum(), x)[0]
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
        for dtype in (torch.bfloat16, torch.float16):
            data = (x.detach().to(dtype) * 2).requires_grad_()
            with torch.autocast('cpu', dtype=dtype):
                result = stable(data)
            self.assertTrue(torch.isfinite(result).all())
            result.sum().backward()
            self.assertTrue(torch.isfinite(data.grad).all())

    def test_model_compatibility_cache_and_checkpoint_override(self):
        config = ChimeraConfig.preset('tiny')
        torch.manual_seed(432)
        legacy = Chimera(config).eval()
        stable = Chimera(replace(config, stable_rms_norm=True)).eval()
        stable.load_state_dict(legacy.state_dict(), strict=True)
        self.assertTrue(all(m.stable for m in stable.modules() if isinstance(m,RMSNorm)))
        self.assertTrue(all(not m.stable for m in legacy.modules() if isinstance(m,RMSNorm)))
        batch = synthetic_batch(config, batch_size=2, seed=901)
        with torch.inference_mode():
            before, after = legacy(**batch), stable(**batch)
            for name in ('classification_logits','forecast','logits'):
                torch.testing.assert_close(getattr(before,name),getattr(after,name),rtol=2e-5,atol=2e-6)
            tokens = batch['input_ids'][:1, :8]
            full = stable(tokens).logits
            prefix = stable(tokens[:, :4],use_cache=True)
            suffix = stable(tokens[:, 4:],cache=prefix.cache,use_cache=True)
            torch.testing.assert_close(suffix.logits,full[:,4:],rtol=2e-5,atol=2e-6)
        for model in (legacy, stable):
            model.train()
            model.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
        for name in ('embedding.weight', 'temporal.proj.weight', 'classifier.weight'):
            old_grad = dict(legacy.named_parameters())[name].grad
            new_grad = dict(stable.named_parameters())[name].grad
            torch.testing.assert_close(old_grad, new_grad, rtol=3e-4, atol=2e-6, msg=name)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.pt'
            save_checkpoint(checkpoint, legacy)
            old, _ = load_checkpoint(checkpoint)
            upgraded, _ = load_checkpoint(checkpoint,overrides={'stable_rms_norm':True})
            self.assertFalse(old.config.stable_rms_norm)
            self.assertTrue(upgraded.config.stable_rms_norm)
            self.assertEqual(set(old.state_dict()),set(upgraded.state_dict()))
            self.assertTrue(all(m.stable for m in upgraded.modules() if isinstance(m,RMSNorm)))
        serialized = config.to_dict()
        del serialized['stable_rms_norm']
        self.assertFalse(ChimeraConfig.from_dict(serialized).stable_rms_norm)
        with self.assertRaises(ValueError):
            replace(config, stable_rms_norm=1)


    def test_stable_norm_with_gradient_checkpointing(self):
        config = replace(ChimeraConfig.preset('tiny'), stable_rms_norm=True, gradient_checkpointing=True)
        model = Chimera(config).train()
        batch = synthetic_batch(config, batch_size=2, seed=975)
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_export_stable_checkpoint_preserves_source_and_weights(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            source, target = root / 'source.pt', root / 'stable.pt'
            save_checkpoint(source, Chimera(ChimeraConfig.preset('tiny')), step=7)
            original_sha = file_sha256(source)
            main(['export','--checkpoint',str(source),'--output',str(target),'--stable-rms-norm'])
            model, payload = load_checkpoint(target)
            source_model, _ = load_checkpoint(source)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, source_model.state_dict()[name], rtol=0, atol=0, msg=name)
            self.assertTrue(model.config.stable_rms_norm)
            self.assertEqual(payload['step'],7)
            self.assertTrue(payload['trainer']['inference_only'])
            self.assertEqual(payload['trainer']['stable_rms_norm_parent_sha256'],original_sha)
            self.assertEqual(file_sha256(source),original_sha)
            with self.assertRaises(SystemExit):
                main(['export','--checkpoint',str(source),'--output',str(source),'--stable-rms-norm'])
            self.assertEqual(file_sha256(source),original_sha)
            alias = root / 'alias.pt'
            os.link(source, alias)
            with self.assertRaises(SystemExit):
                main(['export','--checkpoint',str(source),'--output',str(alias),'--stable-rms-norm'])
            self.assertEqual(file_sha256(source),original_sha)

    def test_export_rejects_source_changed_during_load(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            source, target = root / 'source.pt', root / 'stable.pt'
            save_checkpoint(source, Chimera(ChimeraConfig.preset('tiny')))
            target.write_text('existing')
            def changed(*args, **kwargs):
                result = load_checkpoint(*args, **kwargs)
                with source.open('ab') as stream:
                    stream.write(b'changed')
                return result
            with patch('chimera.cli.load_checkpoint', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'a changé'):
                    main(['export','--checkpoint',str(source),'--output',str(target),'--stable-rms-norm'])
            self.assertEqual(target.read_text(),'existing')


if __name__ == '__main__':
    unittest.main()
