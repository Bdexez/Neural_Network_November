from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.modules import SparseMoE
from chimera.training import load_checkpoint, save_checkpoint


class Negate(nn.Module):
    def forward(self, value):
        return -value


class Zero(nn.Module):
    def forward(self, value):
        return torch.zeros_like(value)


class Top1RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_task_loss_teaches_router_to_choose_the_useful_expert(self):
        results = {}
        for mode in ("normalized", "probability"):
            config = replace(ChimeraConfig.preset("tiny"), n_experts=2, top_k=1, top1_gate=mode)
            block = SparseMoE(config)
            block.experts = nn.ModuleList([nn.Identity(), Negate()])
            block.shared = Zero()
            with torch.no_grad():
                block.router.weight.zero_()
                block.router.weight[0, 0], block.router.weight[1, 0] = -1, 1
            inputs = torch.zeros(2, 1, config.d_model)
            inputs[0, 0, 0], inputs[1, 0, 0] = 1, -1
            valid = torch.ones(2, 1, dtype=torch.bool)
            optimizer = torch.optim.SGD(block.router.parameters(), lr=1.0)

            def task_loss():
                return (block(inputs, valid)[0][:, :, 0] - 1).square().mean()

            before = task_loss().item()
            for _ in range(80):
                optimizer.zero_grad()
                loss = task_loss()
                loss.backward()
                optimizer.step()
            results[mode] = before, task_loss().item()
        self.assertEqual(results["normalized"], (4.0, 4.0))
        self.assertLess(results["probability"][1], results["probability"][0] / 100)

    def test_probability_mode_transmits_task_gradients_through_the_model(self):
        gradients = {}
        for mode in ("normalized", "probability"):
            torch.manual_seed(99)
            config = replace(ChimeraConfig.preset("tiny"), n_experts=2, top_k=1, top1_gate=mode,
                             balance_weight=0, router_z_weight=0)
            model = Chimera(config)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(series=torch.randn(4, 8, 3), class_labels=torch.tensor([0, 1, 2, 3]))
            output.loss.backward()
            gradients[mode] = [block.moe.router.weight.grad.abs().max().item() for block in model.blocks]
        self.assertLess(max(gradients["normalized"]), 1e-8)
        self.assertGreater(min(gradients["probability"]), 1e-6)

    def test_checkpoint_compatibility_and_configuration_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(ChimeraConfig.preset("tiny"), n_experts=2, top_k=1)
            model = Chimera(config).eval()
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, model)
            payload = torch.load(path, weights_only=True)
            del payload["config"]["top1_gate"]
            torch.save(payload, path)
            original, _ = load_checkpoint(path)
            switched, _ = load_checkpoint(path, overrides={"top1_gate": "probability"})
            self.assertEqual(original.config.top1_gate, "normalized")
            self.assertEqual(switched.config.top1_gate, "probability")
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, switched.state_dict()[name], rtol=0, atol=0)
            batch = synthetic_batch(config, batch_size=2, seed=207)
            with torch.inference_mode():
                before, after = model(**batch), original(**batch)
            torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
            torch.testing.assert_close(before.forecast, after.forecast, rtol=0, atol=0)
            destination = Path(directory) / "probability.pt"
            save_checkpoint(destination, switched)
            restored, _ = load_checkpoint(destination)
            self.assertEqual(restored.config.top1_gate, "probability")
            with torch.inference_mode():
                changed, reloaded = switched(**batch), restored(**batch)
            self.assertGreater((before.logits - changed.logits).abs().max().item(), 1e-7)
            torch.testing.assert_close(changed.logits, reloaded.logits, rtol=0, atol=0)
            for changes in (dict(top1_gate="invalid"), dict(top1_gate="probability", top_k=2),
                            dict(top1_gate="probability", n_experts=1)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replace(config, **changes)


if __name__ == "__main__":
    unittest.main()
