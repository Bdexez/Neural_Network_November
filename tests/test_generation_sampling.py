from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from chimera import Chimera, ChimeraConfig


class GenerationSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_extreme_finite_logits_and_small_temperatures_remain_sampleable(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        scores = torch.full((1, 1, model.config.vocab_size), -3e38)
        scores[..., 4], scores[..., 5] = 3e38, 2e38
        fake_output = SimpleNamespace(logits=scores, cache=object())
        with patch.object(model, "forward", return_value=fake_output):
            for temperature in (0.01, 1e-40, 5e-324):
                with self.subTest(temperature=temperature):
                    generated = model.generate(torch.tensor([[1]]), max_new_tokens=1,
                                               temperature=temperature, top_k=0, eos_id=None)
                    self.assertEqual(generated[0, -1].item(), 4)
                    self.assertTrue(model.training)
            generated = model.generate(torch.tensor([[1]]), max_new_tokens=1,
                                       temperature=1e308, top_k=1, eos_id=None)
            self.assertEqual(generated[0, -1].item(), 4)

    def test_tied_logits_remain_sampleable_at_subnormal_temperature(self):
        model = Chimera(ChimeraConfig.preset("tiny"))
        scores = torch.full((1, 1, model.config.vocab_size), -1.0)
        scores[..., 4], scores[..., 5] = 0, 0
        fake_output = SimpleNamespace(logits=scores, cache=object())
        with patch.object(model, "forward", return_value=fake_output):
            outputs = [model.generate(torch.tensor([[1]]), max_new_tokens=1, temperature=5e-324,
                                      top_k=2, top_p=1, eos_id=None)[0, -1].item() for _ in range(20)]
        self.assertTrue(set(outputs).issubset({4, 5}))
        self.assertEqual(set(outputs), {4, 5})

    def test_nonfinite_maximum_is_reported_and_mode_restored(self):
        model = Chimera(ChimeraConfig.preset("tiny")).train()
        scores = torch.zeros(1, 1, model.config.vocab_size)
        scores[..., 4] = float("inf")
        with patch.object(model, "forward", return_value=SimpleNamespace(logits=scores, cache=object())):
            with self.assertRaisesRegex(FloatingPointError, "maximum fini"):
                model.generate(torch.tensor([[1]]), max_new_tokens=1, temperature=0.5)
        self.assertTrue(model.training)

    def test_normal_temperature_preserves_sampling_distribution(self):
        model = Chimera(ChimeraConfig.preset("tiny"))
        scores = torch.full((1, 1, model.config.vocab_size), -3.0)
        scores[..., 4], scores[..., 5], scores[..., 6] = 1.5, -0.25, 0.9
        expected_scores = scores[:, -1].clone()
        expected_scores[:, [0, 1, 3]] = -torch.inf
        expected = (expected_scores / 0.7).softmax(-1)
        probabilities = []

        def choose(probability, count):
            probabilities.append(probability.clone())
            return probability.argmax(-1, keepdim=True)

        with patch.object(model, "forward", return_value=SimpleNamespace(logits=scores, cache=object())):
            with patch("chimera.model.torch.multinomial", side_effect=choose):
                model.generate(torch.tensor([[1]]), max_new_tokens=1, temperature=0.7,
                               top_k=0, top_p=1, eos_id=None)
        torch.testing.assert_close(probabilities[0], expected, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
