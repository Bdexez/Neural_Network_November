from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chimera import Chimera, ChimeraConfig
from chimera.data import synthetic_batch
from chimera.training import load_checkpoint, save_checkpoint


class IsolatedTextAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        config = ChimeraConfig.preset('tiny')
        torch.manual_seed(1711)
        self.joint = Chimera(config).eval()
        self.isolated = Chimera(replace(config, alignment_text_view='isolated')).eval()
        self.isolated.load_state_dict(self.joint.state_dict(), strict=True)
        self.batch = synthetic_batch(config, batch_size=3, seed=1712)
        self.batch = {key:value for key,value in self.batch.items()
                      if key in ('input_ids','attention_mask','images','series','series_mask')}

    @staticmethod
    def projected(model, batch):
        values = []
        handle = model.alignment[2].register_forward_hook(lambda module, inputs, output: values.append(output))
        try:
            result = model(**batch)
        finally:
            handle.remove()
        return result, values[0]

    def test_text_alignment_is_independent_from_other_modalities(self):
        changed = dict(self.batch, images=self.batch['images'].roll(1,0),
                       series=self.batch['series'].roll(1,0), series_mask=self.batch['series_mask'].roll(1,0))
        with torch.inference_mode():
            original, local = self.projected(self.isolated, self.batch)
            _, altered = self.projected(self.isolated, changed)
            baseline, joint = self.projected(self.joint, self.batch)
            _, joint_altered = self.projected(self.joint, changed)
        torch.testing.assert_close(local,altered,rtol=0,atol=0)
        self.assertGreater((joint-joint_altered).abs().max().item(),1e-6)
        for name in ('logits','classification_logits','forecast'):
            torch.testing.assert_close(getattr(original,name),getattr(baseline,name),rtol=0,atol=0)

    def test_gradients_do_not_flow_from_text_view_into_other_inputs(self):
        for model in (self.joint,self.isolated):
            batch = dict(self.batch, images=self.batch['images'].clone().requires_grad_(),
                         series=self.batch['series'].clone().requires_grad_())
            _, text_view = self.projected(model,batch)
            gradients = torch.autograd.grad(text_view.square().sum(), (batch['images'],batch['series']),
                                            allow_unused=True)
            if model is self.isolated:
                self.assertEqual(gradients,(None,None))
            else:
                self.assertTrue(all(value is not None and value.abs().sum().item() > 0 for value in gradients))

    def test_isolated_representation_matches_text_only_forward_with_padding(self):
        tokens = self.batch['input_ids'].clone()
        mask = self.batch['attention_mask'].clone().bool()
        mask[0,:2] = False
        mask[1,:1] = False
        values = []
        handle = self.isolated.norm.register_forward_hook(lambda module, inputs, output: values.append(output.clone()))
        try:
            with torch.inference_mode():
                self.isolated(tokens,attention_mask=mask)
        finally:
            handle.remove()
        self.assertEqual(len(values),1)
        last = torch.arange(mask.size(1))[None,:].expand_as(mask).masked_fill(~mask,-1).max(-1).values
        expected = values[0][torch.arange(tokens.size(0)),last]
        with torch.inference_mode():
            actual = self.isolated._isolated_text_representation(tokens,mask)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_no_extra_pass_when_local_alignment_inactive(self):
        with patch.object(self.isolated,'_isolated_text_representation',side_effect=AssertionError('extra pass')):
            one = {key:value[:1] for key,value in self.batch.items()}
            self.isolated(**one)
            no_text = {key:value for key,value in self.batch.items() if key not in ('input_ids','attention_mask')}
            self.isolated(**no_text)
            text_only = {key:value for key,value in self.batch.items() if key in ('input_ids','attention_mask')}
            self.isolated(**text_only)

    def test_training_dropout_primary_outputs_checkpointing_and_cache(self):
        config = replace(self.joint.config, dropout=.1, gradient_checkpointing=True)
        torch.manual_seed(1720)
        joint = Chimera(config).train()
        isolated = Chimera(replace(config, alignment_text_view='isolated')).train()
        isolated.load_state_dict(joint.state_dict())
        torch.manual_seed(1721)
        left = joint(**self.batch)
        torch.manual_seed(1721)
        right = isolated(**self.batch)
        for name in ('logits','classification_logits','forecast'):
            torch.testing.assert_close(getattr(left,name),getattr(right,name),rtol=0,atol=0)
        right.losses['alignment'].backward()
        for name in ('alignment.2.weight','embedding.weight','blocks.0.attn.q.weight'):
            grad = dict(isolated.named_parameters())[name].grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)
        isolated.eval()
        tokens = self.batch['input_ids'][:2, :8]
        with torch.inference_mode():
            whole = isolated(tokens,images=self.batch['images'][:2],series=self.batch['series'][:2]).logits
            prefix = isolated(tokens[:,:4],images=self.batch['images'][:2],series=self.batch['series'][:2],use_cache=True)
            suffix = isolated(tokens[:,4:],cache=prefix.cache,use_cache=True)
        torch.testing.assert_close(suffix.logits,whole[:,4:],rtol=2e-5,atol=2e-6)

    def test_isolated_view_under_bf16_with_stable_norm(self):
        config = replace(self.joint.config, alignment_text_view='isolated', stable_rms_norm=True)
        model = Chimera(config).eval()
        model.load_state_dict(self.joint.state_dict())
        changed = dict(self.batch, images=self.batch['images'].roll(1,0),
                       series=self.batch['series'].roll(1,0), series_mask=self.batch['series_mask'].roll(1,0))
        with torch.inference_mode(), torch.autocast('cpu',dtype=torch.bfloat16):
            output, left = self.projected(model,self.batch)
            _, right = self.projected(model,changed)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(output.losses['alignment']))
        self.assertTrue(torch.isfinite(output.forecast).all())

    def test_checkpoint_compatibility_and_validation(self):
        config = self.joint.config
        old = config.to_dict()
        del old['alignment_text_view']
        self.assertEqual(ChimeraConfig.from_dict(old).alignment_text_view,'joint')
        with self.assertRaises(ValueError):
            replace(config,alignment_text_view='unknown')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model.pt'
            save_checkpoint(checkpoint,self.joint)
            model, _ = load_checkpoint(checkpoint,overrides={'alignment_text_view':'isolated'})
            self.assertEqual(model.config.alignment_text_view,'isolated')
            self.assertEqual(set(model.state_dict()),set(self.joint.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value,self.joint.state_dict()[name],rtol=0,atol=0,msg=name)


if __name__ == '__main__':
    unittest.main()
