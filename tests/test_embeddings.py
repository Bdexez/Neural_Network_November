from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from chimera import Chimera, ChimeraConfig
from chimera.calibration import file_sha256
from chimera.cli import main
from chimera.data import synthetic_batch
from chimera.training import save_checkpoint


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(481)
        self.config = replace(ChimeraConfig.preset('tiny'), alignment_text_view='isolated')
        self.model = Chimera(self.config).eval()
        self.batch = {k: v for k, v in synthetic_batch(self.config, batch_size=3, seed=482).items()
                      if k in ('input_ids', 'attention_mask', 'images', 'series', 'series_mask')}

    def test_matches_training_projections_and_is_independent(self):
        projected = {}
        handles = [layer.register_forward_hook(lambda module, inputs, output, i=i: projected.update({i: output}))
                   for i, layer in enumerate(self.model.alignment)]
        try:
            with torch.inference_mode():
                self.model(**self.batch)
        finally:
            for handle in handles:
                handle.remove()
        with torch.inference_mode():
            encoded = self.model.encode_modalities(**self.batch)
            for name, index, keys in [('text', 2, ('input_ids', 'attention_mask')),
                                      ('image', 0, ('images',)), ('series', 1, ('series', 'series_mask'))]:
                torch.testing.assert_close(encoded[name], F.normalize(projected[index].float(), dim=-1))
                isolated = self.model.encode_modalities(**{k: self.batch[k] for k in keys})[name]
                torch.testing.assert_close(encoded[name], isolated, rtol=0, atol=0)
                torch.testing.assert_close(encoded[name].norm(dim=-1), torch.ones(3))

    def test_gradients_validation_and_nonfinite_values(self):
        self.model.train()
        vectors = self.model.encode_modalities(**self.batch)
        sum(v[:, 0].sum() for v in vectors.values()).backward()
        self.assertTrue(self.model.training)
        for name in ('embedding.weight', 'alignment.0.weight', 'alignment.1.weight', 'alignment.2.weight'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0, name)
        with self.assertRaises(ValueError):
            self.model.encode_modalities()
        with self.assertRaises(ValueError):
            self.model.encode_modalities(input_ids=self.batch['input_ids'], attention_mask=torch.zeros_like(self.batch['attention_mask']))
        with torch.no_grad():
            self.model.alignment[2].weight.zero_()
            self.assertEqual(self.model.encode_modalities(input_ids=self.batch['input_ids'])['text'].abs().sum().item(), 0)
            self.model.alignment[2].weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            self.model.encode_modalities(input_ids=self.batch['input_ids'])

    def test_bf16_and_padding(self):
        batch = dict(self.batch)
        batch['attention_mask'] = batch['attention_mask'].clone()
        batch['attention_mask'][0, -3:] = False
        batch['series_mask'] = batch['series_mask'].clone()
        batch['series_mask'][0, -3:] = False
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
            vectors = self.model.encode_modalities(**batch)
            altered = dict(batch, input_ids=batch['input_ids'].clone(), series=batch['series'].clone())
            altered['input_ids'][0, -3:] = 0
            altered['series'][0, -3:] = float('nan')
            changed = self.model.encode_modalities(**altered)
        for name in vectors:
            self.assertTrue(torch.isfinite(vectors[name]).all())
            torch.testing.assert_close(vectors[name], changed[name], rtol=0, atol=0)

    def test_cli_batches_metadata_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(directory)
            checkpoint, data, out = root / 'model.pt', root / 'data.jsonl', root / 'vectors.jsonl'
            save_checkpoint(checkpoint, self.model)
            records = [dict(text='signal ' + str(i) * (i + 1), series=[[float(i), float(j)] for j in range(4 + i)], class_label=0)
                       for i in range(4)]
            # Ajuster le nombre de variables au preset utilisé.
            for row in records:
                row['series'] = [[float(i)] * self.config.series_features for i in range(len(row['series']))]
            data.write_text(''.join(json.dumps(row) + '\n' for row in records))
            args = ['--threads', '2', 'embed', '--checkpoint', str(checkpoint), '--data', str(data)]
            main(args + ['--output', str(out), '--batch-size', '3'])
            batched = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([row['index'] for row in batched], list(range(4)))
            self.assertEqual(batched[0]['checkpoint_sha256'], file_sha256(checkpoint))
            self.assertEqual(batched[0]['dataset_sha256'], file_sha256(data))
            self.assertEqual(batched[0]['alignment_text_view'], 'isolated')
            self.assertEqual(set(batched[0]['vectors']), {'text', 'series'})
            main(args + ['--output', str(out), '--batch-size', '1'])
            single = [json.loads(line) for line in out.read_text().splitlines()]
            for left, right in zip(batched, single):
                for name in left['vectors']:
                    torch.testing.assert_close(torch.tensor(left['vectors'][name]), torch.tensor(right['vectors'][name]), rtol=1e-4, atol=1e-5)
            original = out.read_text()
            with patch('chimera.cli.file_sha256', side_effect=[file_sha256(checkpoint), 'changed']):
                with self.assertRaisesRegex(ValueError, 'checkpoint a changé'):
                    main(args + ['--output', str(out)])
            self.assertEqual(out.read_text(), original)
            hardlink = root / 'data-link'
            os.link(data, hardlink)
            with self.assertRaises(SystemExit):
                main(args + ['--output', str(hardlink)])
            self.assertEqual(data.read_text(), ''.join(json.dumps(row) + '\n' for row in records))


if __name__ == '__main__':
    unittest.main()
