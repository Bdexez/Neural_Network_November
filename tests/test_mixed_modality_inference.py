from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from chimera import Chimera
from chimera.cli import main
from chimera.data import JointDataset, collate_records
from chimera.evaluation import evaluate
from chimera.modality_batch_sampler import OrderedModalityBatchSampler
from chimera.training import save_checkpoint
from tests.test_mixed_modality_training import make_data


class MixedModalityInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path, self.data = make_data(self.root)
        from chimera import ChimeraConfig
        self.config = ChimeraConfig.load(self.config_path)
        torch.manual_seed(123)
        self.model = Chimera(self.config).eval()
        self.checkpoint = self.root / 'model.pt'
        save_checkpoint(self.checkpoint, self.model)
        self.dataset = JointDataset(self.data, self.config, allow_mixed=True)

    def test_ordered_sampler_never_reorders_or_mixes(self):
        signatures = self.dataset.signatures
        sampler = OrderedModalityBatchSampler(self.dataset, 3)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual([index for batch in batches for index in batch], list(range(len(self.dataset))))
        for batch in batches:
            self.assertEqual(len({signatures[index] for index in batch}), 1)
            self.assertLessEqual(len(batch), 3)

    def test_predict_and_embed_preserve_order_and_individual_vectors(self):
        output = self.root / 'predictions.jsonl'
        vectors = self.root / 'vectors.jsonl'
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'predict', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--output', str(output)])
            main(['--threads', '2', 'embed', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--batch-size', '3', '--output', str(vectors)])
        predicted = [json.loads(line) for line in output.read_text().splitlines()]
        embedded = [json.loads(line) for line in vectors.read_text().splitlines()]
        self.assertEqual([row['index'] for row in predicted], list(range(len(self.dataset))))
        self.assertEqual([row['index'] for row in embedded], list(range(len(self.dataset))))
        with torch.inference_mode():
            for index, row in enumerate(embedded):
                batch = {key: value for key, value in collate_records([self.dataset[index]]).items()
                         if key in ('input_ids', 'attention_mask', 'images', 'series', 'series_mask')}
                expected = self.model.encode_modalities(**batch)
                self.assertEqual(set(row['vectors']), set(expected))
                for name, value in expected.items():
                    torch.testing.assert_close(torch.tensor(row['vectors'][name]), value[0], rtol=1e-4, atol=1e-5)
                direct = self.model(**batch).classification_logits.argmax(-1).item()
                self.assertEqual(predicted[index]['class_id'], direct)

    def test_evaluate_matches_single_example_batches(self):
        output = self.root / 'quality.json'
        with redirect_stdout(io.StringIO()):
            main(['--threads', '2', 'evaluate', '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                  '--batch-size', '3', '--output', str(output)])
        report = json.loads(output.read_text())
        expected = evaluate(self.model, DataLoader(self.dataset, batch_size=1, collate_fn=collate_records))
        self.assertEqual(report['classified_examples'], len(self.dataset))
        self.assertAlmostEqual(report['accuracy'], expected['accuracy'])
        self.assertAlmostEqual(report['classification'], expected['classification'], places=5)

    def test_late_invalid_record_preserves_outputs(self):
        with self.data.open('a') as stream:
            stream.write(json.dumps({'text': 'bad', 'class_label': 999}) + '\n')
        for command in ('predict', 'embed', 'evaluate'):
            output = self.root / f'{command}.json'
            output.write_text('précédent')
            with redirect_stdout(io.StringIO()), self.subTest(command=command), self.assertRaises(ValueError):
                main(['--threads', '2', command, '--checkpoint', str(self.checkpoint), '--data', str(self.data),
                      '--output', str(output)])
            self.assertEqual(output.read_text(), 'précédent')


if __name__ == '__main__':
    unittest.main()
