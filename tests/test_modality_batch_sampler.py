from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from chimera import ChimeraConfig
from chimera.data import JointDataset, collate_records
from chimera.modality_batch_sampler import ModalityBatchSampler


class ModalityBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'mixed.jsonl'
        records = ([{'text': f'texte {i}', 'class_label': 0} for i in range(5)] +
                   [{'series': [[float(i)] * 3], 'class_label': 1} for i in range(3)] +
                   [{'text': 'associé', 'series': [[1.] * 3], 'class_label': 2}])
        self.path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        self.config = ChimeraConfig.preset('tiny')
        self.dataset = JointDataset(self.path, self.config, allow_mixed=True)

    def test_default_still_rejects_mixed_and_opt_in_indexes_signatures(self):
        with self.assertRaisesRegex(ValueError, 'modalités'):
            JointDataset(self.path, self.config)
        self.assertEqual(list(self.dataset.signatures), [1] * 5 + [4] * 3 + [5])
        self.assertEqual(self.dataset.fingerprint, __import__('hashlib').sha256(self.path.read_bytes()).hexdigest())
        self.assertIn('series', self.dataset[5])

    def test_optional_source_id_is_stable_across_copies_and_negative_indices(self):
        dataset = JointDataset(self.path, self.config, allow_mixed=True, include_alignment_ids=True)
        self.assertNotIn('alignment_ids', self.dataset[0])
        batch = collate_records([dataset[2], dataset[0], dataset[2], dataset[-9]])
        self.assertEqual(batch['alignment_ids'].tolist(), [2, 0, 2, 0])
        with self.assertRaisesRegex(ValueError, 'alignment_ids'):
            collate_records([dataset[0], self.dataset[0]])

    def test_every_example_once_and_collation_is_homogeneous(self):
        sampler = ModalityBatchSampler(self.dataset, 2, seed=42)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(9)))
        for batch in batches:
            self.assertEqual(len({self.dataset.signatures[i] for i in batch}), 1)
            self.assertLessEqual(len(batch), 2)
        for batch in DataLoader(self.dataset, batch_sampler=sampler, collate_fn=collate_records):
            self.assertTrue('input_ids' in batch or 'series' in batch)

    def test_distributed_batches_have_equal_steps_and_only_padding_repeats(self):
        samplers = [ModalityBatchSampler(self.dataset, 2, num_replicas=4, rank=rank, seed=10)
                    for rank in range(4)]
        rows = [list(sampler) for sampler in samplers]
        self.assertEqual({len(row) for row in rows}, {2})
        interleaved = [rows[rank][step] for step in range(2) for rank in range(4)]
        flat = [i for batch in interleaved[:6] for i in batch]
        self.assertEqual(Counter(flat), Counter(range(9)))
        self.assertEqual(interleaved[6:], interleaved[:2])
        for batch in interleaved:
            self.assertEqual(len({self.dataset.signatures[i] for i in batch}), 1)

    def test_shared_batches_match_across_ranks_without_extra_padding(self):
        samplers = [ModalityBatchSampler(self.dataset, 2, num_replicas=4, rank=rank, seed=9,
                                         shared_across_ranks=True) for rank in range(4)]
        for epoch in (0, 2):
            for sampler in samplers:
                sampler.set_epoch(epoch)
            batches = [list(sampler) for sampler in samplers]
            self.assertTrue(all(row == batches[0] for row in batches))
            self.assertEqual(len(batches[0]), 6)
            self.assertEqual(Counter(index for batch in batches[0] for index in batch), Counter(range(9)))
            self.assertEqual(len(samplers[0]), 6)
        with self.assertRaises(ValueError):
            ModalityBatchSampler(self.dataset, 2, shared_across_ranks=1)

    def test_deterministic_epoch_and_rng_independence(self):
        sampler = ModalityBatchSampler(self.dataset, 2, seed=123)
        before = torch.get_rng_state().clone()
        original = list(sampler)
        self.assertEqual(original, list(sampler))
        sampler.set_epoch(2)
        other = list(sampler)
        self.assertNotEqual(original, other)
        copy = ModalityBatchSampler(self.dataset, 2, seed=123)
        copy.set_epoch(2)
        self.assertEqual(other, list(copy))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            sampler.set_epoch(-1)


if __name__ == '__main__':
    unittest.main()
