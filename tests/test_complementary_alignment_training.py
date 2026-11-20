from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import distributed as dist

from chimera import ChimeraConfig
from chimera.modalities import balanced_modalities
from chimera.training import load_checkpoint


class ComplementaryAlignmentTrainingTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_unpaired_records_align_by_class_without_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(ChimeraConfig.preset('tiny'), alignment_scope='global', alignment_mode='class',
                             alignment_text_view='isolated', dropout=0.)
            config_path, data = root / 'config.json', root / 'data.jsonl'
            config.save(config_path)
            rows = ([dict(text=f'signal {index}', class_label=index) for index in range(2)]
                    + [dict(series=[[float(index), 1., 2.], [2., 1., float(index)]], class_label=index)
                       for index in range(2)])
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                    '-m', 'chimera', '--threads', '1', 'train', '--config', str(config_path), '--data', str(data),
                    '--steps', '1', '--batch-size', '2', '--mixed-modalities', '--no-shuffle',
                    '--no-alignment-duplicate-positives', '--modality-dropout', '0',
                    '--loss-normalization', 'observed', '--seed', '0', '--log-every', '1']
            env = dict(os.environ)
            for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name, None)
            for name, extra in (('enabled', ['--alignment-cross-class-views']), ('disabled', [])):
                output = root / name
                result = subprocess.run(base + ['--output', str(output)] + extra, env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-8000:])
            enabled, payload = load_checkpoint(root / 'enabled/latest.pt')
            disabled, _ = load_checkpoint(root / 'disabled/latest.pt')
            self.assertTrue(payload['trainer']['alignment_cross_class_views'])
            metrics = {name: json.loads((root / name / 'metrics.jsonl').read_text().splitlines()[0])
                       for name in ('enabled', 'disabled')}
            self.assertEqual(metrics['disabled']['alignment'], 0.)
            self.assertGreater(metrics['enabled']['alignment'], .01)
            for index in (1, 2):
                self.assertGreater((enabled.alignment[index].weight - disabled.alignment[index].weight)
                                   .abs().max().item(), 0)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo indisponible')
    def test_balanced_modalities_pair_repeated_rows_across_ranks(self):
        batch = {'input_ids': torch.tensor([[1, 2], [1, 3]]), 'labels': torch.tensor([[1, 2], [1, 3]]),
                 'series': torch.ones(2, 2, 3), 'class_labels': torch.tensor([0, 1])}
        selected = [balanced_modalities(batch, index, seed=0) for index in (0, 1)]
        self.assertIn('input_ids', selected[0])
        self.assertNotIn('series', selected[0])
        self.assertIn('series', selected[1])
        self.assertNotIn('input_ids', selected[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(ChimeraConfig.preset('tiny'), alignment_scope='global',
                             alignment_text_view='isolated', dropout=0.)
            config_path, data = root / 'config.json', root / 'data.jsonl'
            config.save(config_path)
            rows = [dict(text='signal ' + str(index), series=[[float(index), 1., 2.], [2., 1., float(index)]],
                         class_label=index) for index in range(2)]
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                    '-m', 'chimera', '--threads', '1', 'train', '--config', str(config_path), '--data', str(data),
                    '--steps', '1', '--batch-size', '2', '--mixed-modalities', '--modality-policy', 'balanced',
                    '--seed', '0', '--log-every', '1']
            env = dict(os.environ)
            for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name, None)
            for normalization in ('batch', 'observed'):
                losses = {}
                for name, extra in (('enabled', []), ('disabled', ['--no-alignment-duplicate-positives'])):
                    output = root / f'{normalization}-{name}'
                    result = subprocess.run(base + ['--loss-normalization', normalization, '--output', str(output)] + extra,
                                            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout[-8000:])
                    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
                    losses[name] = metrics[0]['alignment']
                self.assertEqual(losses['disabled'], 0.)
                self.assertGreater(losses['enabled'], .01)
                enabled, _ = load_checkpoint(root / f'{normalization}-enabled/latest.pt')
                disabled, _ = load_checkpoint(root / f'{normalization}-disabled/latest.pt')
                for index in (1, 2):
                    difference = (enabled.alignment[index].weight - disabled.alignment[index].weight).abs().max().item()
                    self.assertGreater(difference, 0, f'{normalization}, projection {index}')


if __name__ == '__main__':
    unittest.main()
