"""Comparer l'alignement par classe sur des sources synthétiques non appariées."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from chimera.calibration import file_sha256
from chimera.cli import prediction_stream
from chimera.data import JointDataset, synthetic_batch
from chimera.tokenizer import ByteTokenizer
from chimera.training import load_checkpoint
from tools.compare_shared_modality_batches import exposure, measure


def write_data(path, config, *, count, text_seed, series_seed):
    """Chaque ligne ne contient qu'une modalité ; les sources ont des tirages distincts."""
    text = synthetic_batch(config, batch_size=count, seed=text_seed)
    series = synthetic_batch(config, batch_size=count, seed=series_seed)
    tokenizer = ByteTokenizer()
    with path.open('x', encoding='utf-8') as stream:
        for index in range(count):
            row = dict(text=tokenizer.decode(text['input_ids'][index]),
                       class_label=text['class_labels'][index].item())
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        for index in range(count):
            row = dict(series=series['series'][index].tolist(),
                       future=series['forecast_targets'][index].tolist(),
                       class_label=series['class_labels'][index].item())
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True, help='Nouveau dossier de travail')
    parser.add_argument('--seeds', nargs='+', type=int, default=[211000000, 213000000, 215000000])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--train-examples-per-modality', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--text-seed', type=int, default=210000000)
    parser.add_argument('--series-seed', type=int, default=210000001)
    parser.add_argument('--test-seed', type=int, default=220000000)
    parser.add_argument('--test-batches', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--alignment-weight', type=float, default=.2)
    args = parser.parse_args()
    test_range = range(args.test_seed, args.test_seed + args.test_batches)
    validation_seeds = [seed + 1000000 for seed in args.seeds]
    if (min(args.steps, args.train_examples_per_modality, args.batch_size, args.test_batches) < 1
            or args.test_batch_size < 2 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.alignment_weight) or args.alignment_weight < 0
            or len(set(args.seeds)) != len(args.seeds)
            or any(seed < 0 or seed >= 2**63 for seed in [*validation_seeds, *args.seeds, args.text_seed,
                                                           args.series_seed, args.test_seed])
            or args.test_seed + args.test_batches >= 2**63
            or args.text_seed == args.series_seed or args.text_seed in test_range or args.series_seed in test_range
            or any(seed in test_range or seed in (args.text_seed, args.series_seed) for seed in validation_seeds)):
        parser.error('Tailles, seeds, taux ou poids invalides ; les sources et le test doivent être distincts')
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = file_sha256(checkpoint)
    output = Path(args.output_dir).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('output-dir doit être un nouveau dossier')
    torch.set_num_threads(2)
    parent, _ = load_checkpoint(checkpoint)
    data = output / 'train.jsonl'
    write_data(data, parent.config, count=args.train_examples_per_modality,
               text_seed=args.text_seed, series_seed=args.series_seed)
    dataset = JointDataset(data, parent.config, allow_mixed=True)
    data_sha = dataset.fingerprint
    del parent
    override = output / 'alignment.json'
    override.write_text(json.dumps(dict(alignment_scope='global', alignment_text_view='isolated',
                                        alignment_mode='class', alignment_weight=args.alignment_weight)) + '\n')
    env = dict(os.environ, OMP_NUM_THREADS='1')
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    runs = []
    for seed in args.seeds:
        for mode in ('control', 'class'):
            if file_sha256(checkpoint) != checkpoint_sha or file_sha256(data) != data_sha:
                raise ValueError('Le checkpoint ou les données ont changé pendant la comparaison')
            destination = output / f'{mode}-{seed}'
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                       '-m', 'chimera', '--threads', '1', 'train', '--init-from', str(checkpoint),
                       '--model-overrides', str(override), '--data', str(data), '--mixed-modalities',
                       '--no-alignment-duplicate-positives', '--modality-dropout', '0',
                       '--loss-normalization', 'observed', '--steps', str(args.steps),
                       '--batch-size', str(args.batch_size), '--seed', str(seed), '--lr', str(args.lr),
                       '--warmup', str(min(10, args.steps)), '--eval-every', str(args.steps), '--log-every', '1',
                       '--save-every', str(args.steps), '--output', str(destination)]
            if mode == 'class':
                command.append('--alignment-cross-class-views')
            print(json.dumps(dict(event='training', seed=seed, mode=mode)), flush=True)
            start = time.monotonic()
            with (output / f'{mode}-{seed}.log').open('x', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - start
            if result.returncode:
                raise RuntimeError(f'Échec de {mode}, seed {seed} ; voir le log')
            if file_sha256(checkpoint) != checkpoint_sha or file_sha256(data) != data_sha:
                raise ValueError('Le checkpoint ou les données ont changé pendant la comparaison')
            trained_path = destination / 'latest.pt'
            model, payload = load_checkpoint(trained_path)
            metrics = measure(model, seed=args.test_seed, batches=args.test_batches, batch_size=args.test_batch_size)
            logs = [json.loads(line) for line in (destination / 'metrics.jsonl').read_text().splitlines()]
            row = dict(seed=seed, mode=mode, step=payload['step'], checkpoint_sha256=file_sha256(trained_path),
                       elapsed_seconds=elapsed, exposure=exposure(dataset, batch_size=args.batch_size,
                                                                   seed=seed, steps=args.steps, shared=False),
                       nonzero_alignment_steps=sum(item.get('alignment', 0.) > 1e-8 for item in logs), metrics=metrics)
            runs.append(row)
            print(json.dumps(dict(event='evaluated', **row)), flush=True)
    aggregate = {}
    for key in ('text_to_series_class_top1', 'series_class_accuracy', 'series_forecast_mae'):
        baseline = [row['metrics'][key] for row in runs if row['mode'] == 'control']
        enabled = [row['metrics'][key] for row in runs if row['mode'] == 'class']
        diffs = [b - a for a, b in zip(baseline, enabled)]
        aggregate[key] = dict(control_mean=statistics.mean(baseline), class_mean=statistics.mean(enabled),
                              paired_difference_mean=statistics.mean(diffs),
                              paired_difference_std=statistics.stdev(diffs) if len(diffs) > 1 else None)
    report = dict(settings=vars(args), parent_sha256=checkpoint_sha, train_data_sha256=data_sha,
                  torch_version=str(torch.__version__), runs=runs, aggregate=aggregate,
                  scope="Données synthétiques non appariées ; le texte indique la classe. "
                        "L'indépendance vis-à-vis de l'entraînement du checkpoint parent n'est pas établie.")
    with prediction_stream(output / 'comparison.json') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(event='complete', report=str(output / 'comparison.json'))), flush=True)


if __name__ == '__main__':
    main()
