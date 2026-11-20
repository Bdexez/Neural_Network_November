"""Comparer lots distribués partagés et répartis sur un jeu synthétique apparié."""
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
from chimera.modality_batch_sampler import ModalityBatchSampler
from chimera.tokenizer import ByteTokenizer
from chimera.training import load_checkpoint


def write_data(path, config, *, count, seed):
    batch = synthetic_batch(config, batch_size=count, seed=seed)
    with path.open('x', encoding='utf-8') as stream:
        for index in range(count):
            row = dict(text=ByteTokenizer().decode(batch['input_ids'][index]),
                       series=batch['series'][index].tolist(), class_label=batch['class_labels'][index].item(),
                       future=batch['forecast_targets'][index].tolist())
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def exposure(dataset, *, batch_size, seed, steps, shared, interval=1):
    samplers = [ModalityBatchSampler(dataset, batch_size, num_replicas=2, rank=rank, seed=seed,
                                     shared_across_ranks=shared, shared_interval=interval) for rank in range(2)]
    unique, total = 0, 0
    for epoch_start in range(0, steps, len(samplers[0])):
        epoch = epoch_start // len(samplers[0])
        for sampler in samplers:
            sampler.set_epoch(epoch)
        ranks = [list(sampler) for sampler in samplers]
        for offset in range(min(len(samplers[0]), steps - epoch_start)):
            indices = [index for rank in ranks for index in rank[offset]]
            total += len(indices)
            unique += len(set(indices))
    return dict(sample_views=total, unique_examples_per_step_sum=unique)


def measure(model, *, seed, batches, batch_size):
    """Recherche texte isolé vers série et tâches supervisées depuis la série seule."""
    model.eval()
    count = correct = retrieved = observed = 0
    absolute = 0.
    with torch.inference_mode():
        for index in range(batches):
            batch = synthetic_batch(model.config, batch_size=batch_size, seed=seed + index)
            vectors = model.encode_modalities(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                               series=batch['series'], series_mask=batch['series_mask'])
            selected = (vectors['text'] @ vectors['series'].T).argmax(-1)
            labels = batch['class_labels']
            retrieved += (labels[selected] == labels).sum().item()
            output = model(series=batch['series'], series_mask=batch['series_mask'])
            correct += (output.classification_logits.argmax(-1) == labels).sum().item()
            median = output.forecast[..., model.config.quantiles.index(.5)]
            absolute += (median - batch['forecast_targets']).abs().sum().item()
            observed += median.numel()
            count += labels.numel()
    return dict(examples=count, text_to_series_class_top1=retrieved / count,
                series_class_accuracy=correct / count, series_forecast_mae=absolute / observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True, help='Nouveau dossier de travail')
    parser.add_argument('--seeds', nargs='+', type=int, default=[211000000, 213000000, 215000000])
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--train-examples', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--train-seed', type=int, default=210000000)
    parser.add_argument('--test-seed', type=int, default=220000000)
    parser.add_argument('--test-batches', type=int, default=8)
    parser.add_argument('--test-batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--alignment-weight', type=float, default=.2)
    parser.add_argument('--partial-interval', type=int, help='Ajouter un mode partageant un micro-lot sur N')
    args = parser.parse_args()
    if (min(args.steps, args.train_examples, args.batch_size, args.test_batches) < 1 or args.test_batch_size < 2
            or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.alignment_weight)
            or args.alignment_weight < 0 or len(set(args.seeds)) != len(args.seeds)
            or args.partial_interval is not None and args.partial_interval < 2):
        parser.error('Tailles, seeds, taux ou poids invalides')
    validation_seeds = [seed + 1000000 for seed in args.seeds]
    test_range = range(args.test_seed, args.test_seed + args.test_batches)
    if (any(seed < 0 or seed >= 2**63 - 1000000 for seed in [*args.seeds, args.train_seed, args.test_seed])
            or args.test_seed + args.test_batches >= 2**63
            or args.train_seed in test_range or any(seed in test_range or seed == args.train_seed for seed in validation_seeds)):
        parser.error('Les seeds de données et de validation doivent être disjointes du test')
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
    write_data(data, parent.config, count=args.train_examples, seed=args.train_seed)
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
    modes = [('sharded', False, 1), ('shared', True, 1)]
    if args.partial_interval is not None:
        modes.append(('periodic', True, args.partial_interval))
    for seed in args.seeds:
        for label, shared, interval in modes:
            if file_sha256(checkpoint) != checkpoint_sha or file_sha256(data) != data_sha:
                raise ValueError('Le checkpoint ou les données ont changé pendant la comparaison')
            destination = output / f'{label}-{seed}'
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
                       '-m', 'chimera', '--threads', '1', 'train', '--init-from', str(checkpoint),
                       '--model-overrides', str(override), '--data', str(data), '--mixed-modalities',
                       '--modality-policy', 'balanced', '--loss-normalization', 'observed', '--steps', str(args.steps),
                       '--batch-size', str(args.batch_size), '--seed', str(seed), '--lr', str(args.lr),
                       '--warmup', str(min(10, args.steps)), '--eval-every', str(args.steps), '--log-every', '1',
                       '--save-every', str(args.steps), '--output', str(destination)]
            if shared:
                command.append('--shared-modality-batches')
                if interval != 1:
                    command.extend(['--shared-modality-batch-interval', str(interval)])
            print(json.dumps(dict(event='training', seed=seed, mode=label)), flush=True)
            start = time.monotonic()
            with (output / f'{label}-{seed}.log').open('x', encoding='utf-8') as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - start
            if result.returncode:
                raise RuntimeError(f'Échec de {label}, seed {seed} ; voir le log')
            if file_sha256(checkpoint) != checkpoint_sha or file_sha256(data) != data_sha:
                raise ValueError('Le checkpoint ou les données ont changé pendant la comparaison')
            trained_path = destination / 'latest.pt'
            model, payload = load_checkpoint(trained_path)
            metrics = measure(model, seed=args.test_seed, batches=args.test_batches, batch_size=args.test_batch_size)
            logs = [json.loads(line) for line in (destination / 'metrics.jsonl').read_text().splitlines()]
            row = dict(seed=seed, mode=label, step=payload['step'], checkpoint_sha256=file_sha256(trained_path),
                       elapsed_seconds=elapsed, exposure=exposure(dataset, batch_size=args.batch_size,
                                                                   seed=seed, steps=args.steps, shared=shared,
                                                                   interval=interval),
                       nonzero_alignment_steps=sum(item.get('alignment', 0.) > 1e-8 for item in logs), metrics=metrics)
            runs.append(row)
            print(json.dumps(dict(event='evaluated', **row)), flush=True)
    aggregate = {}
    for key in ('text_to_series_class_top1', 'series_class_accuracy', 'series_forecast_mae'):
        baseline = [row['metrics'][key] for row in runs if row['mode'] == 'sharded']
        shared = [row['metrics'][key] for row in runs if row['mode'] == 'shared']
        diffs = [b - a for a, b in zip(baseline, shared)]
        aggregate[key] = dict(sharded_mean=statistics.mean(baseline), shared_mean=statistics.mean(shared),
                              paired_difference_mean=statistics.mean(diffs),
                              paired_difference_std=statistics.stdev(diffs) if len(diffs) > 1 else None)
        if args.partial_interval is not None:
            periodic = [row['metrics'][key] for row in runs if row['mode'] == 'periodic']
            periodic_diffs = [b - a for a, b in zip(baseline, periodic)]
            aggregate[key].update(periodic_mean=statistics.mean(periodic),
                                  periodic_difference_mean=statistics.mean(periodic_diffs),
                                  periodic_difference_std=statistics.stdev(periodic_diffs) if len(periodic_diffs) > 1 else None)
    report = dict(settings=vars(args), parent_sha256=checkpoint_sha, train_data_sha256=data_sha,
                  torch_version=str(torch.__version__), runs=runs, aggregate=aggregate,
                  scope="Données synthétiques ; le texte indique la classe. Le partage traite moins de lignes uniques "
                        "par étape ; l'indépendance vis-à-vis de l'entraînement du checkpoint parent n'est pas établie.")
    with prediction_stream(output / 'comparison.json') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(event='complete', report=str(output / 'comparison.json'))), flush=True)


if __name__ == '__main__':
    main()
