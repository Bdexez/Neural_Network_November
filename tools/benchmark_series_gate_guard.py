"""Mesurer une protection de porte sur des séries synthétiques indépendantes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.augmentation import SeriesAugmentation
from chimera.calibration import file_sha256
from chimera.cli import prediction_stream, same_file
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.modalities import keep_modalities
from chimera.series_classifier import fit_series_classifier
from chimera.series_gate import fit_series_gate
from chimera.series_gate_guard import fit_series_gate_guard
from chimera.training import load_checkpoint


TRANSFORMS = {'identity': (1., 0.), 'shift1': (1., 1.), 'shift3': (1., 3.), 'shift20': (1., 20.),
              'scale0.5': (.5, 0.), 'scale0.1': (.1, 0.), 'scale2': (2., 0.), 'scale10': (10., 0.),
              'scale4_shift-50': (4., -50.)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--head-batches', type=int, default=96)
    parser.add_argument('--gate-batches', type=int, default=192)
    parser.add_argument('--validation-batches', type=int, default=48)
    parser.add_argument('--guard-batches', type=int, default=48)
    parser.add_argument('--test-batches', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--head-seed', type=int, default=78000000)
    parser.add_argument('--head-mask-seed', type=int, default=79000000)
    parser.add_argument('--gate-seed', type=int, default=80000000)
    parser.add_argument('--gate-mask-seed', type=int, default=81000000)
    parser.add_argument('--gate-transform-seed', type=int, default=82000000)
    parser.add_argument('--validation-seed', type=int, default=83000000)
    parser.add_argument('--validation-mask-seed', type=int, default=84000000)
    parser.add_argument('--validation-transform-seed', type=int, default=85000000)
    parser.add_argument('--guard-seed', type=int, default=88000000)
    parser.add_argument('--guard-mask-seed', type=int, default=89000000)
    parser.add_argument('--max-harm', type=float, default=.01)
    parser.add_argument('--test-seed', type=int, default=90000000)
    parser.add_argument('--test-mask-seed', type=int, default=91000000)
    parser.add_argument('--optimizer-seed', type=int, default=713)
    parser.add_argument('--shape-context', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--fit-batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--threshold', type=float, default=.03)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.head_batches, args.gate_batches, args.validation_batches, args.guard_batches, args.test_batches,
           args.batch_size, args.epochs, args.fit_batch_size, args.threads) < 1:
        parser.error('Tailles strictement positives requises')
    ranges = [(args.head_seed, args.head_batches), (args.head_mask_seed, args.head_batches),
              (args.gate_seed, args.gate_batches), (args.gate_mask_seed, args.gate_batches),
              (args.gate_transform_seed, 1), (args.validation_seed, args.validation_batches),
              (args.validation_mask_seed, args.validation_batches), (args.validation_transform_seed, 1),
              (args.guard_seed, args.guard_batches), (args.guard_mask_seed, args.guard_batches),
              (args.test_seed, args.test_batches), (args.test_mask_seed, args.test_batches), (args.optimizer_seed, 1)]
    if any(start < 0 for start, _ in ranges) or any(
            max(a, b) < min(a+n, b+m) for i, (a, n) in enumerate(ranges) for b, m in ranges[i+1:]):
        parser.error('Les plages de seeds doivent être positives et disjointes')
    if same_file(args.output, args.checkpoint):
        parser.error('La sortie doit être distincte du checkpoint')
    torch.set_num_threads(args.threads)
    checkpoint_sha = file_sha256(args.checkpoint)
    model, payload = load_checkpoint(args.checkpoint)
    if model.config.max_series_len < 32:
        parser.error('Les scénarios requièrent max_series_len >= 32')
    augmentation = SeriesAugmentation(missing_rate=.3, mask_probability=.5)

    def batches(count, seed, mask_seed, *, augment=True, length=None):
        for index in range(count):
            batch = synthetic_batch(model.config, batch_size=args.batch_size, seed=seed + index)
            current = length if length is not None else (8, 16, 32)[index % 3]
            batch['series'], batch['series_mask'] = batch['series'][:, -current:], batch['series_mask'][:, -current:]
            batch = keep_modalities(batch, ('series',))
            if augment:
                batch = augmentation(batch, index=index, seed=mask_seed)
            batch['series'] = batch['series'].double()
            yield {key: value for key, value in batch.items() if key in ('series', 'series_mask', 'class_labels')}

    def transformed(count, seed, mask_seed, transform_seed):
        generator = torch.Generator().manual_seed(transform_seed)
        for batch in batches(count, seed, mask_seed):
            n = batch['series'].size(0)
            kind = torch.rand(n, generator=generator)
            shift = (torch.rand(n, generator=generator) * 6 - 3).double()
            scale = torch.exp((torch.rand(n, generator=generator) * 2 - 1).double() * 1.6)
            shift = torch.where(kind < .3, torch.zeros_like(shift), shift)
            scale = torch.where(kind < .5, torch.ones_like(scale), scale)
            shift = torch.where(kind > .95, shift * 8, shift)
            batch['series'] = batch['series'] * scale[:, None, None] + shift[:, None, None]
            yield batch

    classifier = fit_series_classifier(model, batches(args.head_batches, args.head_seed, args.head_mask_seed,
                                                       augment=False), shape_context=args.shape_context,
                                       augmentation=augmentation, augmentation_seed=args.head_mask_seed,
                                       augmentation_batch_size=args.batch_size, checkpoint_sha256=checkpoint_sha)
    classifier_bytes = json.dumps(classifier.to_dict(), sort_keys=True, separators=(',', ':')).encode()
    classifier_sha = hashlib.sha256(classifier_bytes).hexdigest()
    gate = fit_series_gate(model, classifier,
                           transformed(args.gate_batches, args.gate_seed, args.gate_mask_seed, args.gate_transform_seed),
                           transformed(args.validation_batches, args.validation_seed, args.validation_mask_seed,
                                       args.validation_transform_seed), checkpoint_sha256=checkpoint_sha,
                           classifier_sha256=classifier_sha, epochs=args.epochs, batch_size=args.fit_batch_size,
                           learning_rate=args.learning_rate, threshold=args.threshold, seed=args.optimizer_seed,
                           feature_batch_size=args.batch_size)
    gate_bytes = json.dumps(gate.to_dict(), sort_keys=True, separators=(',', ':')).encode()
    gate_sha = hashlib.sha256(gate_bytes).hexdigest()
    guard = fit_series_gate_guard(model, classifier, gate,
                                  batches(args.guard_batches, args.guard_seed, args.guard_mask_seed),
                                  checkpoint_sha256=checkpoint_sha, classifier_sha256=classifier_sha,
                                  gate_sha256=gate_sha, max_harm=args.max_harm, feature_batch_size=args.batch_size)
    results = {}
    for length in (8, 16, 32):
        results[str(length)] = {}
        for name, (scale, shift) in TRANSFORMS.items():
            def test():
                for batch in batches(args.test_batches, args.test_seed, args.test_mask_seed, length=length):
                    batch['series'] = batch['series'] * scale + shift
                    yield batch
            report = evaluate(model, test(), series_classifier=classifier, series_gate=gate, series_gate_guard=guard)
            results[str(length)][name] = dict(examples=report['series_gate_labeled_examples'],
                                              base_accuracy=report['accuracy'],
                                              head_accuracy=report['series_classifier_accuracy'],
                                              unguarded_accuracy=report['series_gate_guard_unguarded_accuracy'],
                                              guarded_accuracy=report['series_gate_accuracy'],
                                              guarded_gain_vs_base=report['series_gate_gain_vs_base'],
                                              guarded_harm_rate=report['series_gate_guard_harm_rate'],
                                              guarded_benefit_rate=report['series_gate_guard_benefit_rate'],
                                              head_selection_rate=report['series_gate_head_selection_rate'],
                                              prevented_selection_rate=report['series_gate_guard_prevented_selection_rate'])
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError('Le checkpoint a changé pendant la comparaison')
    guard_bytes = json.dumps(guard.to_dict(), sort_keys=True, separators=(',', ':')).encode()
    result = dict(settings=vars(args), checkpoint_sha256=checkpoint_sha, checkpoint_step=payload['step'],
                  classifier_sha256=classifier_sha, gate_sha256=gate_sha,
                  guard_sha256=hashlib.sha256(guard_bytes).hexdigest(), guard=guard.to_dict(),
                  artifact_hash_format='JSON UTF-8 avec clés triées et séparateurs compacts',
                  gate=dict(train_examples=gate.train_examples, validation_examples=gate.validation_examples,
                            best_epoch=gate.best_epoch, validation_gain=gate.validation_gain),
                  results=results, torch_version=str(torch.__version__),
                  scope='Un checkpoint et des données synthétiques ; ne démontre pas la robustesse sur données réelles.')
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=args.output, checkpoint_sha256=checkpoint_sha)))


if __name__ == '__main__':
    main()
