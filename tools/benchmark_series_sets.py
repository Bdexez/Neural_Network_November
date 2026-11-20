"""Mesurer la calibration de probabilités et les ensembles conformes sur quatre jeux synthétiques disjoints."""

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
from chimera.series_probability import fit_series_probability
from chimera.series_set import fit_series_set
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-batches", type=int, default=96)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--set-batches", type=int, default=32)
    parser.add_argument("--test-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=58000000)
    parser.add_argument("--train-mask-seed", type=int, default=59000000)
    parser.add_argument("--calibration-seed", type=int, default=60000000)
    parser.add_argument("--calibration-mask-seed", type=int, default=61000000)
    parser.add_argument("--set-seed", type=int, default=62000000)
    parser.add_argument("--set-mask-seed", type=int, default=63000000)
    parser.add_argument("--coverage", type=float, default=.9)
    parser.add_argument("--test-seed", type=int, default=64000000)
    parser.add_argument("--test-mask-seed", type=int, default=65000000)
    parser.add_argument("--shape-context", type=int, default=16)
    parser.add_argument("--missing-rate", type=float, default=.3)
    parser.add_argument("--mask-probability", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.train_batches, args.calibration_batches, args.set_batches, args.test_batches, args.batch_size, args.threads) < 1:
        parser.error("Tailles et threads strictement positifs requis")
    ranges = [(args.train_seed, args.train_batches), (args.train_mask_seed, args.train_batches),
              (args.calibration_seed, args.calibration_batches), (args.calibration_mask_seed, args.calibration_batches),
              (args.set_seed, args.set_batches), (args.set_mask_seed, args.set_batches),
              (args.test_seed, args.test_batches), (args.test_mask_seed, args.test_batches)]
    if any(start < 0 for start, _ in ranges) or any(
            max(a, b) < min(a + n, b + m) for i, (a, n) in enumerate(ranges) for b, m in ranges[i + 1:]):
        parser.error("Les plages de seeds doivent être positives et disjointes")
    if same_file(args.output, args.checkpoint):
        parser.error("La sortie doit être distincte du checkpoint")
    augmentation = SeriesAugmentation(missing_rate=args.missing_rate, mask_probability=args.mask_probability)
    torch.set_num_threads(args.threads)
    checkpoint_sha = file_sha256(args.checkpoint)
    model, payload = load_checkpoint(args.checkpoint)
    if model.config.max_series_len < 32:
        parser.error("Les scénarios requièrent max_series_len >= 32")

    def batches(count, seed, mask_seed, *, augment, shift=0.):
        for index in range(count):
            batch = synthetic_batch(model.config, batch_size=args.batch_size, seed=seed + index)
            length = (8, 16, 32)[index % 3]
            batch["series"], batch["series_mask"] = batch["series"][:, -length:], batch["series_mask"][:, -length:]
            batch = keep_modalities(batch, ("series",))
            if augment:
                batch = augmentation(batch, index=index, seed=mask_seed)
            batch["series"] = batch["series"].double() + shift
            batch["forecast_targets"] = batch["forecast_targets"].double() + shift
            yield batch

    classifier = fit_series_classifier(model, batches(args.train_batches, args.train_seed, args.train_mask_seed,
                                                        augment=False), shape_context=args.shape_context,
                                       augmentation=augmentation, augmentation_seed=args.train_mask_seed,
                                       augmentation_batch_size=args.batch_size, checkpoint_sha256=checkpoint_sha)
    classifier_data = json.dumps(classifier.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    classifier_sha = hashlib.sha256(classifier_data).hexdigest()
    probability = fit_series_probability(model, classifier,
                                         batches(args.calibration_batches, args.calibration_seed,
                                                 args.calibration_mask_seed, augment=True),
                                         checkpoint_sha256=checkpoint_sha, classifier_sha256=classifier_sha)
    probability_data = json.dumps(probability.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    probability_sha = hashlib.sha256(probability_data).hexdigest()
    measurements, sets = {}, {}
    for method in ("lac", "aps"):
        fitted = fit_series_set(model, classifier,
                                batches(args.set_batches, args.set_seed, args.set_mask_seed, augment=True),
                                checkpoint_sha256=checkpoint_sha, classifier_sha256=classifier_sha,
                                probability=probability, probability_sha256=probability_sha,
                                method=method, coverage=args.coverage)
        sets[method] = fitted.to_dict()
        measurements[method] = {}
        for label, shift in (("identity", 0.), ("shift20", 20.)):
            report = evaluate(model, batches(args.test_batches, args.test_seed, args.test_mask_seed,
                                             augment=True, shift=shift), series_classifier=classifier,
                              series_probability=probability, series_set=fitted)
            measurements[method][label] = {key: value for key, value in report.items()
                                            if key.startswith("series_prediction_set_")}
            measurements[method][label]["accuracy"] = report["series_classifier_accuracy"]
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError("Le checkpoint a changé pendant la comparaison")
    result = dict(settings=vars(args), checkpoint_sha256=checkpoint_sha, checkpoint_step=payload["step"],
                  classifier_sha256=classifier_sha, classifier=classifier.to_dict(), calibration=probability.to_dict(),
                  probability_sha256=probability_sha, prediction_sets=sets,
                  classifier_hash_format="JSON UTF-8 avec clés triées et séparateurs compacts",
                  measurements=measurements, torch_version=torch.__version__,
                  scope="Données synthétiques indépendantes ; ne démontre pas une calibration sur données réelles.")
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=args.output, checkpoint_sha256=checkpoint_sha)))


if __name__ == "__main__":
    main()
