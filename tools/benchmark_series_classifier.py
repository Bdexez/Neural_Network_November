"""Comparer une tête temporelle standardisée aux classifications du modèle de base."""

import argparse
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
from chimera.training import load_checkpoint


TRANSFORMS = {"identity": (1., 0.), "shift20": (1., 20.), "scale0.1": (.1, 0.),
              "scale10": (10., 0.), "scale4_shift-50": (4., -50.)}
CASES = {"history8": (8, 0.), "history16": (16, 0.), "complete": (32, 0.),
         "history8_missing30": (8, .3), "history16_missing30": (16, .3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-batches", type=int, default=96)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=43000000)
    parser.add_argument("--test-seed", type=int, default=44000000)
    parser.add_argument("--mask-seed", type=int, default=45000000)
    parser.add_argument("--train-mask-seed", type=int, default=46000000)
    parser.add_argument("--train-missing-rate", type=float, default=0.)
    parser.add_argument("--train-mask-probability", type=float, default=.5)
    parser.add_argument("--shape-context", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=.001)
    parser.add_argument("--scale-floor", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.train_batches, args.test_batches, args.batch_size, args.threads) < 1:
        parser.error("Tailles et threads strictement positifs requis")
    ranges = [(args.train_seed, args.train_batches), (args.test_seed, args.test_batches),
              (args.mask_seed, args.test_batches)]
    if args.train_missing_rate:
        ranges.append((args.train_mask_seed, args.train_batches))
    if any(start < 0 for start, _ in ranges) or any(
            max(a, b) < min(a + n, b + m) for i, (a, n) in enumerate(ranges) for b, m in ranges[i + 1:]):
        parser.error("Les plages de seeds doivent être positives et disjointes")
    if same_file(args.output, args.checkpoint):
        parser.error("La sortie doit être distincte du checkpoint")
    torch.set_num_threads(args.threads)
    digest = file_sha256(args.checkpoint)
    model, payload = load_checkpoint(args.checkpoint)
    if model.config.max_series_len < 32:
        parser.error("Les scénarios requièrent max_series_len >= 32")

    def training():
        for index in range(args.train_batches):
            batch = synthetic_batch(model.config, batch_size=args.batch_size, seed=args.train_seed + index)
            length = (8, 16, 32)[index % 3]
            batch["series"] = batch["series"][:, -length:]
            batch["series_mask"] = batch["series_mask"][:, -length:]
            yield keep_modalities(batch, ("series",))

    augmentation = SeriesAugmentation(missing_rate=args.train_missing_rate, mask_probability=args.train_mask_probability)
    augmentation = augmentation if args.train_missing_rate else None
    classifier = fit_series_classifier(model, training(), ridge=args.ridge, scale_floor=args.scale_floor,
                                       augmentation=augmentation, augmentation_seed=args.train_mask_seed,
                                       augmentation_batch_size=args.batch_size if augmentation is not None else None,
                                       checkpoint_sha256=digest)
    shaped = (fit_series_classifier(model, training(), ridge=args.ridge, scale_floor=args.scale_floor,
                                    shape_context=args.shape_context, augmentation=augmentation,
                                    augmentation_seed=args.train_mask_seed,
                                    augmentation_batch_size=args.batch_size if augmentation is not None else None,
                                    checkpoint_sha256=digest)
              if args.shape_context else None)
    model.eval()
    measurements = {}
    for case, (length, missing_rate) in CASES.items():
        measurements[case] = {}
        identity_scores = None
        identity_shaped_scores = None
        for transform, (gain, shift) in TRANSFORMS.items():
            def testing():
                for index in range(args.test_batches):
                    batch = synthetic_batch(model.config, batch_size=args.batch_size, seed=args.test_seed + index)
                    series = batch["series"][:, -length:].double()
                    if missing_rate:
                        missing = torch.rand(series.shape, generator=torch.Generator().manual_seed(args.mask_seed + index)) < missing_rate
                        series.masked_fill_(missing, float("nan"))
                    batch["series"] = series * gain + shift
                    batch["series_mask"] = batch["series_mask"][:, -length:]
                    batch["forecast_targets"] = batch["forecast_targets"].double() * gain + shift
                    yield keep_modalities(batch, ("series",))
            report = evaluate(model, testing(), series_classifier=classifier)
            with torch.inference_mode():
                scores, shaped_scores, labels = [], [], []
                for batch in testing():
                    scores.append(classifier.scores(model, batch).cpu())
                    labels.append(batch["class_labels"].cpu())
                    if shaped is not None:
                        shaped_scores.append(shaped.scores(model, batch).cpu())
                scores, labels = torch.cat(scores), torch.cat(labels)
                if shaped is not None:
                    shaped_scores = torch.cat(shaped_scores)
            if identity_scores is None:
                identity_scores = scores
                identity_shaped_scores = shaped_scores if shaped is not None else None
            measurements[case][transform] = dict(examples=report["classified_examples"],
                                                  base_accuracy=report["accuracy"],
                                                  series_classifier_accuracy=report["series_classifier_accuracy"],
                                                  series_classifier_confusion=report["series_classifier_confusion"],
                                                  changed_from_identity=int((scores.argmax(-1) != identity_scores.argmax(-1)).sum()),
                                                  max_score_difference=float((scores - identity_scores).abs().max()))
            if shaped is not None:
                measurements[case][transform].update(
                    shape_accuracy=float((shaped_scores.argmax(-1) == labels).double().mean()),
                    shape_changed_from_identity=int((shaped_scores.argmax(-1) != identity_shaped_scores.argmax(-1)).sum()),
                    shape_max_score_difference=float((shaped_scores - identity_shaped_scores).abs().max()))
    if file_sha256(args.checkpoint) != digest:
        raise ValueError("Le checkpoint a changé pendant la comparaison")
    result = dict(settings=vars(args), checkpoint_sha256=digest, checkpoint_step=payload["step"],
                  torch_version=torch.__version__, classifier=classifier.to_dict(), measurements=measurements,
                  scope="Données synthétiques appariées ; ne valide ni données réelles ni classification multimodale.")
    if shaped is not None:
        result["shape_classifier"] = shaped.to_dict()
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=args.output, checkpoint_sha256=digest)))


if __name__ == "__main__":
    main()
