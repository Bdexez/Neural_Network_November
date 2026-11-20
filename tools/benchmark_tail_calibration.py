"""Comparer les erreurs de queue de deux calibrations sur des chocs synthétiques asymétriques."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.calibration import file_sha256, fit_calibration
from chimera.cli import prediction_stream, same_file
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.modalities import keep_modalities
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--test-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=33000000)
    parser.add_argument("--calibration-noise-seed", type=int, default=34000000)
    parser.add_argument("--test-seed", type=int, default=35000000)
    parser.add_argument("--test-noise-seed", type=int, default=36000000)
    parser.add_argument("--noise-scale", type=float, default=.5)
    parser.add_argument("--coverage", type=float, default=.8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.calibration_batches, args.test_batches, args.batch_size, args.threads) < 1:
        parser.error("Tailles et threads doivent être positifs")
    if not math.isfinite(args.noise_scale) or args.noise_scale <= 0:
        parser.error("noise-scale doit être fini et positif")
    ranges = [(args.calibration_seed, args.calibration_batches),
              (args.calibration_noise_seed, args.calibration_batches),
              (args.test_seed, args.test_batches), (args.test_noise_seed, args.test_batches)]
    if any(start < 0 for start, _ in ranges) or any(
            max(a, b) < min(a + n, b + m) for i, (a, n) in enumerate(ranges) for b, m in ranges[i + 1:]):
        parser.error("Les plages de seeds doivent être positives et disjointes")
    if same_file(args.output, args.checkpoint):
        parser.error("La sortie doit être distincte du checkpoint")
    torch.set_num_threads(args.threads)
    checkpoint_sha = file_sha256(args.checkpoint)
    model, payload = load_checkpoint(args.checkpoint)

    def batches(names, count, seed, noise_seed):
        for index in range(count):
            batch = synthetic_batch(model.config, batch_size=args.batch_size, seed=seed + index)
            generator = torch.Generator().manual_seed(noise_seed + index)
            uniform = torch.rand(batch["forecast_targets"].shape, generator=generator).clamp_min(1e-7).double()
            noise = (-uniform.log() - 1) * args.noise_scale
            transformed = batch["forecast_targets"].double() + noise
            if not torch.isfinite(transformed).all():
                raise ValueError("Les chocs synthétiques dépassent la précision numérique")
            batch["forecast_targets"] = transformed
            yield keep_modalities(batch, names)

    measurements = {}
    for label, names in (("series", ("series",)), ("all", ("input_ids", "images", "series"))):
        def calibration_data():
            return batches(names, args.calibration_batches, args.calibration_seed, args.calibration_noise_seed)
        def test_data():
            return batches(names, args.test_batches, args.test_seed, args.test_noise_seed)
        models = {"symmetric": fit_calibration(model, calibration_data(), coverage=args.coverage,
                                                checkpoint_sha256=checkpoint_sha),
                  "separate_tails": fit_calibration(model, calibration_data(), coverage=args.coverage,
                                                     lower_tail_fraction=.5, checkpoint_sha256=checkpoint_sha)}
        measurements[label] = {name: dict(calibration=calibration.to_dict(),
                                          evaluation=evaluate(model, test_data(), calibration=calibration))
                               for name, calibration in models.items()}
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError("Le checkpoint a changé pendant la comparaison")
    report = dict(settings=vars(args), checkpoint_sha256=checkpoint_sha, checkpoint_step=payload["step"],
                  torch_version=torch.__version__, measurements=measurements,
                  scope="Chocs exponentiels synthétiques indépendants ; ne valide pas des séries réelles ou dépendantes.")
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=args.output, checkpoint_sha256=checkpoint_sha)))


if __name__ == "__main__":
    main()
