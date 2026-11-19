"""Comparer couvertures marginale et simultanée sur des trajectoires synthétiques disjointes."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from benchmark import filter_modalities
from chimera.calibration import file_sha256, fit_calibration
from chimera.cli import prediction_stream, same_file
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--test-batches", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=4100000)
    parser.add_argument("--test-seed", type=int, default=5100000)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.calibration_batches, args.test_batches, args.threads) < 1:
        parser.error("Les tailles et nombres de lots doivent être positifs")
    if min(args.calibration_seed, args.test_seed) < 0:
        parser.error("Les seeds doivent être positives ou nulles")
    if not math.isfinite(args.coverage) or not 0 < args.coverage < 1:
        parser.error("coverage doit appartenir à ]0, 1[")
    cal_end = args.calibration_seed + args.calibration_batches
    test_end = args.test_seed + args.test_batches
    if args.calibration_seed < test_end and args.test_seed < cal_end:
        parser.error("Les seeds de calibration et de test doivent être disjointes")
    if same_file(args.output, args.checkpoint):
        parser.error("La sortie doit être distincte du checkpoint")
    torch.set_num_threads(args.threads)
    model, payload = load_checkpoint(args.checkpoint)
    checkpoint_sha = file_sha256(args.checkpoint)
    calibration_batches = [synthetic_batch(model.config, batch_size=args.batch_size,
                                           seed=args.calibration_seed + index)
                           for index in range(args.calibration_batches)]
    test_batches = [synthetic_batch(model.config, batch_size=args.batch_size, seed=args.test_seed + index)
                    for index in range(args.test_batches)]
    measurements = {}
    for label, names in (("series", ("series",)), ("all", ("text", "image", "series"))):
        calibration_data = [filter_modalities(batch, names) for batch in calibration_batches]
        test_data = [filter_modalities(batch, names) for batch in test_batches]
        fitted = {scope: fit_calibration(model, calibration_data, coverage=args.coverage, scope=scope,
                                         checkpoint_sha256=checkpoint_sha)
                  for scope in ("marginal", "joint")}
        measurements[label] = dict(raw=evaluate(model, test_data),
                                   **{scope: dict(calibration=calibration.to_dict(),
                                                  evaluation=evaluate(model, test_data, calibration=calibration))
                                      for scope, calibration in fitted.items()})
    report = dict(settings=vars(args), checkpoint_sha256=checkpoint_sha, checkpoint_step=payload["step"],
                  torch_version=str(torch.__version__), measurements=measurements,
                  scope="Synthetic complete trajectories only; not evidence of coverage on dependent real series.")
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=args.output, calibration_examples=args.batch_size * args.calibration_batches,
                          test_examples=args.batch_size * args.test_batches)))


if __name__ == "__main__":
    main()
