"""Évaluer la calibration sur les sept combinaisons avec deux jeux disjoints."""

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.calibration import file_sha256, fit_calibration
from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.modalities import MODALITIES, keep_modalities
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="runs/calibration-study")
    parser.add_argument("--calibration-batches", type=int, default=32)
    parser.add_argument("--test-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calibration-seed", type=int, default=5000000)
    parser.add_argument("--test-seed", type=int, default=6000000)
    parser.add_argument("--coverage", type=float)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.calibration_batches, args.test_batches, args.batch_size, args.threads) < 1:
        parser.error("Les tailles des jeux, des lots et le nombre de threads doivent être > 0")
    if min(args.calibration_seed, args.test_seed) < 0:
        parser.error("Les seeds doivent être >= 0")
    if max(args.calibration_seed, args.test_seed) < min(args.calibration_seed + args.calibration_batches,
                                                      args.test_seed + args.test_batches):
        parser.error("Les seeds de calibration et de test doivent être disjointes")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Le dossier de sortie doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    checkpoint_digest = file_sha256(args.checkpoint)
    model, payload = load_checkpoint(args.checkpoint)
    step = payload["step"]
    del payload

    def batches(names, seed, count):
        for index in range(count):
            yield keep_modalities(synthetic_batch(model.config, batch_size=args.batch_size, seed=seed + index), names)

    reports = {}
    for count in range(1, 4):
        for names in itertools.combinations(MODALITIES, count):
            label = "+".join(dict(input_ids="text", images="image", series="series")[name] for name in names)
            print(json.dumps(dict(event="calibrating", modalities=label)), flush=True)
            started = time.monotonic()
            calibration = fit_calibration(model, batches(names, args.calibration_seed, args.calibration_batches),
                                          coverage=args.coverage, checkpoint_sha256=checkpoint_digest)
            (output / f"{label}.json").write_text(json.dumps(calibration.to_dict(), indent=2, allow_nan=False) + "\n")
            metrics = evaluate(model, batches(names, args.test_seed, args.test_batches), calibration=calibration)
            reports[label] = dict(metrics=metrics, calibration=calibration.to_dict(),
                                  seconds=time.monotonic() - started)
            print(json.dumps(dict(event="evaluated", modalities=label,
                                  raw_coverage=metrics["forecast_interval_coverage"],
                                  calibrated_coverage=metrics["calibrated_interval_coverage"],
                                  raw_width=metrics["forecast_interval_width"],
                                  calibrated_width=metrics["calibrated_interval_width"])), flush=True)
    assert file_sha256(args.checkpoint) == checkpoint_digest
    report = dict(settings=vars(args), checkpoint_step=step, checkpoint_sha256=checkpoint_digest,
                  torch_version=torch.__version__, calibration_examples=args.batch_size * args.calibration_batches,
                  test_examples=args.batch_size * args.test_batches, reports=reports,
                  scope="Independent synthetic calibration/test seeds. Marginal per-target coverage, not simultaneous trajectory coverage or real-world validation.")
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="complete", report=str(output / "benchmark.json"))), flush=True)


if __name__ == "__main__":
    main()
