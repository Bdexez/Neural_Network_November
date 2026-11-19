"""Comparer les corrections sur historiques courts et valeurs manquantes synthétiques."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.calibration import file_sha256
from chimera.cli import prediction_stream, same_file
from chimera.evaluation import evaluate
from chimera.forecast_adapter import fit_forecast_adapter
from chimera.training import load_checkpoint
from tools.benchmark_robustness import CASES, stress_batches


TRAIN_CASES = ("complete", "history16", "history8", "missing10", "missing30",
               "history16_missing30", "history8_missing30")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-batches", type=int, default=64)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=22000000)
    parser.add_argument("--train-mask-seed", type=int, default=23000000)
    parser.add_argument("--test-seed", type=int, default=24000000)
    parser.add_argument("--test-mask-seed", type=int, default=25000000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.train_batches, args.test_batches, args.batch_size, args.threads) < 1:
        parser.error("Tailles et threads doivent être positifs")
    ranges = [(args.train_seed, args.train_batches), (args.train_mask_seed, args.train_batches),
              (args.test_seed, args.test_batches), (args.test_mask_seed, args.test_batches)]
    if any(start < 0 for start, _ in ranges):
        parser.error("Les seeds doivent être positives ou nulles")
    if any(max(a, b) < min(a + n, b + m)
           for i, (a, n) in enumerate(ranges) for b, m in ranges[i + 1:]):
        parser.error("Les plages de seeds ne doivent pas se chevaucher")
    if same_file(args.output, args.checkpoint):
        parser.error("La sortie doit être distincte du checkpoint")
    torch.set_num_threads(args.threads)
    checkpoint_sha = file_sha256(args.checkpoint)
    measurements = {}
    for label, overrides in (("base", None),
                             ("length_gate", {"forecast_adapter_context": 8, "forecast_adapter_missing_gate": False}),
                             ("missing_gate", {"forecast_adapter_context": 8, "forecast_adapter_missing_gate": True}),
                             ("separate_missing", {"forecast_adapter_context": 8, "forecast_adapter_missing_gate": True,
                                                   "forecast_adapter_separate_missing": True})):
        model, payload = load_checkpoint(args.checkpoint, overrides=overrides)
        if label == "base" and model.forecast_adapter is not None:
            parser.error("Le checkpoint de départ ne doit pas contenir d'adaptateur")
        if model.config.forecast_context != 32 or model.config.max_series_len < 32:
            parser.error("Les scénarios requièrent forecast_context=32 et max_series_len>=32")
        entry = {"fit": None, "cases": {}}
        if overrides is not None:
            def training_batches():
                for index in range(args.train_batches):
                    case = TRAIN_CASES[index % len(TRAIN_CASES)]
                    yield from stress_batches(model.config, case=case, names=("series",), batches=1,
                                              batch_size=args.batch_size, seed=args.train_seed + index,
                                              mask_seed=args.train_mask_seed + index)
            entry["fit"] = fit_forecast_adapter(model, training_batches())
        for case in CASES:
            entry["cases"][case] = {}
            for modality, names in (("series", ("series",)), ("all", ("input_ids", "images", "series"))):
                batches = stress_batches(model.config, case=case, names=names, batches=args.test_batches,
                                         batch_size=args.batch_size, seed=args.test_seed, mask_seed=args.test_mask_seed)
                metrics = evaluate(model, batches)
                entry["cases"][case][modality] = {
                    "forecast_mae": metrics["forecast_mae"], "forecast_rmse": metrics["forecast_rmse"],
                    "forecast_interval_coverage": metrics["forecast_interval_coverage"],
                    "accuracy": metrics["accuracy"],
                }
        measurements[label] = entry
    if file_sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError("Le checkpoint a changé pendant la comparaison")
    report = dict(settings=vars(args), checkpoint_sha256=checkpoint_sha, checkpoint_step=payload["step"],
                  torch_version=torch.__version__, train_cases=list(TRAIN_CASES), measurements=measurements,
                  scope="Données synthétiques ; masquage indépendant des valeurs ; aucune garantie sur données réelles.")
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=args.output, checkpoint_sha256=checkpoint_sha)))


if __name__ == "__main__":
    main()
