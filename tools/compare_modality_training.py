"""Comparer dropout et couverture équilibrée à initialisation et budget égaux.

Les deux politiques utilisent les mêmes lots synthétiques avant sélection des
modalités. Le test final est disjoint de l'entraînement et de la validation.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from chimera.data import synthetic_batch
from chimera.training import load_checkpoint
from benchmark import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--modality-dropout", type=float, default=0.3)
    parser.add_argument("--output-dir", default="runs/modality-study")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.test_batches, args.threads) < 1:
        parser.error("Les dimensions et les nombres d'étapes doivent être > 0")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Les seeds doivent être distinctes")
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir doit être vide pour éviter de mélanger plusieurs expériences")
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in args.seeds:
        for policy in ("dropout", "balanced"):
            directory = output / f"{policy}-{seed}"
            directory.mkdir()
            command = [sys.executable, "-m", "chimera", "--threads", str(args.threads),
                       "train", "--preset", "tiny", "--seed", str(seed), "--steps", str(args.steps),
                       "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                       "--modality-dropout", str(args.modality_dropout), "--modality-policy", policy,
                       "--save-every", str(args.steps), "--eval-every", str(args.steps),
                       "--output", str(directory)]
            print(json.dumps(dict(event="training", seed=seed, policy=policy)), flush=True)
            started = time.monotonic()
            with (directory / "train.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            seconds = time.monotonic() - started
            model, payload = load_checkpoint(directory / "latest.pt")
            del payload
            batches = [synthetic_batch(model.config, batch_size=args.batch_size, seed=3000000 + i)
                       for i in range(args.test_batches)]
            metrics = measure(model, batches)
            counts = {}
            for line in (directory / "metrics.jsonl").read_text().splitlines():
                for name, count in json.loads(line)["modality_batches"].items():
                    counts[name] = counts.get(name, 0) + count
            runs.append(dict(seed=seed, policy=policy, training_seconds=seconds,
                             modality_batches=counts, metrics=metrics, command=command))
            print(json.dumps(dict(event="evaluated", seed=seed, policy=policy,
                                  series_accuracy=metrics["series"]["accuracy"],
                                  series_mae=metrics["series"]["forecast_mae"])), flush=True)
            del model, batches
            gc.collect()
    aggregate = {}
    for policy in ("dropout", "balanced"):
        records = [run for run in runs if run["policy"] == policy]
        aggregate[policy] = {}
        for modality in records[0]["metrics"]:
            aggregate[policy][modality] = {}
            for metric in ("accuracy", "forecast_mae", "forecast_interval_coverage", "text"):
                values = [run["metrics"][modality].get(metric) for run in records]
                if all(value is not None for value in values):
                    aggregate[policy][modality][metric] = dict(
                        mean=statistics.mean(values),
                        sample_std=statistics.stdev(values) if len(values) > 1 else None,
                    )
    report = dict(settings=vars(args), torch_version=torch.__version__, runs=runs, aggregate=aggregate,
                  test_seed_start=3000000, test_examples=args.batch_size * args.test_batches,
                  checkpoint_selection="Final step, identical optimizer-step and example budgets",
                  scope="Synthetic fixture only; three seeds do not establish general real-world quality.")
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="complete", report=str(output / "comparison.json"))), flush=True)


if __name__ == "__main__":
    main()
