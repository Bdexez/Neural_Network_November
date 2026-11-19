"""Comparer les portes top-1 sur des entraînements synthétiques appariés."""

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

from benchmark import measure
from chimera.data import synthetic_batch
from chimera.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batches", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", default="runs/top1-routing-study")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.test_batches, args.threads) < 1:
        parser.error("Les nombres d'étapes, de lots et de threads doivent être positifs")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Les seeds doivent être distinctes")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir doit être vide")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    base = json.loads((ROOT / "configs/tiny.json").read_text())
    runs = []
    for seed in args.seeds:
        for mode in ("normalized", "probability"):
            directory = output / f"{mode}-{seed}"
            directory.mkdir()
            config = directory / "input-config.json"
            config.write_text(json.dumps(base | dict(top_k=1, top1_gate=mode), indent=2) + "\n")
            command = [sys.executable, "-m", "chimera", "--threads", str(args.threads), "train",
                       "--config", str(config), "--seed", str(seed), "--steps", str(args.steps),
                       "--batch-size", str(args.batch_size), "--lr", "0.001", "--warmup", "10",
                       "--modality-dropout", "0.3", "--save-every", str(args.steps),
                       "--eval-every", str(args.steps), "--output", str(directory)]
            print(json.dumps(dict(event="training", seed=seed, mode=mode)), flush=True)
            started = time.monotonic()
            with (directory / "train.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            elapsed = time.monotonic() - started
            model, _ = load_checkpoint(directory / "latest.pt")
            batches = [synthetic_batch(model.config, batch_size=args.batch_size, seed=3100000 + index)
                       for index in range(args.test_batches)]
            metrics = measure(model, batches)
            runs.append(dict(seed=seed, mode=mode, training_seconds=elapsed, command=command, metrics=metrics))
            print(json.dumps(dict(event="evaluated", seed=seed, mode=mode,
                                  series_accuracy=metrics["series"]["accuracy"],
                                  series_mae=metrics["series"]["forecast_mae"])), flush=True)
            del model, batches
            gc.collect()
    aggregate = {}
    for mode in ("normalized", "probability"):
        records = [run for run in runs if run["mode"] == mode]
        aggregate[mode] = {}
        for modality in records[0]["metrics"]:
            aggregate[mode][modality] = {}
            for metric in ("accuracy", "forecast_mae", "text"):
                values = [run["metrics"][modality].get(metric) for run in records]
                if all(value is not None for value in values):
                    aggregate[mode][modality][metric] = dict(
                        mean=statistics.mean(values),
                        sample_std=statistics.stdev(values) if len(values) > 1 else None,
                    )
    report = dict(settings=vars(args), torch_version=str(torch.__version__), runs=runs, aggregate=aggregate,
                  test_seed_start=3100000, test_examples=args.batch_size * args.test_batches,
                  checkpoint_selection="Final step; identical seeds, optimizer steps and batch sizes",
                  scope="Synthetic held-out examples only; results do not establish real-world quality.")
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(event="complete", report=str(output / "comparison.json"))), flush=True)


if __name__ == "__main__":
    main()
