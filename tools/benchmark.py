"""Comparaison reproductible avant/après entraînement, sur données synthétiques inédites.

Exécuter : .venv/bin/python tools/benchmark.py --checkpoint runs/workstation/latest.pt
"""

import argparse
import gc
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera import Chimera, ChimeraConfig
from chimera.cli import inspect_config
from chimera.data import synthetic_batch
from chimera.training import evaluate, load_checkpoint, seed_everything


def filter_modalities(batch, names):
    keys = {"class_labels", "forecast_targets"}
    if "text" in names:
        keys.update(("input_ids", "labels", "attention_mask"))
    if "image" in names:
        keys.add("images")
    if "series" in names:
        keys.update(("series", "series_mask"))
    return {key: value for key, value in batch.items() if key in keys}


def measure(model, batches):
    result = {}
    for count in (1, 2, 3):
        for names in itertools.combinations(("text", "image", "series"), count):
            result["+".join(names)] = evaluate(model, (filter_modalities(b, names) for b in batches))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", help="Comparer à ces poids plutôt qu'à une initialisation aléatoire")
    parser.add_argument("--output", default="artifacts/benchmark.json")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=2000000, help="Première seed des lots de test synthétiques")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.threads) < 1:
        parser.error("batches, batch-size et threads doivent être > 0")
    if args.seed_start < 0:
        parser.error("seed-start doit être >= 0")
    torch.set_num_threads(args.threads)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = ChimeraConfig.from_dict(payload["config"])
    step = payload["step"]
    seed = payload.get("trainer", {}).get("seed", 42)
    initialization = payload.get("trainer", {}).get("initialization")
    del payload
    batches = [synthetic_batch(config, batch_size=args.batch_size, seed=args.seed_start + i)
               for i in range(args.batches)]
    persistence_error, values = 0, 0
    for batch in batches:
        target = batch["forecast_targets"]
        prediction = batch["series"][:, -1:, :].expand_as(target)
        persistence_error += (prediction - target).abs().sum().item()
        values += target.numel()
    reference = {"seed": seed, "kind": "random_initialization"}
    if args.reference_checkpoint:
        initial, payload = load_checkpoint(args.reference_checkpoint)
        if initial.config.to_dict() != config.to_dict():
            raise ValueError("Le checkpoint de référence doit utiliser la même configuration")
        reference = {"checkpoint": args.reference_checkpoint, "step": payload["step"], "kind": "checkpoint"}
        del payload
    else:
        seed_everything(seed)
        initial = Chimera(config)
    before = measure(initial, batches)
    del initial
    gc.collect()
    trained, payload = load_checkpoint(args.checkpoint)
    del payload
    after = measure(trained, batches)
    report = {
        "checkpoint": args.checkpoint, "training_steps": step,
        "reference": reference, "initialization": initialization,
        "torch_version": torch.__version__, "evaluation_seed_start": args.seed_start,
        "evaluation_examples": args.batches * args.batch_size,
        "architecture": inspect_config(config), "before": before, "after": after,
        "persistence_forecast_mae": persistence_error / values,
        "scope": "Synthetic held-out data only; does not establish real-world language, vision or forecasting quality.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
