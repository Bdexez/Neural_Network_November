"""Comparer des poids sur des historiques courts et partiellement observés."""

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.data import synthetic_batch
from chimera.evaluation import evaluate
from chimera.modalities import keep_modalities
from chimera.training import load_checkpoint


CASES = dict(complete=(32, 0.0), history16=(16, 0.0), history8=(8, 0.0),
             missing10=(32, 0.1), missing30=(32, 0.3),
             history16_missing30=(16, 0.3), history8_missing30=(8, 0.3))


def stress_batches(config, *, case, names, batches, batch_size, seed, mask_seed):
    length, missing_rate = CASES[case]
    for index in range(batches):
        batch = synthetic_batch(config, batch_size=batch_size, seed=seed + index)
        series = batch["series"].clone()
        if missing_rate:
            missing = torch.rand(series.shape, generator=torch.Generator().manual_seed(mask_seed + index)) < missing_rate
            series.masked_fill_(missing, float("nan"))
        batch["series"] = series[:, -length:]
        batch["series_mask"] = batch["series_mask"][:, -length:]
        yield keep_modalities(batch, names)


def persistence_mae(batches):
    error, count = 0.0, 0
    for batch in batches:
        series, target = batch["series"], batch["forecast_targets"]
        observed = torch.isfinite(series)
        positions = torch.arange(series.size(1))[None, :, None].expand_as(series)
        last = positions.masked_fill(~observed, -1).max(1).values
        clean = torch.where(observed, series, 0)
        anchor = clean.gather(1, last.clamp_min(0).unsqueeze(1))
        valid = torch.isfinite(target)
        error += torch.where(valid, (anchor - target).abs(), 0).double().sum().item()
        count += valid.sum().item()
    return error / count if count else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--control-checkpoint", help="Même budget de fine-tuning sans augmentation")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9000000)
    parser.add_argument("--mask-seed", type=int, default=9100000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--output", default="runs/robustness-benchmark.json")
    args = parser.parse_args()
    if min(args.batches, args.batch_size, args.threads) < 1 or min(args.seed, args.mask_seed) < 0:
        parser.error("Tailles et threads > 0 ; seeds >= 0 requis")
    checkpoints = dict(source=args.reference_checkpoint)
    if args.control_checkpoint:
        checkpoints["control"] = args.control_checkpoint
    checkpoints["augmented"] = args.checkpoint
    if any(Path(args.output).resolve() == Path(path).resolve() for path in checkpoints.values()):
        parser.error("La sortie doit être distincte des checkpoints")
    torch.set_num_threads(args.threads)
    reference_config, reports, baselines = None, {}, {}
    for label, path in checkpoints.items():
        model, payload = load_checkpoint(path)
        if reference_config is not None and model.config.to_dict() != reference_config:
            raise ValueError("Les checkpoints doivent utiliser la même configuration")
        reference_config = model.config.to_dict()
        entry = dict(checkpoint=path, step=payload["step"], trainer=payload.get("trainer", {}), cases={})
        del payload
        for case in args.cases:
            entry["cases"][case] = {}
            for modality, names in (("series", ("series",)), ("all", ("input_ids", "images", "series"))):
                def batches():
                    return stress_batches(model.config, case=case, names=names, batches=args.batches,
                                          batch_size=args.batch_size, seed=args.seed, mask_seed=args.mask_seed)
                metrics = evaluate(model, batches())
                entry["cases"][case][modality] = metrics
                if label == "source" and modality == "series":
                    baselines[case] = persistence_mae(batches())
                print(json.dumps(dict(checkpoint=label, case=case, modalities=modality,
                                      mae=metrics["forecast_mae"], accuracy=metrics["accuracy"])), flush=True)
        reports[label] = entry
        del model
        gc.collect()
    report = dict(settings=vars(args), examples_per_case=args.batches * args.batch_size,
                  torch_version=torch.__version__, persistence_forecast_mae=baselines, checkpoints=reports,
                  scope="Synthetic held-out examples; masks independent of values. Does not establish real-world robustness or performance under informative missingness.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
