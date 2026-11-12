"""Mesurer les prévisions après un changement d'unités ou de niveau des séries."""

import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.calibration import file_sha256
from chimera.cli import prediction_stream
from chimera.data import evaluation_data_signature
from chimera.evaluation import evaluate
from chimera.training import load_checkpoint
from tools.benchmark_robustness import CASES, persistence_mae, stress_batches


TRANSFORMS = dict(original=(1.0, 0.0), offset20=(1.0, 20.0), scale01=(0.1, 0.0),
                  scale10=(10.0, 0.0), scale4_offset50=(4.0, -50.0))
MODALITIES = dict(series=("series",), all=("input_ids", "images", "series"))


def affine_batches(config, *, gain, shift, **settings):
    for batch in stress_batches(config, **settings):
        batch["series"] = batch["series"] * gain + shift
        batch["forecast_targets"] = batch["forecast_targets"] * gain + shift
        yield batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--control-checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=["history8", "history16", "complete"])
    parser.add_argument("--transforms", nargs="+", choices=tuple(TRANSFORMS), default=list(TRANSFORMS))
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITIES), default=list(MODALITIES))
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23000000)
    parser.add_argument("--mask-seed", type=int, default=23100000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if min(args.batches, args.batch_size, args.threads) < 1 or min(args.seed, args.mask_seed) < 0:
        parser.error("Tailles et threads > 0 ; seeds >= 0 requis")
    paths = dict(source=args.reference_checkpoint)
    if args.control_checkpoint:
        paths["control"] = args.control_checkpoint
    paths["candidate"] = args.checkpoint
    if any(Path(args.output).resolve() == Path(path).resolve() for path in paths.values()):
        parser.error("La sortie doit être distincte des checkpoints")
    torch.set_num_threads(args.threads)
    reports, persistence, signature = {}, {}, None
    for label, path in paths.items():
        model, payload = load_checkpoint(path)
        current_signature = evaluation_data_signature(model.config)
        if signature is not None and current_signature != signature:
            raise ValueError("Les checkpoints doivent partager les mêmes entrées et cibles d'évaluation")
        signature = current_signature
        entry = dict(checkpoint=path, sha256=file_sha256(path), step=payload["step"],
                     config=model.config.to_dict(), fit=payload.get("trainer", {}).get("forecast_adapter_fit"), cases={})
        del payload
        for case in dict.fromkeys(args.cases):
            entry["cases"][case] = {}
            persistence.setdefault(case, {})
            for transform in dict.fromkeys(args.transforms):
                gain, shift = TRANSFORMS[transform]
                entry["cases"][case][transform] = {}
                for modality in dict.fromkeys(args.modalities):
                    def batches():
                        return affine_batches(model.config, gain=gain, shift=shift, case=case,
                                              names=MODALITIES[modality], batches=args.batches,
                                              batch_size=args.batch_size, seed=args.seed, mask_seed=args.mask_seed)
                    metrics = evaluate(model, batches())
                    # La division par le gain positif remet la MAE dans les unités
                    # des exemples d'origine, pour comparer les transformations.
                    metrics["original_unit_forecast_mae"] = metrics["forecast_mae"] / gain
                    entry["cases"][case][transform][modality] = metrics
                    if label == "source" and transform not in persistence[case]:
                        persistence[case][transform] = persistence_mae(batches()) / gain
                    print(json.dumps(dict(checkpoint=label, case=case, transform=transform, modalities=modality,
                                          mae=metrics["forecast_mae"], original_unit_mae=metrics["original_unit_forecast_mae"],
                                          accuracy=metrics["accuracy"])), flush=True)
        reports[label] = entry
        del model
        gc.collect()
    report = dict(settings=vars(args), examples_per_case=args.batches * args.batch_size,
                  transforms={name: dict(gain=TRANSFORMS[name][0], shift=TRANSFORMS[name][1]) for name in args.transforms},
                  checkpoints=reports, persistence_original_unit_mae=persistence, torch_version=str(torch.__version__),
                  scope="Paired affine transformations of synthetic held-out data. Original-unit MAE divides transformed-unit MAE by gain. Does not establish performance on real data or new dynamics.")
    with prediction_stream(args.output) as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
