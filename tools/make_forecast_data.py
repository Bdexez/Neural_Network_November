"""Créer des historiques synthétiques raccourcis pour ajuster une correction."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from chimera.augmentation import SeriesAffineAugmentation, SeriesAugmentation
from chimera.cli import prediction_stream
from chimera.config import ChimeraConfig
from chimera.data import JointDataset, synthetic_batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--augmentation-seed", type=int, default=1050000)
    parser.add_argument("--min-history", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--mask-probability", type=float, default=0.5)
    parser.add_argument("--affine-probability", type=float, default=0.0)
    parser.add_argument("--min-scale", type=float, default=0.1)
    parser.add_argument("--max-scale", type=float, default=10.0)
    parser.add_argument("--max-shift", type=float, default=20.0)
    parser.add_argument("--affine-seed", type=int, default=2050000)
    args = parser.parse_args(argv)
    if min(args.examples, args.batch_size) < 1 or min(args.seed, args.augmentation_seed, args.affine_seed) < 0:
        parser.error("Tailles strictement positives et seeds positifs ou nuls requis")
    if Path(args.output).resolve() == Path(args.checkpoint).resolve():
        parser.error("La sortie doit être distincte du checkpoint")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = ChimeraConfig.from_dict(payload["config"])
    del payload
    augmentation = SeriesAugmentation(crop_probability=1, min_history=args.min_history,
                                      missing_rate=args.missing_rate, mask_probability=args.mask_probability)
    affine = SeriesAffineAugmentation(probability=args.affine_probability, min_scale=args.min_scale,
                                      max_scale=args.max_scale, max_shift=args.max_shift)
    lengths, missing, values = Counter(), 0, 0
    with prediction_stream(args.output) as stream:
        for index, start in enumerate(range(0, args.examples, args.batch_size)):
            batch = synthetic_batch(config, batch_size=min(args.batch_size, args.examples - start),
                                    seed=args.seed + index)
            batch = augmentation(batch, index=index, seed=args.augmentation_seed)
            batch = affine(batch, index=index, seed=args.affine_seed)
            for series, valid, future in zip(batch["series"], batch["series_mask"], batch["forecast_targets"]):
                series = series[valid]
                lengths[len(series)] += 1
                missing += (~torch.isfinite(series)).sum().item()
                values += series.numel()
                record = dict(series=[[value if math.isfinite(value) else None for value in row]
                                      for row in series.tolist()], future=future.tolist())
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(dict(settings=vars(args), dataset_sha256=JointDataset(args.output, config).fingerprint,
                          lengths=dict(sorted(lengths.items())), missing_values=missing, values=values), indent=2))


if __name__ == "__main__":
    main()
