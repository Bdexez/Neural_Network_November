"""Écrire un exemple de jeu JSONL multimodal et sa validation indépendante."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chimera import ByteTokenizer, ChimeraConfig
from chimera.data import synthetic_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--config")
    parser.add_argument("--output", default="runs/data")
    parser.add_argument("--train-examples", type=int, default=32)
    parser.add_argument("--validation-examples", type=int, default=16)
    args = parser.parse_args()
    if min(args.train_examples, args.validation_examples) < 1:
        parser.error("Les nombres d'exemples doivent être > 0")
    config = ChimeraConfig.load(args.config) if args.config else ChimeraConfig.preset(args.preset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, count, offset in (("train.jsonl", args.train_examples, 0),
                                    ("validation.jsonl", args.validation_examples, 1000000)):
        batch = synthetic_batch(config, batch_size=count, seed=42 + offset)
        with (output / filename).open("w", encoding="utf-8") as stream:
            for i in range(count):
                record = {
                    "text": ByteTokenizer().decode(batch["input_ids"][i]),
                    "image": batch["images"][i].tolist(),
                    "series": batch["series"][i].tolist(),
                    "class_label": batch["class_labels"][i].item(),
                    "future": batch["forecast_targets"][i].tolist(),
                }
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"{output / filename}: {count} exemples")


if __name__ == "__main__":
    main()
