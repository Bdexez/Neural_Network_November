"""Construire des jeux chronologiques à fenêtres strictement séparées."""

from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from .data import file_identity


SPLITS = ("train", "validation", "calibration", "test")


def prepare_temporal_splits(source, output_dir, config, *, history, train_end, validation_end, calibration_end,
                            stride=1, gap=0):
    """Écrire quatre JSONL et un manifeste dans un nouveau dossier réservé exclusivement."""
    for name, value, minimum in (("history", history, 1), ("train_end", train_end, 1),
                                 ("validation_end", validation_end, 1), ("calibration_end", calibration_end, 1),
                                 ("stride", stride, 1), ("gap", gap, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} doit être un entier >= {minimum}")
    if history > config.max_series_len:
        raise ValueError("history dépasse max_series_len")
    if not train_end < validation_end < calibration_end:
        raise ValueError("train_end < validation_end < calibration_end requis")
    window = history + config.forecast_horizon
    if (train_end < window or validation_end - train_end - gap < window
            or calibration_end - validation_end - gap < window):
        raise ValueError("Les périodes d'entraînement, validation et calibration doivent contenir une fenêtre")
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"Le dossier de sortie existe déjà : {output_dir}")
    identity = file_identity(source.stat())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    records = {name: 0 for name in SPLITS}
    skipped = {name: 0 for name in SPLITS}
    hashes = {name: hashlib.sha256() for name in SPLITS}
    source_hash = hashlib.sha256()
    groups = set()
    limit = torch.finfo(torch.float32).max
    with tempfile.TemporaryDirectory(prefix=".temporal-splits-", dir=output_dir.parent) as temp:
        temporary = Path(temp)
        with ExitStack() as stack:
            source_stream = stack.enter_context(source.open("rb"))
            if file_identity(os.fstat(source_stream.fileno())) != identity:
                raise ValueError("Le fichier source a changé pendant sa lecture")
            streams = {name: stack.enter_context((temporary / f"{name}.jsonl").open("wb")) for name in SPLITS}
            for line_number, line in enumerate(source_stream, 1):
                source_hash.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("un objet JSON est requis")
                    group = record.get("series_id")
                    if not ((isinstance(group, str) and group) or type(group) is int):
                        raise ValueError("series_id doit être une chaîne non vide ou un entier")
                    key = (type(group), group)
                    if key in groups:
                        raise ValueError("series_id doit être unique dans le fichier source")
                    groups.add(key)
                    values = record.get("values")
                    if not isinstance(values, list) or not values:
                        raise ValueError("values doit être un tableau temporel non vide")
                    for row in values:
                        if not isinstance(row, list) or len(row) != config.series_features:
                            raise ValueError(f"chaque pas doit contenir {config.series_features} variables")
                        for value in row:
                            if value is None:
                                continue
                            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
                                raise ValueError("valeurs finies représentables en FP32 ou null requises")
                    if len(values) - calibration_end - gap < window:
                        raise ValueError("la période de test doit contenir une fenêtre après la coupure")
                    zones = {"train": (0, train_end), "validation": (train_end + gap, validation_end),
                             "calibration": (validation_end + gap, calibration_end),
                             "test": (calibration_end + gap, len(values))}
                    for name, (begin, end) in zones.items():
                        group_count = 0
                        for start in range(begin, end - window + 1, stride):
                            pivot = start + history
                            future = values[pivot:pivot + config.forecast_horizon]
                            if not any(value is not None for row in future for value in row):
                                skipped[name] += 1
                                continue
                            sample = dict(series=values[start:pivot], future=future, series_id=group,
                                          window_start=start, forecast_start=pivot, window_end=pivot + config.forecast_horizon)
                            encoded = (json.dumps(sample, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                                       + "\n").encode("ascii")
                            streams[name].write(encoded)
                            hashes[name].update(encoded)
                            records[name] += 1
                            group_count += 1
                        if not group_count:
                            raise ValueError(f"aucune fenêtre supervisée pour {name}")
                except (ValueError, TypeError, OverflowError, UnicodeDecodeError) as exc:
                    raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if (file_identity(os.fstat(source_stream.fileno())) != identity
                    or file_identity(source.stat()) != identity):
                raise ValueError("Le fichier source a changé pendant sa lecture")
        if not groups:
            raise ValueError("Le fichier source ne contient aucune série")
        manifest = dict(format_version=1, source=str(source), source_sha256=source_hash.hexdigest(),
                        series_count=len(groups), series_features=config.series_features,
                        forecast_horizon=config.forecast_horizon, history=history, stride=stride, gap=gap,
                        train_end=train_end, validation_end=validation_end, calibration_end=calibration_end,
                        splits={name: dict(file=f"{name}.jsonl", examples=records[name],
                                           skipped_unobserved_target=skipped[name], sha256=hashes[name].hexdigest())
                                for name in SPLITS},
                        scope="Indices de positions supposées chronologiques ; aucun recouvrement de fenêtre entre jeux.")
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Le dossier est réservé sans remplacer un dossier existant. Les liens
        # sont créés sans écrasement ; le manifeste est publié en dernier.
        output_dir.mkdir()
        published = []
        try:
            for name in (*(f"{split}.jsonl" for split in SPLITS), "manifest.json"):
                os.link(temporary / name, output_dir / name)
                published.append(name)
        except BaseException:
            for name in published:
                target = output_dir / name
                try:
                    if os.path.samefile(temporary / name, target):
                        target.unlink()
                except FileNotFoundError:
                    pass
            try:
                output_dir.rmdir()
            except OSError:
                pass
            raise
    return manifest
