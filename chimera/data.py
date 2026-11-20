from array import array
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


def evaluation_data_signature(config):
    """Champs qui définissent les entrées et les cibles des comparaisons synthétiques."""
    names = ("vocab_size", "max_seq_len", "image_size", "image_channels", "series_features",
             "max_series_len", "num_classes", "forecast_horizon", "quantiles")
    return {name: getattr(config, name) for name in names}


def numeric_tensor(values):
    def convert(value):
        if value is None:
            return float("nan")
        if isinstance(value, list):
            return [convert(element) for element in value]
        return value
    return torch.tensor(convert(values), dtype=torch.float32)


def file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class JointDataset(Dataset):
    """JSONL local : texte, image, historique et cibles, sans téléchargement implicite."""

    def __init__(self, path, config, *, allow_mixed=False, include_alignment_ids=False):
        self.path = Path(path).resolve()
        self.include_alignment_ids = include_alignment_ids
        self.config = config
        initial_identity = file_identity(self.path.stat())
        self.offsets, self.line_numbers = array("Q"), array("Q")
        self.signatures = array("B") if allow_mixed else None
        signature = None
        digest = hashlib.sha256()
        image_lines = {}
        with self.path.open("rb") as stream:
            if file_identity(os.fstat(stream.fileno())) != initial_identity:
                raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: JSON invalide") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{self.path}:{line_number}: un objet JSON est attendu")
                current = tuple(key in record or (key == "image" and "image_path" in record)
                                for key in ("text", "image", "series"))
                if not any(current):
                    raise ValueError(f"{self.path}:{line_number}: fournir text, image/image_path ou series")
                if not allow_mixed and signature is not None and current != signature:
                    raise ValueError(f"{self.path}:{line_number}: les modalités doivent être identiques dans le fichier")
                signature = current
                if self.signatures is not None:
                    self.signatures.append(sum(int(present) << index for index, present in enumerate(current)))
                if "image_path" in record and "image" not in record:
                    image_path = record["image_path"]
                    if not isinstance(image_path, str) or not image_path:
                        raise ValueError(f"{self.path}:{line_number}: image_path doit être une chaîne non vide")
                    image_lines.setdefault(image_path, line_number)
                self.offsets.append(offset)
                self.line_numbers.append(line_number)
            if (file_identity(os.fstat(stream.fileno())) != initial_identity
                    or file_identity(self.path.stat()) != initial_identity):
                raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")
        if not self.offsets:
            raise ValueError("Le fichier JSONL ne contient aucun exemple")
        self.fingerprint = digest.hexdigest()
        self._image_identities = {}
        image_digests = {}
        for name, line_number in image_lines.items():
            image_path = self.path.parent / name
            try:
                image_digest = hashlib.sha256()
                with image_path.open("rb") as image_stream:
                    before = file_identity(os.fstat(image_stream.fileno()))
                    for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
                        image_digest.update(chunk)
                    after = file_identity(os.fstat(image_stream.fileno()))
                if before != after or after != file_identity(image_path.stat()):
                    raise ValueError(f"L'image a changé pendant son indexation : {image_path}")
            except (OSError, ValueError) as exc:
                raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
            self._image_identities[name] = after
            image_digests[name] = image_digest.hexdigest()
        manifest = json.dumps(image_digests, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.image_fingerprint = hashlib.sha256(manifest.encode("ascii")).hexdigest() if image_digests else None
        self._identity = file_identity(self.path.stat())
        if self._identity != initial_identity:
            raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")

    def __len__(self):
        return len(self.offsets)

    def references_image(self, path):
        """Reconnaître aussi les liens physiques vers une image du jeu."""
        if path is None:
            return False
        try:
            identity = file_identity(Path(path).stat())[:2]
        except FileNotFoundError:
            return False
        return any(identity == known[:2] for known in self._image_identities.values())

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            if file_identity(os.fstat(stream.fileno())) != self._identity:
                raise ValueError(f"Le jeu de données a changé après son indexation : {self.path}")
            stream.seek(self.offsets[index])
            line = stream.readline()
            if (file_identity(os.fstat(stream.fileno())) != self._identity
                    or file_identity(self.path.stat()) != self._identity):
                raise ValueError(f"Le jeu de données a changé pendant sa lecture : {self.path}")
        record = json.loads(line)
        try:
            item = self._convert(record)
            if self.include_alignment_ids:
                source_index = index if index >= 0 else len(self) + index
                item['alignment_ids'] = torch.tensor(source_index, dtype=torch.long)
            return item
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            raise ValueError(f"{self.path}:{self.line_numbers[index]}: {exc}") from exc

    def _convert(self, record):
        item = {}
        if "text" in record:
            if not isinstance(record["text"], str):
                raise ValueError("text doit être une chaîne UTF-8")
            ids = ByteTokenizer().encode(record["text"])
            if len(ids) > self.config.max_seq_len:
                ids = ids[:self.config.max_seq_len]
                ids[-1] = ByteTokenizer.eos_id
            item["input_ids"] = torch.tensor(ids, dtype=torch.long)
        if "image" in record:
            item["images"] = torch.tensor(record["image"], dtype=torch.float32)
        elif "image_path" in record:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pour lire PNG/JPEG : pip install '.[vision]'") from exc
            path = self.path.parent / record["image_path"]
            if self.config.image_channels not in (1, 3):
                raise ValueError("image_path prend en charge image_channels=1 ou 3 ; utiliser image sinon")
            if file_identity(path.stat()) != self._image_identities[record["image_path"]]:
                raise ValueError(f"L'image a changé après son indexation : {path}")
            with Image.open(path) as source:
                mode = "RGB" if self.config.image_channels == 3 else "L"
                image = source.convert(mode).resize((self.config.image_size,) * 2)
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            if file_identity(path.stat()) != self._image_identities[record["image_path"]]:
                raise ValueError(f"L'image a changé pendant sa lecture : {path}")
            item["images"] = pixels.view(self.config.image_size, self.config.image_size, -1).permute(2, 0, 1).float() / 255
        if "series" in record:
            item["series"] = numeric_tensor(record["series"])
        if "class_label" in record:
            label = record["class_label"]
            if type(label) is not int or not (label == -100 or 0 <= label < self.config.num_classes):
                raise ValueError("class_label doit être une classe entière valide ou -100")
            item["class_labels"] = torch.tensor(record["class_label"], dtype=torch.long)
        if "future" in record:
            item["forecast_targets"] = numeric_tensor(record["future"])
        return item


def collate_records(items):
    if not items:
        raise ValueError("Le batch ne peut pas être vide")
    result = {}
    for name in ("input_ids", "images", "series"):
        present = [name in item for item in items]
        if any(present) and not all(present):
            raise ValueError(f"{name} doit être fourni pour tout le batch")
    identities = ["alignment_ids" in item for item in items]
    if any(identities) and not all(identities):
        raise ValueError("alignment_ids doit être fourni pour tout le batch")
    if all(identities):
        result["alignment_ids"] = torch.stack([item["alignment_ids"] for item in items])
    if "images" in items[0]:
        result["images"] = torch.stack([item["images"] for item in items])
    for name, missing in (("class_labels", -100), ("forecast_targets", float("nan"))):
        example = next((item[name] for item in items if name in item), None)
        if example is not None:
            result[name] = torch.stack([item[name] if name in item else torch.full_like(example, missing)
                                        for item in items])
    if "input_ids" in items[0]:
        length = max(item["input_ids"].size(0) for item in items)
        ids = torch.full((len(items), length), ByteTokenizer.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["input_ids"].size(0)
            ids[i, :n] = item["input_ids"]
            mask[i, :n] = True
        result.update(input_ids=ids, labels=ids.clone(), attention_mask=mask)
    if "series" in items[0]:
        length = max(item["series"].size(0) for item in items)
        features = items[0]["series"].size(-1)
        values = torch.full((len(items), length, features), float("nan"))
        mask = torch.zeros(len(items), length, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["series"].size(0)
            values[i, :n] = item["series"]
            mask[i, :n] = True
        result.update(series=values, series_mask=mask)
    return result


def synthetic_batch(config, *, batch_size=4, seed=0, device="cpu"):
    """Les modalités décrivent le même régime latent ; futur jamais donné en entrée."""
    gen = torch.Generator().manual_seed(seed)
    classes = torch.randint(config.num_classes, (batch_size,), generator=gen)
    length = min(config.max_series_len, 32)
    times = torch.arange(length + config.forecast_horizon).float()[None, :, None]
    phase = torch.rand(batch_size, 1, config.series_features, generator=gen) * 3
    frequency = 0.08 + classes[:, None, None].float() * 0.025
    slope = (classes[:, None, None].float() - config.num_classes / 2) * 0.015
    sequence = torch.sin(times * frequency + phase) + times * slope
    sequence = sequence + torch.randn(sequence.shape, generator=gen) * 0.015
    axis = torch.linspace(-1, 1, config.image_size)
    pattern = torch.sin((classes[:, None, None].float() + 1) * axis[None, None, :] * 3)
    image = pattern.expand(-1, config.image_size, -1)[:, None].expand(-1, config.image_channels, -1, -1)
    image = (image + 1) / 2 + torch.randn(image.shape, generator=gen) * 0.01
    tokenizer = ByteTokenizer()
    items = []
    for i in range(batch_size):
        text = f"Signal de classe {classes[i].item()}."
        tokens = tokenizer.encode(text)[:config.max_seq_len]
        items.append(dict(input_ids=torch.tensor(tokens), images=image[i], series=sequence[i, :length],
                          class_labels=classes[i], forecast_targets=sequence[i, length:]))
    return {key: value.to(device) for key, value in collate_records(items).items()}
