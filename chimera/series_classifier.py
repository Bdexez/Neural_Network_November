"""Tête de classification temporelle sur historiques standardisés."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .augmentation import SeriesAugmentation
from .calibration import file_sha256, modality_signature
from .temporal_precision import temporal_work_dtype


def normalized_temporal_features(model, batch, scale_floor, shape_context=0):
    """Standardiser chaque variable sur les observations seules, puis moyenner l'encodeur gelé."""
    if model.training:
        raise ValueError("La classification temporelle nécessite model.eval()")
    if modality_signature(batch) != ("series",):
        raise ValueError("Le classifieur temporel nécessite uniquement la modalité series")
    series = batch["series"]
    if (series.ndim != 3 or not series.is_floating_point() or series.size(0) < 1
            or not 1 <= series.size(1) <= model.config.max_series_len
            or series.size(2) != model.config.series_features):
        raise ValueError("series doit être flottant [B, T, F] et compatible avec la configuration")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor doit être fini et strictement positif")
    if type(shape_context) is not int or not 0 <= shape_context <= model.config.max_series_len:
        raise ValueError("shape_context doit être un entier entre 0 et max_series_len")
    valid = batch.get("series_mask")
    if valid is None:
        valid = torch.ones(series.shape[:2], dtype=torch.bool, device=series.device)
    elif (valid.shape != series.shape[:2] or valid.device != series.device
          or not ((valid == 0) | (valid == 1)).all() or not valid.bool().any(-1).all()):
        raise ValueError("series_mask doit être binaire avec au moins un pas valide par exemple")
    else:
        valid = valid.bool()
    if series.device != next(model.parameters()).device:
        raise ValueError("series et modèle doivent être sur le même device")
    temporal_work_dtype(series, valid)  # Refuser les amplitudes incompatibles avec la variance FP64.
    values = series.double()
    observed = valid.unsqueeze(-1) & torch.isfinite(values)
    clean = torch.where(observed, values, 0)
    count = observed.sum(1, keepdim=True).clamp_min(1)
    mean = clean.sum(1, keepdim=True) / count
    centered = torch.where(observed, clean - mean, 0)
    scale = (centered.square().sum(1, keepdim=True) / count).sqrt().clamp_min(scale_floor)
    normalized = torch.where(observed, centered / scale, float("nan"))
    encoded, mask, *_ = model.temporal(normalized, valid)
    pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    if not torch.isfinite(pooled).all():
        raise ValueError("La représentation temporelle doit être finie")
    if shape_context:
        shape_rows = []
        for row in range(series.size(0)):
            current = centered[row, valid[row]] / scale[row]
            present = observed[row, valid[row]]
            occupancy = F.adaptive_avg_pool1d(present.T.double().unsqueeze(0), shape_context).squeeze(0)
            totals = F.adaptive_avg_pool1d(current.T.unsqueeze(0), shape_context).squeeze(0)
            bins = torch.where(occupancy > 0, totals / occupancy.clamp_min(1e-12), 0)
            moments = []
            for lag in (1, 2, 4):
                if current.size(0) <= lag:
                    moments.extend([current.new_zeros(series.size(-1)) for _ in range(3)])
                    continue
                pairs = present[lag:] & present[:-lag]
                count_pairs = pairs.sum(0).clamp_min(1)
                delta = torch.where(pairs, current[lag:] - current[:-lag], 0)
                product = torch.where(pairs, current[lag:] * current[:-lag], 0)
                moments.extend((delta.sum(0) / count_pairs, delta.square().sum(0) / count_pairs,
                                product.sum(0) / count_pairs))
            shape_rows.append(torch.cat((bins.flatten(), occupancy.flatten(), *moments)))
        pooled = torch.cat((pooled.double(), torch.stack(shape_rows)), dim=-1)
        if not torch.isfinite(pooled).all():
            raise ValueError("Les caractéristiques de forme doivent être finies")
    return pooled


@dataclass
class SeriesClassifier:
    weight: torch.Tensor
    bias: torch.Tensor
    series_features: int
    d_model: int
    num_classes: int
    scale_floor: float
    amp: str
    checkpoint_sha256: str | None = None
    dataset_sha256: str | None = None
    ridge: float = .001
    class_counts: tuple[int, ...] = ()
    shape_context: int = 0
    augmentation: dict | None = None
    augmentation_seed: int | None = None
    augmentation_batch_size: int | None = None
    augmentation_device: str | None = None

    def __post_init__(self):
        for name in ("series_features", "d_model", "num_classes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} doit être un entier positif")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0 or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("scale_floor et ridge doivent être finis et positifs")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp doit être none, bf16 ou fp16")
        if type(self.shape_context) is not int or self.shape_context < 0:
            raise ValueError("shape_context doit être un entier positif ou nul")
        if (self.augmentation is None) != (self.augmentation_seed is None):
            raise ValueError("L'augmentation et sa seed doivent être enregistrées ensemble")
        if self.augmentation is None and (self.augmentation_batch_size is not None or self.augmentation_device is not None):
            raise ValueError("La provenance d'augmentation nécessite une augmentation")
        if self.augmentation_batch_size is not None and (type(self.augmentation_batch_size) is not int
                                                          or self.augmentation_batch_size < 1):
            raise ValueError("augmentation_batch_size doit être un entier positif")
        if self.augmentation_device is not None and (not isinstance(self.augmentation_device, str)
                                                       or not self.augmentation_device):
            raise ValueError("augmentation_device doit être une chaîne non vide")
        if self.augmentation is not None:
            if not isinstance(self.augmentation, dict) or type(self.augmentation_seed) is not int or self.augmentation_seed < 0:
                raise ValueError("Paramètres d'augmentation invalides")
            if set(self.augmentation) != set(SeriesAugmentation().to_dict()):
                raise ValueError("Champs d'augmentation invalides")
            self.augmentation = SeriesAugmentation(**self.augmentation).to_dict()
        self.weight = torch.as_tensor(self.weight, dtype=torch.float64).detach().cpu()
        self.bias = torch.as_tensor(self.bias, dtype=torch.float64).detach().cpu()
        dimension = self.d_model + (self.series_features * (2 * self.shape_context + 9) if self.shape_context else 0)
        if (self.weight.shape != (dimension, self.num_classes) or self.bias.shape != (self.num_classes,)
                or not torch.isfinite(self.weight).all() or not torch.isfinite(self.bias).all()):
            raise ValueError("Poids et biais finis compatibles avec le classifieur requis")
        self.class_counts = tuple(self.class_counts)
        if (len(self.class_counts) != self.num_classes
                or any(type(count) is not int or count < 1 for count in self.class_counts)):
            raise ValueError("Un effectif entier positif est requis pour chaque classe")
        for digest in (self.checkpoint_sha256, self.dataset_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")

    def validate(self, config, *, amp=None):
        if (config.series_features, config.d_model, config.num_classes) != (
                self.series_features, self.d_model, self.num_classes):
            raise ValueError("Le classifieur temporel est incompatible avec la configuration")
        if amp is not None and amp != self.amp:
            raise ValueError("Le classifieur temporel nécessite la même précision AMP qu'à son ajustement")
        if self.shape_context > config.max_series_len:
            raise ValueError("shape_context dépasse max_series_len")

    def validate_checkpoint(self, checkpoint):
        if self.checkpoint_sha256 is None or file_sha256(checkpoint) != self.checkpoint_sha256:
            raise ValueError("Le classifieur temporel ne correspond pas à ce checkpoint")

    def scores(self, model, batch):
        self.validate(model.config)
        features = normalized_temporal_features(model, batch, self.scale_floor, self.shape_context).double()
        scores = features @ self.weight.to(features.device) + self.bias.to(features.device)
        if not torch.isfinite(scores).all():
            raise ValueError("Les scores de classification doivent être finis")
        return scores

    def to_dict(self):
        result = dict(format_version=1, method="ridge_one_vs_all", weight=self.weight.tolist(), bias=self.bias.tolist(),
                      series_features=self.series_features, d_model=self.d_model, num_classes=self.num_classes,
                      scale_floor=self.scale_floor, amp=self.amp, checkpoint_sha256=self.checkpoint_sha256,
                      dataset_sha256=self.dataset_sha256, ridge=self.ridge, class_counts=list(self.class_counts))
        if self.shape_context or self.augmentation is not None:
            result.update(format_version=2, shape_context=self.shape_context)
            if self.augmentation is not None:
                result.update(augmentation=self.augmentation, augmentation_seed=self.augmentation_seed)
                if self.augmentation_batch_size is not None:
                    result["augmentation_batch_size"] = self.augmentation_batch_size
                if self.augmentation_device is not None:
                    result["augmentation_device"] = self.augmentation_device
        return result

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        version = values.pop("format_version", None)
        if (version not in (1, 2) or values.pop("method", None) != "ridge_one_vs_all"
                or version == 1 and any(name in values for name in ("shape_context", "augmentation", "augmentation_seed",
                                                                     "augmentation_batch_size", "augmentation_device"))
                or version == 2 and "shape_context" not in values):
            raise ValueError("Format de classifieur temporel inconnu")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_series_classifier(model, batches, *, ridge=.001, scale_floor=1e-4, amp="none", shape_context=0,
                          augmentation=None, augmentation_seed=0, augmentation_batch_size=None, **provenance):
    """Ajuster une tête ridge multiclasse ; aucun poids du modèle n'est modifié."""
    if not math.isfinite(ridge) or ridge <= 0 or not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("ridge et scale_floor doivent être finis et positifs")
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    if type(shape_context) is not int or not 0 <= shape_context <= model.config.max_series_len:
        raise ValueError("shape_context doit être un entier entre 0 et max_series_len")
    if augmentation is not None and (not isinstance(augmentation, SeriesAugmentation)
                                     or type(augmentation_seed) is not int or augmentation_seed < 0):
        raise ValueError("Augmentation ou seed invalide")
    if augmentation_batch_size is not None and (augmentation is None or type(augmentation_batch_size) is not int
                                                 or augmentation_batch_size < 1):
        raise ValueError("augmentation_batch_size nécessite une augmentation et un entier positif")
    extra = model.config.series_features * (2 * shape_context + 9) if shape_context else 0
    dim, classes = model.config.d_model + extra + 1, model.config.num_classes
    gram = torch.zeros(dim, dim, dtype=torch.float64)
    cross = torch.zeros(dim, classes, dtype=torch.float64)
    counts = torch.zeros(classes, dtype=torch.long)
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16))
    was_training = model.training
    model.eval()
    previous_batch_size = None
    try:
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                batch = {key: value.to(device) for key, value in batch.items()}
                if modality_signature(batch) != ("series",):
                    raise ValueError("Le classifieur temporel nécessite uniquement la modalité series")
                size = batch["series"].size(0)
                if augmentation_batch_size is not None and (
                        size < 1 or size > augmentation_batch_size
                        or previous_batch_size is not None and previous_batch_size != augmentation_batch_size):
                    raise ValueError("La taille des lots ne correspond pas à augmentation_batch_size")
                previous_batch_size = size
                if "class_labels" not in batch:
                    continue
                if augmentation is not None:
                    batch = augmentation(batch, index=index, seed=augmentation_seed)
                labels = batch["class_labels"]
                if (labels.dtype != torch.long or labels.ndim != 1 or labels.size(0) != batch["series"].size(0)
                        or ((labels != -100) & ((labels < 0) | (labels >= classes))).any()):
                    raise ValueError("class_labels doit contenir des classes valides ou -100")
                with autocast():
                    features = normalized_temporal_features(model, batch, scale_floor, shape_context)
                observed = labels != -100
                x = features[observed].double().cpu()
                y = labels[observed].cpu()
                x = torch.cat((x, torch.ones(x.size(0), 1, dtype=torch.float64)), dim=-1)
                gram += x.T @ x
                cross += x.T @ F.one_hot(y, classes).double()
                counts += torch.bincount(y, minlength=classes)
        if (counts == 0).any():
            raise ValueError("Chaque classe doit avoir au moins un exemple étiqueté")
        n = counts.sum().item()
        penalty = torch.eye(dim, dtype=torch.float64) * ridge
        penalty[-1, -1] = 0
        coefficients = torch.linalg.solve(gram / n + penalty, cross / n)
        return SeriesClassifier(coefficients[:-1], coefficients[-1], model.config.series_features,
                                model.config.d_model, classes, scale_floor, amp, ridge=ridge,
                                class_counts=tuple(counts.tolist()), shape_context=shape_context,
                                augmentation=augmentation.to_dict() if augmentation is not None else None,
                                augmentation_seed=augmentation_seed if augmentation is not None else None,
                                augmentation_batch_size=augmentation_batch_size,
                                augmentation_device=str(device) if augmentation is not None else None, **provenance)
    finally:
        model.train(was_training)
