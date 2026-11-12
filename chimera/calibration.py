"""Calibration CQR marginale, séparée par horizon et variable.

Référence : https://arxiv.org/abs/1905.03222. Les exemples de calibration
doivent être distincts de l'apprentissage et échangeables avec les futurs cas.
"""

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import torch

from .modalities import MODALITIES, TARGETS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def modality_signature(batch):
    return tuple(name for name in MODALITIES if name in batch)


def conformity_scores(forecast, targets):
    if forecast.ndim != 4 or forecast.shape[:-1] != targets.shape or forecast.size(-1) < 2:
        raise ValueError("Prévisions [N, H, F, Q>=2] et cibles [N, H, F] attendues")
    if not torch.isfinite(forecast).all() or (forecast[..., 1:] < forecast[..., :-1]).any():
        raise ValueError("Les quantiles prédits doivent être finis et ordonnés")
    prediction, target = forecast.double(), targets.double()
    scores = torch.maximum(prediction[..., 0] - target, target - prediction[..., -1])
    return scores.masked_fill(~torch.isfinite(target), float("nan"))


@dataclass
class ForecastCalibration:
    quantiles: tuple[float, ...]
    coverage: float
    correction: torch.Tensor
    counts: torch.Tensor
    modalities: tuple[str, ...]
    checkpoint_sha256: str | None = None
    dataset_sha256: str | None = None

    def __post_init__(self):
        self.quantiles, self.modalities = tuple(self.quantiles), tuple(self.modalities)
        if (len(self.quantiles) < 2 or 0.5 not in self.quantiles
                or any(not 0 < q < 1 for q in self.quantiles)
                or tuple(sorted(set(self.quantiles))) != self.quantiles):
            raise ValueError("Des quantiles ordonnés contenant la médiane sont requis")
        if not math.isfinite(self.coverage) or not 0 < self.coverage < 1:
            raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
        self.correction = torch.as_tensor(self.correction, dtype=torch.float64).detach().cpu()
        counts = torch.as_tensor(self.counts).detach().cpu()
        if (self.correction.ndim != 2 or not all(self.correction.shape)
                or counts.shape != self.correction.shape or not torch.isfinite(self.correction).all()
                or not torch.isfinite(counts).all() or (counts < 1).any()
                or (counts != counts.floor()).any()):
            raise ValueError("Corrections finies [H, F] et effectifs entiers positifs requis")
        self.counts = counts.long()
        if any(math.ceil((int(n) + 1) * self.coverage) > int(n) for n in self.counts.flatten()):
            raise ValueError("Effectif insuffisant pour une correction conformelle finie")
        if not self.modalities or self.modalities != tuple(name for name in MODALITIES if name in self.modalities):
            raise ValueError("La signature de modalités est invalide")
        for digest in (self.checkpoint_sha256, self.dataset_sha256):
            if digest is not None and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")

    @classmethod
    def from_scores(cls, scores, quantiles, modalities, *, coverage=None, **provenance):
        scores = torch.as_tensor(scores, dtype=torch.float64).detach().cpu()
        if scores.ndim != 3 or not all(scores.shape) or torch.isinf(scores).any():
            raise ValueError("Scores [N, H, F] requis ; seuls les NaN représentent des cibles absentes")
        quantiles = tuple(quantiles)
        if len(quantiles) < 2:
            raise ValueError("Au moins deux quantiles sont requis")
        coverage = quantiles[-1] - quantiles[0] if coverage is None else coverage
        if not math.isfinite(coverage) or not 0 < coverage < 1:
            raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
        correction = torch.empty(scores.shape[1:], dtype=torch.float64)
        counts = torch.isfinite(scores).sum(0)
        for horizon in range(scores.size(1)):
            for feature in range(scores.size(2)):
                values = scores[:, horizon, feature]
                values = values[torch.isfinite(values)]
                rank = math.ceil((len(values) + 1) * coverage)
                if rank > len(values):
                    raise ValueError(f"Calibration insuffisante : horizon {horizon}, variable {feature}, n={len(values)}")
                # Statistique d'ordre exacte ; pas d'interpolation ni de rang tronqué.
                correction[horizon, feature] = values.kthvalue(rank).values
        return cls(quantiles, coverage, correction, counts, tuple(modalities), **provenance)

    def validate(self, config, batch=None):
        if (tuple(config.quantiles) != self.quantiles
                or self.correction.shape != (config.forecast_horizon, config.series_features)):
            raise ValueError("Calibration incompatible avec la configuration du modèle")
        if batch is not None and modality_signature(batch) != self.modalities:
            raise ValueError("La calibration nécessite les mêmes modalités qu'à son ajustement")

    def validate_checkpoint(self, checkpoint):
        if self.checkpoint_sha256 is None or file_sha256(checkpoint) != self.checkpoint_sha256:
            raise ValueError("La calibration ne correspond pas à ce fichier de checkpoint")

    def interval(self, forecast):
        if forecast.ndim != 4 or forecast.shape[1:] != (*self.correction.shape, len(self.quantiles)):
            raise ValueError("La forme des prévisions est incompatible avec la calibration")
        if not torch.isfinite(forecast).all() or (forecast[..., 1:] < forecast[..., :-1]).any():
            raise ValueError("Les quantiles prédits doivent être finis et ordonnés")
        prediction = forecast if forecast.dtype in (torch.float32, torch.float64) else forecast.float()
        correction = self.correction.to(device=forecast.device, dtype=prediction.dtype)
        median = prediction[..., self.quantiles.index(0.5)]
        # Agrandir si nécessaire l'ensemble CQR pour garder une médiane incluse
        # et des bornes ordonnées, même avec une correction négative importante.
        lower = torch.minimum(prediction[..., 0] - correction, median)
        upper = torch.maximum(prediction[..., -1] + correction, median)
        return torch.stack((lower, upper), dim=-1)

    def to_dict(self):
        return dict(format_version=1, quantiles=list(self.quantiles), coverage=self.coverage,
                    correction=self.correction.tolist(), counts=self.counts.tolist(),
                    modalities=list(self.modalities), checkpoint_sha256=self.checkpoint_sha256,
                    dataset_sha256=self.dataset_sha256)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop("format_version", None) != 1:
            raise ValueError("Version de calibration inconnue")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_calibration(model, batches, *, coverage=None, amp="none", **provenance):
    """Conserver les seuls scores scalaires, sans stocker les entrées du réseau."""
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    quantiles = tuple(model.config.quantiles)
    if len(quantiles) < 2:
        raise ValueError("Au moins deux quantiles sont requis")
    coverage = quantiles[-1] - quantiles[0] if coverage is None else coverage
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )
    was_training = model.training
    model.eval()
    scores, signature = [], None
    try:
        with torch.inference_mode():
            for batch in batches:
                if "forecast_targets" not in batch:
                    raise ValueError("La calibration nécessite forecast_targets dans chaque lot")
                current = modality_signature(batch)
                if signature is not None and current != signature:
                    raise ValueError("Les modalités doivent rester identiques pendant la calibration")
                signature = current
                inputs = {key: value.to(device) for key, value in batch.items() if key not in TARGETS}
                with autocast():
                    output = model(**inputs)
                scores.append(conformity_scores(output.forecast, batch["forecast_targets"].to(device)).cpu())
        if not scores:
            raise ValueError("Le jeu de calibration est vide")
        return ForecastCalibration.from_scores(torch.cat(scores), quantiles, signature,
                                               coverage=coverage, **provenance)
    finally:
        model.train(was_training)
