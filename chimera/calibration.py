"""Calibration CQR marginale ou simultanée sur une trajectoire complète.

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


def validate_score_scale(score_scale, scale_floor):
    if score_scale not in ("none", "width"):
        raise ValueError("score_scale doit être none ou width")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor doit être fini et strictement positif")
    if score_scale == "none" and scale_floor != 1e-6:
        raise ValueError("scale_floor personnalisé nécessite score_scale width")


def score_scale_values(prediction, score_scale, scale_floor):
    if score_scale == "none":
        return 1.0
    width = (prediction[..., -1] - prediction[..., 0]).clamp_min(scale_floor)
    if not torch.isfinite(width).all():
        raise ValueError("Largeur d'intervalle non finie")
    return width


def conformity_scores(forecast, targets, *, score_scale="none", scale_floor=1e-6):
    validate_score_scale(score_scale, scale_floor)
    if forecast.ndim != 4 or forecast.shape[:-1] != targets.shape or forecast.size(-1) < 2:
        raise ValueError("Prévisions [N, H, F, Q>=2] et cibles [N, H, F] attendues")
    if not torch.isfinite(forecast).all() or (forecast[..., 1:] < forecast[..., :-1]).any():
        raise ValueError("Les quantiles prédits doivent être finis et ordonnés")
    prediction, target = forecast.double(), targets.double()
    scores = torch.maximum(prediction[..., 0] - target, target - prediction[..., -1])
    scores = scores / score_scale_values(prediction, score_scale, scale_floor)
    return scores.masked_fill(~torch.isfinite(target), float("nan"))


def tail_conformity_scores(forecast, targets, *, score_scale="none", scale_floor=1e-6):
    """Scores séparés des bornes basse et haute, dans cet ordre."""
    validate_score_scale(score_scale, scale_floor)
    if forecast.ndim != 4 or forecast.shape[:-1] != targets.shape or forecast.size(-1) < 2:
        raise ValueError("Prévisions [N, H, F, Q>=2] et cibles [N, H, F] attendues")
    if not torch.isfinite(forecast).all() or (forecast[..., 1:] < forecast[..., :-1]).any():
        raise ValueError("Les quantiles prédits doivent être finis et ordonnés")
    prediction, target = forecast.double(), targets.double()
    scores = torch.stack((prediction[..., 0] - target, target - prediction[..., -1]), dim=-1)
    scale = score_scale_values(prediction, score_scale, scale_floor)
    if isinstance(scale, torch.Tensor):
        scale = scale.unsqueeze(-1)
    return (scores / scale).masked_fill(~torch.isfinite(target).unsqueeze(-1), float("nan"))


@dataclass
class ForecastCalibration:
    quantiles: tuple[float, ...]
    coverage: float
    correction: torch.Tensor
    counts: torch.Tensor
    modalities: tuple[str, ...]
    checkpoint_sha256: str | None = None
    dataset_sha256: str | None = None
    amp: str = "none"
    scope: str = "marginal"
    score_scale: str = "none"
    scale_floor: float = 1e-6
    dataset_images_sha256: str | None = None
    dataset_images_recorded: bool = False
    upper_correction: torch.Tensor | None = None
    lower_tail_fraction: float | None = None

    def __post_init__(self):
        self.quantiles, self.modalities = tuple(self.quantiles), tuple(self.modalities)
        if (len(self.quantiles) < 2 or 0.5 not in self.quantiles
                or any(not 0 < q < 1 for q in self.quantiles)
                or tuple(sorted(set(self.quantiles))) != self.quantiles):
            raise ValueError("Des quantiles ordonnés contenant la médiane sont requis")
        if not math.isfinite(self.coverage) or not 0 < self.coverage < 1:
            raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp doit être none, bf16 ou fp16")
        if self.scope not in ("marginal", "joint"):
            raise ValueError("scope doit être marginal ou joint")
        validate_score_scale(self.score_scale, self.scale_floor)
        if self.score_scale != "none" and self.scope != "joint":
            raise ValueError("score_scale width nécessite scope joint")
        self.correction = torch.as_tensor(self.correction, dtype=torch.float64).detach().cpu()
        counts = torch.as_tensor(self.counts).detach().cpu()
        if (self.correction.ndim != 2 or not all(self.correction.shape)
                or counts.shape != self.correction.shape or not torch.isfinite(self.correction).all()
                or not torch.isfinite(counts).all() or (counts < 1).any()
                or (counts != counts.floor()).any()):
            raise ValueError("Corrections finies [H, F] et effectifs entiers positifs requis")
        self.counts = counts.long()
        if (self.upper_correction is None) != (self.lower_tail_fraction is None):
            raise ValueError("Une calibration asymétrique nécessite les deux corrections et la répartition des queues")
        if self.lower_tail_fraction is not None:
            if not math.isfinite(self.lower_tail_fraction) or not 0 < self.lower_tail_fraction < 1:
                raise ValueError("lower_tail_fraction doit appartenir à ]0, 1[")
            self.upper_correction = torch.as_tensor(self.upper_correction, dtype=torch.float64).detach().cpu()
            if self.upper_correction.shape != self.correction.shape or not torch.isfinite(self.upper_correction).all():
                raise ValueError("Correction haute finie [H, F] requise")
        if self.scope == "joint" and ((self.correction != self.correction[0, 0]).any()
                                      or (self.counts != self.counts[0, 0]).any()
                                      or (self.upper_correction is not None
                                          and (self.upper_correction != self.upper_correction[0, 0]).any())):
            raise ValueError("La calibration jointe nécessite une correction et un effectif communs")
        levels = ((self.coverage,) if self.lower_tail_fraction is None else
                  (1 - (1 - self.coverage) * self.lower_tail_fraction,
                   1 - (1 - self.coverage) * (1 - self.lower_tail_fraction)))
        if any(math.ceil((int(n) + 1) * level) > int(n) for n in self.counts.flatten() for level in levels):
            raise ValueError("Effectif insuffisant pour une correction conformelle finie")
        if not self.modalities or self.modalities != tuple(name for name in MODALITIES if name in self.modalities):
            raise ValueError("La signature de modalités est invalide")
        if type(self.dataset_images_recorded) is not bool or (
                self.dataset_images_sha256 is not None and not self.dataset_images_recorded):
            raise ValueError("La provenance des images doit être explicitement enregistrée")
        for digest in (self.checkpoint_sha256, self.dataset_sha256, self.dataset_images_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")

    @classmethod
    def from_scores(cls, scores, quantiles, modalities, *, coverage=None, scope="marginal",
                    score_scale="none", scale_floor=1e-6, **provenance):
        """Ajuster à partir de scores déjà calculés avec l'échelle indiquée."""
        scores = torch.as_tensor(scores, dtype=torch.float64).detach().cpu()
        if scores.ndim != 3 or not all(scores.shape) or torch.isinf(scores).any():
            raise ValueError("Scores [N, H, F] requis ; seuls les NaN représentent des cibles absentes")
        quantiles = tuple(quantiles)
        if len(quantiles) < 2:
            raise ValueError("Au moins deux quantiles sont requis")
        coverage = quantiles[-1] - quantiles[0] if coverage is None else coverage
        if not math.isfinite(coverage) or not 0 < coverage < 1:
            raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
        if scope not in ("marginal", "joint"):
            raise ValueError("scope doit être marginal ou joint")
        validate_score_scale(score_scale, scale_floor)
        if score_scale != "none" and scope != "joint":
            raise ValueError("score_scale width nécessite scope joint")
        if scope == "joint":
            complete = torch.isfinite(scores).flatten(1).all(-1)
            count = complete.sum().item()
            rank = math.ceil((count + 1) * coverage)
            if rank > count:
                raise ValueError(f"Calibration jointe insuffisante : trajectoires complètes n={count}")
            values = scores[complete].flatten(1).amax(-1)
            correction = values.kthvalue(rank).values.expand(scores.shape[1:]).clone()
            counts = torch.full(scores.shape[1:], count, dtype=torch.long)
            return cls(quantiles, coverage, correction, counts, tuple(modalities), scope=scope,
                       score_scale=score_scale, scale_floor=scale_floor, **provenance)
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
        return cls(quantiles, coverage, correction, counts, tuple(modalities), scope=scope,
                   score_scale=score_scale, scale_floor=scale_floor, **provenance)

    @classmethod
    def from_tail_scores(cls, scores, quantiles, modalities, *, coverage=None, scope="marginal",
                         score_scale="none", scale_floor=1e-6, lower_tail_fraction=0.5, **provenance):
        """Ajuster séparément les scores bas et hauts avec des budgets fixés à l'avance."""
        scores = torch.as_tensor(scores, dtype=torch.float64).detach().cpu()
        if scores.ndim != 4 or scores.size(-1) != 2 or not all(scores.shape) or torch.isinf(scores).any():
            raise ValueError("Scores [N, H, F, 2] requis ; seuls les NaN représentent des cibles absentes")
        if not torch.equal(torch.isnan(scores[..., 0]), torch.isnan(scores[..., 1])):
            raise ValueError("Les deux scores doivent partager les mêmes cibles observées")
        quantiles = tuple(quantiles)
        if len(quantiles) < 2:
            raise ValueError("Au moins deux quantiles sont requis")
        coverage = quantiles[-1] - quantiles[0] if coverage is None else coverage
        if not math.isfinite(coverage) or not 0 < coverage < 1:
            raise ValueError("coverage doit être finie et appartenir à ]0, 1[")
        if not math.isfinite(lower_tail_fraction) or not 0 < lower_tail_fraction < 1:
            raise ValueError("lower_tail_fraction doit appartenir à ]0, 1[")
        if scope not in ("marginal", "joint"):
            raise ValueError("scope doit être marginal ou joint")
        validate_score_scale(score_scale, scale_floor)
        if score_scale != "none" and scope != "joint":
            raise ValueError("score_scale width nécessite scope joint")
        levels = (1 - (1 - coverage) * lower_tail_fraction,
                  1 - (1 - coverage) * (1 - lower_tail_fraction))
        if scope == "joint":
            complete = torch.isfinite(scores).flatten(1).all(-1)
            count = complete.sum().item()
            ranks = [math.ceil((count + 1) * level) for level in levels]
            if any(rank > count for rank in ranks):
                raise ValueError(f"Calibration asymétrique jointe insuffisante : trajectoires complètes n={count}")
            maxima = scores[complete].flatten(1, 2).amax(1)
            corrections = [maxima[:, side].kthvalue(rank).values.expand(scores.shape[1:3]).clone()
                           for side, rank in enumerate(ranks)]
            counts = torch.full(scores.shape[1:3], count, dtype=torch.long)
        else:
            counts = torch.isfinite(scores[..., 0]).sum(0)
            corrections = [torch.empty(scores.shape[1:3], dtype=torch.float64) for _ in range(2)]
            for horizon in range(scores.size(1)):
                for feature in range(scores.size(2)):
                    values = scores[:, horizon, feature]
                    values = values[torch.isfinite(values).all(-1)]
                    ranks = [math.ceil((len(values) + 1) * level) for level in levels]
                    if any(rank > len(values) for rank in ranks):
                        raise ValueError(f"Calibration asymétrique insuffisante : horizon {horizon}, variable {feature}, n={len(values)}")
                    for side, rank in enumerate(ranks):
                        corrections[side][horizon, feature] = values[:, side].kthvalue(rank).values
        return cls(quantiles, coverage, corrections[0], counts, tuple(modalities), scope=scope,
                   score_scale=score_scale, scale_floor=scale_floor, upper_correction=corrections[1],
                   lower_tail_fraction=lower_tail_fraction, **provenance)

    def validate(self, config, batch=None, *, amp=None):
        if (tuple(config.quantiles) != self.quantiles
                or self.correction.shape != (config.forecast_horizon, config.series_features)):
            raise ValueError("Calibration incompatible avec la configuration du modèle")
        if batch is not None and modality_signature(batch) != self.modalities:
            raise ValueError("La calibration nécessite les mêmes modalités qu'à son ajustement")
        if amp is not None and self.amp != amp:
            raise ValueError("La calibration nécessite la même précision AMP qu'à son ajustement")

    def validate_checkpoint(self, checkpoint):
        if self.checkpoint_sha256 is None or file_sha256(checkpoint) != self.checkpoint_sha256:
            raise ValueError("La calibration ne correspond pas à ce fichier de checkpoint")

    def interval(self, forecast):
        if forecast.ndim != 4 or forecast.shape[1:] != (*self.correction.shape, len(self.quantiles)):
            raise ValueError("La forme des prévisions est incompatible avec la calibration")
        if not torch.isfinite(forecast).all() or (forecast[..., 1:] < forecast[..., :-1]).any():
            raise ValueError("Les quantiles prédits doivent être finis et ordonnés")
        prediction = forecast if forecast.dtype in (torch.float32, torch.float64) else forecast.float()
        precise = prediction.double()
        scale = score_scale_values(precise, self.score_scale, self.scale_floor)
        lower_correction = self.correction.to(device=forecast.device) * scale
        upper_correction = (lower_correction if self.upper_correction is None else
                            self.upper_correction.to(device=forecast.device) * scale)
        median = precise[..., self.quantiles.index(0.5)]
        # Agrandir si nécessaire l'ensemble CQR pour garder une médiane incluse
        # et des bornes ordonnées, même avec une correction négative importante.
        lower = torch.minimum(precise[..., 0] - lower_correction, median)
        upper = torch.maximum(precise[..., -1] + upper_correction, median)
        if prediction.dtype == torch.float32:
            # Un arrondi vers l'intérieur pourrait exclure une cible située
            # exactement sur le bord de l'ensemble conformel.
            low32, high32 = lower.float(), upper.float()
            low32 = torch.where(low32.double() > lower, torch.nextafter(low32, torch.full_like(low32, -torch.inf)), low32)
            high32 = torch.where(high32.double() < upper, torch.nextafter(high32, torch.full_like(high32, torch.inf)), high32)
            lower, upper = low32, high32
        return torch.stack((lower, upper), dim=-1)

    def to_dict(self):
        data = dict(format_version=1, quantiles=list(self.quantiles), coverage=self.coverage,
                    correction=self.correction.tolist(), counts=self.counts.tolist(),
                    modalities=list(self.modalities), checkpoint_sha256=self.checkpoint_sha256,
                    dataset_sha256=self.dataset_sha256, amp=self.amp)
        if self.scope == "joint":
            data.update(format_version=2, scope="joint")
        if self.score_scale == "width":
            data.update(format_version=3, score_scale="width", scale_floor=self.scale_floor)
        if self.lower_tail_fraction is not None:
            data.update(format_version=4, scope=self.scope, score_scale=self.score_scale,
                        upper_correction=self.upper_correction.tolist(), lower_tail_fraction=self.lower_tail_fraction)
            if self.score_scale == "width":
                data["scale_floor"] = self.scale_floor
        if self.dataset_images_recorded:
            data["dataset_images_sha256"] = self.dataset_images_sha256
        return data

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        version = values.pop("format_version", None)
        values["dataset_images_recorded"] = "dataset_images_sha256" in values
        if version not in (1, 2, 3, 4):
            raise ValueError("Version de calibration inconnue")
        if (version == 1 and values.get("scope", "marginal") != "marginal"
                or version in (2, 3) and values.get("scope") != "joint"
                or version in (1, 2) and values.get("score_scale", "none") != "none"
                or version == 3 and (values.get("score_scale") != "width" or "scale_floor" not in values)
                or version in (1, 2, 3) and ("upper_correction" in values or "lower_tail_fraction" in values)
                or version == 4 and (values.get("scope") not in ("marginal", "joint")
                                      or values.get("score_scale") not in ("none", "width")
                                      or values.get("upper_correction") is None
                                      or values.get("lower_tail_fraction") is None
                                      or values.get("score_scale") == "width" and "scale_floor" not in values)):
            raise ValueError("Version et portée de calibration incompatibles")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_calibration(model, batches, *, coverage=None, amp="none", scope="marginal",
                    score_scale="none", scale_floor=1e-6, lower_tail_fraction=None, **provenance):
    """Conserver les seuls scores scalaires, sans stocker les entrées du réseau."""
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    if scope not in ("marginal", "joint"):
        raise ValueError("scope doit être marginal ou joint")
    validate_score_scale(score_scale, scale_floor)
    if score_scale != "none" and scope != "joint":
        raise ValueError("score_scale width nécessite scope joint")
    if lower_tail_fraction is not None and (not math.isfinite(lower_tail_fraction)
                                             or not 0 < lower_tail_fraction < 1):
        raise ValueError("lower_tail_fraction doit appartenir à ]0, 1[")
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
                score_fn = conformity_scores if lower_tail_fraction is None else tail_conformity_scores
                scores.append(score_fn(output.forecast, batch["forecast_targets"].to(device),
                                       score_scale=score_scale, scale_floor=scale_floor).cpu())
        if not scores:
            raise ValueError("Le jeu de calibration est vide")
        if lower_tail_fraction is not None:
            return ForecastCalibration.from_tail_scores(torch.cat(scores), quantiles, signature,
                                                        coverage=coverage, amp=amp, scope=scope,
                                                        score_scale=score_scale, scale_floor=scale_floor,
                                                        lower_tail_fraction=lower_tail_fraction, **provenance)
        return ForecastCalibration.from_scores(torch.cat(scores), quantiles, signature, coverage=coverage,
                                               amp=amp, scope=scope, score_scale=score_scale,
                                               scale_floor=scale_floor, **provenance)
    finally:
        model.train(was_training)
