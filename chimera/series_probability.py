"""Calibration par température d'une tête de classification temporelle."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .calibration import file_sha256, modality_signature


def _center(scores):
    scores = scores.double()
    if scores.ndim != 2 or scores.size(0) < 1 or scores.size(1) < 1 or not torch.isfinite(scores).all():
        raise ValueError("Scores de classification finis [B, C] requis")
    centered = scores - scores.amax(-1, keepdim=True)
    if not torch.isfinite(centered).all():
        raise ValueError("Amplitude des scores incompatible avec la précision FP64")
    return centered


def log_probabilities(scores, inverse_temperature=1.):
    if not math.isfinite(inverse_temperature) or inverse_temperature <= 0:
        raise ValueError("Température inverse finie et positive requise")
    result = F.log_softmax(_center(scores) * inverse_temperature, dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("Probabilités incompatibles avec la précision FP64")
    return result


@dataclass
class SeriesProbabilityCalibration:
    inverse_temperature: float
    num_classes: int
    amp: str
    checkpoint_sha256: str
    classifier_sha256: str
    dataset_sha256: str | None
    examples: int
    nll_before: float
    nll_after: float
    min_inverse_temperature: float = 1e-4
    max_inverse_temperature: float = 1e4

    def __post_init__(self):
        bounds = (self.min_inverse_temperature, self.inverse_temperature, self.max_inverse_temperature)
        if (not all(math.isfinite(value) for value in bounds)
                or not 0 < bounds[0] <= bounds[1] <= bounds[2] or not bounds[0] <= 1 <= bounds[2]):
            raise ValueError("Température inverse et bornes finies, positives et cohérentes requises")
        if type(self.num_classes) is not int or self.num_classes < 2:
            raise ValueError("Au moins deux classes sont requises")
        if type(self.examples) is not int or self.examples < 1:
            raise ValueError("Au moins un exemple étiqueté est requis")
        if self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("amp doit être none, bf16 ou fp16")
        if not all(math.isfinite(value) and value >= 0 for value in (self.nll_before, self.nll_after)):
            raise ValueError("NLL de calibration finies et positives requises")
        for digest in (self.checkpoint_sha256, self.classifier_sha256, self.dataset_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")
        if self.checkpoint_sha256 is None or self.classifier_sha256 is None:
            raise ValueError("Les empreintes du checkpoint et de la tête sont requises")

    def validate(self, classifier, *, amp=None):
        if classifier.num_classes != self.num_classes:
            raise ValueError("La calibration de probabilités est incompatible avec le nombre de classes")
        if classifier.checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("La calibration de probabilités est incompatible avec le checkpoint")
        if classifier.amp != self.amp or amp is not None and amp != self.amp:
            raise ValueError("La calibration de probabilités nécessite la même précision AMP")

    def validate_files(self, checkpoint, classifier_path):
        if file_sha256(checkpoint) != self.checkpoint_sha256 or file_sha256(classifier_path) != self.classifier_sha256:
            raise ValueError("La calibration de probabilités ne correspond pas au checkpoint ou à la tête")

    def log_probabilities(self, scores):
        if scores.ndim != 2 or scores.size(-1) != self.num_classes:
            raise ValueError("Scores incompatibles avec le nombre de classes")
        return log_probabilities(scores, self.inverse_temperature)

    def probabilities(self, scores):
        return self.log_probabilities(scores).exp()

    def to_dict(self):
        return dict(format_version=1, method="temperature", inverse_temperature=self.inverse_temperature,
                    num_classes=self.num_classes, amp=self.amp, checkpoint_sha256=self.checkpoint_sha256,
                    classifier_sha256=self.classifier_sha256, dataset_sha256=self.dataset_sha256,
                    examples=self.examples, nll_before=self.nll_before, nll_after=self.nll_after,
                    min_inverse_temperature=self.min_inverse_temperature,
                    max_inverse_temperature=self.max_inverse_temperature)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop("format_version", None) != 1 or values.pop("method", None) != "temperature":
            raise ValueError("Format de calibration de probabilités inconnu")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_series_probability(model, classifier, batches, *, checkpoint_sha256, classifier_sha256,
                           dataset_sha256=None, amp="none", min_inverse_temperature=1e-4,
                           max_inverse_temperature=1e4):
    """Minimiser la NLL sur des scores réservés, par dichotomie convexe en température inverse."""
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    classifier.validate(model.config, amp=amp)
    if classifier.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("La tête ne correspond pas au checkpoint")
    if classifier.dataset_sha256 is not None and classifier.dataset_sha256 == dataset_sha256:
        raise ValueError("Le jeu de calibration est identique au jeu d'ajustement de la tête")
    bounds = (min_inverse_temperature, max_inverse_temperature)
    if not all(math.isfinite(value) for value in bounds) or not 0 < bounds[0] <= 1 <= bounds[1]:
        raise ValueError("Les bornes de température inverse doivent être finies, positives et contenir 1")
    if model.config.num_classes < 2:
        raise ValueError("Au moins deux classes sont requises")
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16))
    was_training = model.training
    model.eval()
    scores, labels = [], []
    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                if modality_signature(batch) != ("series",):
                    raise ValueError("La calibration nécessite uniquement la modalité series")
                if "class_labels" not in batch:
                    continue
                target = batch["class_labels"]
                if (target.dtype != torch.long or target.ndim != 1 or target.size(0) != batch["series"].size(0)
                        or ((target != -100) & ((target < 0) | (target >= model.config.num_classes))).any()):
                    raise ValueError("class_labels doit contenir des classes valides ou -100")
                observed = target != -100
                if not observed.any():
                    continue
                with autocast():
                    value = classifier.scores(model, batch)
                scores.append(value[observed].double().cpu())
                labels.append(target[observed].cpu())
        if not scores:
            raise ValueError("La calibration nécessite des exemples étiquetés")
        values = _center(torch.cat(scores))
        target = torch.cat(labels)
        truth = values[torch.arange(target.size(0)), target]

        def derivative(beta):
            value = ((torch.softmax(values * beta, dim=-1) * values).sum(-1) - truth).mean().item()
            if not math.isfinite(value):
                raise ValueError("Gradient non fini pendant la calibration")
            return value

        def nll(beta):
            value = (torch.logsumexp(values * beta, dim=-1) - beta * truth).mean().item()
            if not math.isfinite(value):
                raise ValueError("NLL non finie pendant la calibration")
            return value

        before = nll(1.)
        lower, upper = bounds
        gradient_one = derivative(1.)
        if gradient_one == 0:
            beta = 1.
        elif derivative(lower) >= 0:
            beta = lower
        elif derivative(upper) <= 0:
            beta = upper
        else:
            for _ in range(64):
                middle = (lower + upper) / 2
                if derivative(middle) < 0:
                    lower = middle
                else:
                    upper = middle
            beta = (lower + upper) / 2
        after = nll(beta)
        if after > before:
            beta, after = 1., before
        return SeriesProbabilityCalibration(beta, model.config.num_classes, amp, checkpoint_sha256,
                                            classifier_sha256, dataset_sha256, target.numel(), before, after,
                                            min_inverse_temperature, max_inverse_temperature)
    finally:
        model.train(was_training)
