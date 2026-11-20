"""Ensembles conformes de classes pour une tête temporelle figée."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from .calibration import file_sha256, modality_signature
from .series_probability import log_probabilities


def class_scores(probabilities, method):
    p = probabilities.double()
    if (p.ndim != 2 or p.size(0) < 1 or p.size(1) < 2 or not torch.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()
            or not torch.allclose(p.sum(-1), torch.ones(p.size(0), device=p.device, dtype=p.dtype),
                                  atol=1e-6, rtol=0)):
        raise ValueError("Probabilités finies et normalisées [B, C] requises")
    if method == "lac":
        return 1 - p
    if method == "aps":
        order = torch.argsort(p, dim=-1, descending=True, stable=True)
        cumulative = p.gather(1, order).cumsum(-1).clamp(max=1)
        return torch.zeros_like(p).scatter(1, order, cumulative)
    raise ValueError("Méthode de score inconnue")


def conformal_threshold(scores, coverage):
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("La couverture doit être strictement comprise entre 0 et 1")
    if scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all():
        raise ValueError("Des scores de calibration finis sont requis")
    rank = math.ceil((scores.numel() + 1) * coverage)
    if rank > scores.numel():
        raise ValueError("Nombre d'exemples insuffisant pour cette couverture")
    return scores.kthvalue(rank).values.item()


@dataclass
class SeriesPredictionSet:
    coverage: float
    method: str
    threshold: float
    examples: int
    num_classes: int
    amp: str
    checkpoint_sha256: str
    classifier_sha256: str
    probability_sha256: str | None
    dataset_sha256: str | None

    def __post_init__(self):
        if not math.isfinite(self.coverage) or not 0 < self.coverage < 1:
            raise ValueError("Couverture invalide")
        if self.method not in ("lac", "aps") or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Méthode ou seuil invalide")
        if type(self.examples) is not int or self.examples < 1 or math.ceil((self.examples + 1) * self.coverage) > self.examples:
            raise ValueError("Nombre d'exemples insuffisant pour cette couverture")
        if type(self.num_classes) is not int or self.num_classes < 2 or self.amp not in ("none", "bf16", "fp16"):
            raise ValueError("Classes ou précision invalides")
        for digest in (self.checkpoint_sha256, self.classifier_sha256, self.probability_sha256, self.dataset_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Empreinte SHA-256 invalide")
        if self.checkpoint_sha256 is None or self.classifier_sha256 is None:
            raise ValueError("Empreintes du checkpoint et de la tête requises")

    def validate(self, classifier, probability=None, *, amp=None):
        if (classifier.num_classes != self.num_classes or classifier.checkpoint_sha256 != self.checkpoint_sha256
                or classifier.amp != self.amp or (amp is not None and amp != self.amp)):
            raise ValueError("Ensemble conforme incompatible avec la tête ou la précision")
        if (probability is None) != (self.probability_sha256 is None):
            raise ValueError("La calibration de probabilités doit correspondre à celle de l'ensemble conforme")
        if probability is not None:
            probability.validate(classifier, amp=amp)

    def validate_files(self, checkpoint, classifier_path, probability_path=None):
        if (file_sha256(checkpoint) != self.checkpoint_sha256 or file_sha256(classifier_path) != self.classifier_sha256
                or (file_sha256(probability_path) if probability_path else None) != self.probability_sha256):
            raise ValueError("Ensemble conforme incompatible avec les fichiers d'entrée")

    def predict(self, probabilities, top_indices=None):
        if probabilities.ndim != 2 or probabilities.size(-1) != self.num_classes:
            raise ValueError("Nombre de classes incompatible")
        mask = class_scores(probabilities, self.method) <= self.threshold
        top = probabilities.argmax(-1) if top_indices is None else top_indices
        if top.ndim != 1 or top.size(0) != probabilities.size(0) or top.dtype != torch.long or ((top < 0) | (top >= self.num_classes)).any():
            raise ValueError("Indices de classes invalides")
        return mask.scatter(1, top.unsqueeze(1), True)

    def to_dict(self):
        return dict(format_version=1, include_top=True, **self.__dict__)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop("format_version", None) != 1 or values.pop("include_top", None) is not True:
            raise ValueError("Format d'ensemble conforme inconnu")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_series_set(model, classifier, batches, *, checkpoint_sha256, classifier_sha256,
                   probability=None, probability_sha256=None, dataset_sha256=None,
                   coverage=.9, method="aps", amp="none"):
    if amp not in ("none", "bf16", "fp16") or method not in ("lac", "aps"):
        raise ValueError("Précision ou méthode invalide")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("Couverture invalide")
    classifier.validate(model.config, amp=amp)
    if classifier.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("La tête ne correspond pas au checkpoint")
    if (probability is None) != (probability_sha256 is None):
        raise ValueError("L'empreinte de calibration de probabilités est requise")
    if probability is not None:
        probability.validate(classifier, amp=amp)
    if dataset_sha256 is not None and dataset_sha256 in (classifier.dataset_sha256, getattr(probability, "dataset_sha256", None)):
        raise ValueError("Le jeu de calibration est identique à un jeu d'ajustement précédent")
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16))
    was_training = model.training
    model.eval()
    values = []
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
                    scores = classifier.scores(model, batch)[observed]
                probs = probability.probabilities(scores) if probability else log_probabilities(scores).exp()
                values.append(class_scores(probs, method).gather(1, target[observed, None]).flatten().cpu())
        if not values:
            raise ValueError("La calibration nécessite des exemples étiquetés")
        scores = torch.cat(values)
        threshold = conformal_threshold(scores, coverage)
        return SeriesPredictionSet(coverage, method, threshold, scores.numel(), model.config.num_classes,
                                   amp, checkpoint_sha256, classifier_sha256, probability_sha256, dataset_sha256)
    finally:
        model.train(was_training)
