"""Réglage conforme du seuil d'une porte pour limiter les erreurs supplémentaires."""

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from .calibration import file_sha256, modality_signature


def _content_sha256(artifact):
    content = json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _conformal_rank(n, max_harm):
    numerator, denominator = float(max_harm).as_integer_ratio()
    return ((n + 1) * (denominator - numerator) + denominator - 1) // denominator


def harm_threshold(advantages, harmful, max_harm, minimum_threshold):
    """Quantile conforme de l'avantage sur les cas où la tête aggrave la prédiction.

    Un résultat None désactive toute sélection de la tête lorsque l'effectif ne
    permet pas d'atteindre le niveau demandé. Les égalités au seuil restent au modèle.
    """
    if not math.isfinite(max_harm) or not 0 < max_harm < 1 or not math.isfinite(minimum_threshold):
        raise ValueError("Risque et seuil invalides")
    if (advantages.ndim != 1 or harmful.shape != advantages.shape or harmful.dtype != torch.bool
            or not advantages.numel() or not torch.isfinite(advantages).all()):
        raise ValueError("Avantages finis et indicateurs de dommage requis")
    n = advantages.numel()
    rank = _conformal_rank(n, max_harm)
    if rank > n:
        return None
    values = torch.where(harmful, advantages.double(), -torch.inf)
    return max(minimum_threshold, values.kthvalue(rank).values.item())


@dataclass
class SeriesGateGuard:
    max_harm: float
    threshold: float | None
    gate_threshold: float
    examples: int
    unguarded_harms: int
    guarded_harms: int
    checkpoint_sha256: str
    classifier_sha256: str
    gate_sha256: str
    classifier_content_sha256: str
    gate_content_sha256: str
    dataset_sha256: str | None
    amp: str
    feature_batch_size: int | None
    feature_device: str
    torch_version: str

    def __post_init__(self):
        if (not math.isfinite(self.max_harm) or not 0 < self.max_harm < 1
                or not math.isfinite(self.gate_threshold) or not 0 <= self.gate_threshold < 1
                or self.threshold is not None and
                   (not math.isfinite(self.threshold) or self.threshold < self.gate_threshold
                    or self.threshold > torch.finfo(torch.float32).max)):
            raise ValueError("Paramètres de protection invalides")
        if (type(self.examples) is not int or self.examples < 1
                or type(self.unguarded_harms) is not int or type(self.guarded_harms) is not int
                or not 0 <= self.guarded_harms <= self.unguarded_harms <= self.examples
                or self.threshold is None and self.guarded_harms != 0):
            raise ValueError("Effectifs de protection invalides")
        rank = _conformal_rank(self.examples, self.max_harm)
        if self.threshold is not None and (rank > self.examples or self.guarded_harms > self.examples - rank):
            raise ValueError("Effectifs incompatibles avec le quantile conforme")
        if (self.amp not in ("none", "bf16", "fp16")
                or self.feature_batch_size is not None and
                   (type(self.feature_batch_size) is not int or self.feature_batch_size < 1)
                or not isinstance(self.feature_device, str) or not self.feature_device
                or not isinstance(self.torch_version, str) or not self.torch_version):
            raise ValueError("Précision ou provenance invalide")
        for digest in (self.checkpoint_sha256, self.classifier_sha256, self.gate_sha256,
                       self.classifier_content_sha256, self.gate_content_sha256, self.dataset_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")
        if any(value is None for value in (self.checkpoint_sha256, self.classifier_sha256, self.gate_sha256,
                                           self.classifier_content_sha256, self.gate_content_sha256)):
            raise ValueError("Empreintes du checkpoint, de la tête et de la porte requises")

    def validate(self, gate, classifier, *, amp=None):
        gate.validate(classifier, amp=amp)
        if (gate.checkpoint_sha256 != self.checkpoint_sha256 or gate.classifier_sha256 != self.classifier_sha256
                or gate.threshold != self.gate_threshold or gate.amp != self.amp
                or amp is not None and amp != self.amp
                or _content_sha256(classifier) != self.classifier_content_sha256
                or _content_sha256(gate) != self.gate_content_sha256):
            raise ValueError("Protection incompatible avec la tête, la porte ou la précision")

    def validate_files(self, checkpoint, classifier_path, gate_path):
        if (file_sha256(checkpoint) != self.checkpoint_sha256
                or file_sha256(classifier_path) != self.classifier_sha256
                or file_sha256(gate_path) != self.gate_sha256):
            raise ValueError("Protection incompatible avec le checkpoint, la tête ou la porte")

    def select(self, advantage):
        if advantage.ndim != 1 or not torch.isfinite(advantage).all():
            raise ValueError("Avantages finis [B] requis")
        if self.threshold is None:
            return torch.zeros_like(advantage, dtype=torch.bool)
        return advantage > self.threshold

    def to_dict(self):
        return dict(format_version=1, method="conformal_harm", **self.__dict__)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop("format_version", None) != 1 or values.pop("method", None) != "conformal_harm":
            raise ValueError("Format de protection de porte inconnu")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_series_gate_guard(model, classifier, gate, batches, *, checkpoint_sha256, classifier_sha256,
                          gate_sha256, dataset_sha256=None, max_harm=.01, amp="none", feature_batch_size=None):
    if (amp not in ("none", "bf16", "fp16") or not math.isfinite(max_harm) or not 0 < max_harm < 1
            or feature_batch_size is not None and (type(feature_batch_size) is not int or feature_batch_size < 1)):
        raise ValueError("Précision ou risque invalide")
    classifier.validate(model.config, amp=amp)
    gate.validate(classifier, amp=amp)
    if gate.checkpoint_sha256 != checkpoint_sha256 or gate.classifier_sha256 != classifier_sha256:
        raise ValueError("La porte ne correspond pas au checkpoint ou à la tête")
    if dataset_sha256 is not None and dataset_sha256 in (classifier.dataset_sha256, gate.dataset_sha256,
                                                           gate.validation_sha256):
        raise ValueError("Le jeu de protection est identique à un jeu d'ajustement ou de validation")
    classifier_content_sha = _content_sha256(classifier)
    gate_content_sha = _content_sha256(gate)
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16))
    modes = [(module, module.training) for module in model.modules()]
    values, harms = [], []
    previous_size = None
    try:
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                size = batch["series"].size(0) if "series" in batch else 0
                if feature_batch_size is not None and (size < 1 or size > feature_batch_size
                        or previous_size is not None and previous_size != feature_batch_size):
                    raise ValueError("La taille des lots ne correspond pas à feature_batch_size")
                previous_size = size
                if modality_signature(batch) != ("series",):
                    raise ValueError("La protection nécessite uniquement la modalité series")
                if "class_labels" not in batch:
                    continue
                target = batch["class_labels"]
                if (target.dtype != torch.long or target.ndim != 1 or target.size(0) != batch["series"].size(0)
                        or ((target != -100) & ((target < 0) | (target >= model.config.num_classes))).any()):
                    raise ValueError("class_labels doit contenir des classes valides ou -100")
                observed = target != -100
                if not observed.any():
                    continue
                inputs = {key: value for key, value in batch.items() if key in ("series", "series_mask")}
                with autocast():
                    logits = model(**inputs).classification_logits
                    scores = classifier.scores(model, inputs)
                advantage = gate.advantage(inputs, logits, scores)
                harmful = (logits.argmax(-1) == target) & (scores.argmax(-1) != target)
                # Préserver FP32 pour reproduire la comparaison du seuil à l'inférence.
                values.append(advantage[observed].cpu())
                harms.append(harmful[observed].cpu())
        if not values:
            raise ValueError("La protection nécessite des exemples étiquetés")
        if _content_sha256(classifier) != classifier_content_sha or _content_sha256(gate) != gate_content_sha:
            raise ValueError("La tête ou la porte a changé pendant la calibration")
        advantage, harmful = torch.cat(values), torch.cat(harms)
        threshold = harm_threshold(advantage, harmful, max_harm, gate.threshold)
        unguarded = int(((advantage > gate.threshold) & harmful).sum().item())
        guarded = int(((advantage > threshold) & harmful).sum().item()) if threshold is not None else 0
        return SeriesGateGuard(max_harm, threshold, gate.threshold, advantage.numel(), unguarded, guarded,
                               checkpoint_sha256, classifier_sha256, gate_sha256, classifier_content_sha,
                               gate_content_sha, dataset_sha256, amp,
                               feature_batch_size, str(device), str(torch.__version__))
    finally:
        for module, training in modes:
            module.training = training
