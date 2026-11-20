"""Porte supervisée entre classifieurs temporels figés."""

from contextlib import nullcontext
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .calibration import file_sha256, modality_signature
from .temporal_precision import temporal_work_dtype


def gate_features(batch, logits, scores):
    """Scores des deux experts et statistiques observées, sans utiliser les étiquettes."""
    if modality_signature(batch) != ("series",):
        raise ValueError("La porte nécessite uniquement la modalité series")
    series = batch["series"]
    if (series.ndim != 3 or not series.is_floating_point() or series.size(0) < 1 or series.size(1) < 1
            or logits.ndim != 2 or scores.shape != logits.shape or logits.size(0) != series.size(0)
            or logits.device != series.device or scores.device != series.device
            or not torch.isfinite(logits).all() or not torch.isfinite(scores).all()):
        raise ValueError("Séries et scores des experts incompatibles ou non finis")
    mask = batch.get("series_mask")
    if mask is None:
        mask = torch.ones(series.shape[:2], dtype=torch.bool, device=series.device)
    elif (mask.shape != series.shape[:2] or mask.device != series.device
          or not ((mask == 0) | (mask == 1)).all() or not mask.bool().any(-1).all()):
        raise ValueError("series_mask doit être binaire avec au moins une position valide")
    mask = mask.bool()
    temporal_work_dtype(series, mask)
    x = series.double()
    observed = mask.unsqueeze(-1) & torch.isfinite(x)
    clean = torch.where(observed, x, 0.)
    counts = observed.sum(1).clamp_min(1)
    means = clean.sum(1) / counts
    centered = torch.where(observed, x - means.unsqueeze(1), 0.)
    scales = (centered.square().sum(1) / counts).sqrt().clamp_min(1e-5)
    length = mask.sum(1, keepdim=True).double()
    features = torch.cat((logits.double(), scores.double(), means, scales.log(), length), -1)
    if not torch.isfinite(features).all():
        raise ValueError("Les caractéristiques de la porte doivent être finies")
    return features


@dataclass
class SeriesGate:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: list
    biases: list
    series_features: int
    num_classes: int
    amp: str
    checkpoint_sha256: str
    classifier_sha256: str
    dataset_sha256: str | None
    validation_sha256: str | None
    threshold: float
    train_examples: int
    validation_examples: int
    best_epoch: int
    validation_gain: float
    epochs: int
    fit_batch_size: int
    learning_rate: float
    seed: int
    feature_batch_size: int | None
    feature_device: str
    torch_version: str

    def __post_init__(self):
        if type(self.series_features) is not int or self.series_features < 1 or type(self.num_classes) is not int or self.num_classes < 2:
            raise ValueError("Dimensions de la porte invalides")
        if self.amp not in ("none", "bf16", "fp16") or not math.isfinite(self.threshold) or not 0 <= self.threshold < 1:
            raise ValueError("Précision ou seuil invalide")
        if (type(self.train_examples) is not int or self.train_examples < 1 or type(self.validation_examples) is not int
                or self.validation_examples < 1 or type(self.best_epoch) is not int or self.best_epoch < 0
                or not math.isfinite(self.validation_gain) or not 0 <= self.validation_gain <= 1):
            raise ValueError("Statistiques de validation invalides")
        if (type(self.epochs) is not int or self.epochs < 1 or self.best_epoch > self.epochs
                or type(self.fit_batch_size) is not int or self.fit_batch_size < 1
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or type(self.seed) is not int or not 0 <= self.seed < 2**63
                or self.feature_batch_size is not None and
                    (type(self.feature_batch_size) is not int or self.feature_batch_size < 1)
                or not isinstance(self.feature_device, str) or not self.feature_device
                or not isinstance(self.torch_version, str) or not self.torch_version):
            raise ValueError("Provenance d'ajustement invalide")
        for digest in (self.checkpoint_sha256, self.classifier_sha256, self.dataset_sha256, self.validation_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Une empreinte SHA-256 hexadécimale est requise")
        if self.checkpoint_sha256 is None or self.classifier_sha256 is None:
            raise ValueError("Empreintes du checkpoint et de la tête requises")
        dimension = 2 * self.num_classes + 2 * self.series_features + 1
        self.mean = torch.as_tensor(self.mean, dtype=torch.float64).detach().cpu()
        self.scale = torch.as_tensor(self.scale, dtype=torch.float64).detach().cpu()
        if (self.mean.shape != (dimension,) or self.scale.shape != (dimension,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all() or (self.scale <= 0).any()):
            raise ValueError("Standardisation de la porte invalide")
        if len(self.weights) != 3 or len(self.biases) != 3:
            raise ValueError("Trois couches sont requises")
        self.weights = [torch.as_tensor(w, dtype=torch.float32).detach().cpu() for w in self.weights]
        self.biases = [torch.as_tensor(b, dtype=torch.float32).detach().cpu() for b in self.biases]
        shapes = ((64, dimension), (32, 64), (1, 32))
        for weight, bias, shape in zip(self.weights, self.biases, shapes):
            if (weight.shape != shape or bias.shape != (shape[0],)
                    or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
                raise ValueError("Poids de la porte invalides")

    def validate(self, classifier, *, amp=None):
        if (classifier.num_classes != self.num_classes or classifier.series_features != self.series_features
                or classifier.checkpoint_sha256 != self.checkpoint_sha256 or classifier.amp != self.amp
                or amp is not None and amp != self.amp):
            raise ValueError("Porte incompatible avec la tête, le checkpoint ou la précision")

    def validate_files(self, checkpoint, classifier_path):
        if file_sha256(checkpoint) != self.checkpoint_sha256 or file_sha256(classifier_path) != self.classifier_sha256:
            raise ValueError("La porte ne correspond pas au checkpoint ou à la tête")

    def advantage(self, batch, logits, scores):
        if logits.ndim != 2 or logits.size(-1) != self.num_classes or batch["series"].size(-1) != self.series_features:
            raise ValueError("Dimensions incompatibles avec la porte")
        value = gate_features(batch, logits, scores)
        value = ((value - self.mean.to(value.device)) / self.scale.to(value.device)).float()
        if not torch.isfinite(value).all():
            raise ValueError("Caractéristiques normalisées incompatibles avec FP32")
        with torch.autocast(device_type=value.device.type, enabled=False):
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                value = F.linear(value, weight.to(value.device), bias.to(value.device))
                if index < 2:
                    value = F.silu(value)
        if not torch.isfinite(value).all():
            raise ValueError("Avantage de la porte non fini")
        return value.squeeze(-1)

    def choose(self, batch, logits, scores):
        advantage = self.advantage(batch, logits, scores)
        use_head = advantage > self.threshold
        return torch.where(use_head, scores.argmax(-1), logits.argmax(-1)), use_head, advantage

    def to_dict(self):
        return dict(format_version=1, method="advantage_mlp", mean=self.mean.tolist(), scale=self.scale.tolist(),
                    weights=[x.tolist() for x in self.weights], biases=[x.tolist() for x in self.biases],
                    series_features=self.series_features, num_classes=self.num_classes, amp=self.amp,
                    checkpoint_sha256=self.checkpoint_sha256, classifier_sha256=self.classifier_sha256,
                    dataset_sha256=self.dataset_sha256, validation_sha256=self.validation_sha256,
                    threshold=self.threshold, train_examples=self.train_examples,
                    validation_examples=self.validation_examples, best_epoch=self.best_epoch,
                    validation_gain=self.validation_gain, epochs=self.epochs, fit_batch_size=self.fit_batch_size,
                    learning_rate=self.learning_rate, seed=self.seed, feature_batch_size=self.feature_batch_size,
                    feature_device=self.feature_device, torch_version=self.torch_version)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if values.pop("format_version", None) != 1 or values.pop("method", None) != "advantage_mlp":
            raise ValueError("Format de porte inconnu")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_series_gate(model, classifier, batches, validation_batches, *, checkpoint_sha256, classifier_sha256,
                    dataset_sha256=None, validation_sha256=None, amp="none", threshold=.03,
                    epochs=50, batch_size=128, learning_rate=.001, seed=0, feature_batch_size=None):
    """Apprendre le gain de classification de la tête, puis sélectionner sur validation réservée."""
    if (amp not in ("none", "bf16", "fp16") or not math.isfinite(threshold) or not 0 <= threshold < 1
            or type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0 or type(seed) is not int or not 0 <= seed < 2**63
            or feature_batch_size is not None and (type(feature_batch_size) is not int or feature_batch_size < 1)):
        raise ValueError("Paramètres d'ajustement invalides")
    if model.config.num_classes < 2:
        raise ValueError("Au moins deux classes sont requises")
    classifier.validate(model.config, amp=amp)
    if classifier.checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("La tête ne correspond pas au checkpoint")
    if (dataset_sha256 is not None and dataset_sha256 in (classifier.dataset_sha256, validation_sha256)
            or validation_sha256 is not None and validation_sha256 == classifier.dataset_sha256):
        raise ValueError("Les jeux d'ajustement et de validation doivent être distincts")
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16))
    modes = [(module, module.training) for module in model.modules()]
    model.eval()

    def collect(source):
        features, labels = [], []
        previous_size = None
        with torch.inference_mode():
            for batch in source:
                batch = {key: value.to(device) for key, value in batch.items()}
                size = batch["series"].size(0) if "series" in batch else 0
                if feature_batch_size is not None and (size < 1 or size > feature_batch_size
                        or previous_size is not None and previous_size != feature_batch_size):
                    raise ValueError("La taille des lots ne correspond pas à feature_batch_size")
                previous_size = size
                if modality_signature(batch) != ("series",):
                    raise ValueError("La porte nécessite uniquement la modalité series")
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
                features.append(gate_features(inputs, logits, scores)[observed].cpu())
                labels.append(((scores.argmax(-1) == target).float() - (logits.argmax(-1) == target).float())
                              [observed].cpu())
        if not features:
            raise ValueError("La porte nécessite des exemples étiquetés dans chaque jeu")
        # Cat hors inference_mode pour conserver des tenseurs compatibles autograd.
        return torch.cat(features), torch.cat(labels)

    try:
        x, y = collect(batches)
        xv, yv = collect(validation_batches)
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(.01)
        x, xv = ((x - mean) / scale).float(), ((xv - mean) / scale).float()
        if not torch.isfinite(x).all() or not torch.isfinite(xv).all():
            raise ValueError("Caractéristiques normalisées incompatibles avec FP32")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            network = nn.Sequential(nn.Linear(x.size(1), 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=.01)
        rng = torch.Generator().manual_seed(seed)
        best, best_epoch, best_state = 0., 0, None
        for epoch in range(1, epochs + 1):
            for indices in torch.randperm(x.size(0), generator=rng).split(batch_size):
                prediction = network(x[indices]).flatten()
                loss = (prediction - y[indices]).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Perte de la porte non finie")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                values = network(xv).flatten()
                gain = ((values > threshold) * yv).mean().item()
                if gain > best:
                    best, best_epoch = gain, epoch
                    best_state = [layer.weight.detach().clone() for layer in (network[0], network[2], network[4])], [
                        layer.bias.detach().clone() for layer in (network[0], network[2], network[4])]
        if best_state is None:
            # Retour sûr au modèle de base si aucune époque n'améliore la validation.
            with torch.no_grad():
                for layer in (network[0], network[2], network[4]):
                    layer.weight.zero_()
                    layer.bias.zero_()
                network[4].bias.fill_(-1.)
            best_state = [layer.weight.detach().clone() for layer in (network[0], network[2], network[4])], [
                layer.bias.detach().clone() for layer in (network[0], network[2], network[4])]
        return SeriesGate(mean, scale, *best_state, model.config.series_features, model.config.num_classes,
                          amp, checkpoint_sha256, classifier_sha256, dataset_sha256, validation_sha256,
                          threshold, x.size(0), xv.size(0), best_epoch, best, epochs, batch_size, learning_rate,
                          seed, feature_batch_size, str(device), str(torch.__version__))
    finally:
        for module, training in modes:
            module.training = training
