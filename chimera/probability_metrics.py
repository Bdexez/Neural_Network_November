"""Mesures de probabilités multiclasse pondérées par exemple."""

import torch
from torch.nn import functional as F


class ProbabilityMetrics:
    def __init__(self, bins=15):
        if type(bins) is not int or bins < 1:
            raise ValueError("bins doit être un entier positif")
        self.bins = bins
        self.count = 0
        self.nll = self.brier = 0.
        self.bin_counts = torch.zeros(bins, dtype=torch.long)
        self.bin_correct = torch.zeros(bins, dtype=torch.float64)
        self.bin_confidence = torch.zeros(bins, dtype=torch.float64)

    def update(self, log_probabilities, labels):
        logp = log_probabilities.double()
        if (logp.ndim != 2 or labels.dtype != torch.long or labels.device != logp.device
                or labels.shape != (logp.size(0),) or not torch.isfinite(logp).all()):
            raise ValueError("Log-probabilités finies [B, C] et étiquettes [B] requises")
        if ((labels < 0) | (labels >= logp.size(1))).any():
            raise ValueError("Étiquette hors du nombre de classes")
        if not labels.numel():
            return
        probabilities = logp.exp()
        if not torch.allclose(probabilities.sum(-1), torch.ones_like(labels, dtype=probabilities.dtype),
                              rtol=1e-8, atol=1e-10):
            raise ValueError("Les probabilités doivent sommer à un")
        predicted = logp.argmax(-1)
        confidence = probabilities.gather(1, predicted[:, None]).squeeze(1)
        correct = predicted == labels
        self.count += labels.numel()
        self.nll += (-logp.gather(1, labels[:, None])).sum().item()
        self.brier += (probabilities - F.one_hot(labels, logp.size(1))).square().sum().item()
        index = (confidence * self.bins).long().clamp_max(self.bins - 1).cpu()
        self.bin_counts += torch.bincount(index, minlength=self.bins)
        self.bin_correct += torch.bincount(index, weights=correct.double().cpu(), minlength=self.bins)
        self.bin_confidence += torch.bincount(index, weights=confidence.cpu(), minlength=self.bins)

    def result(self):
        rows = []
        ece = 0.
        for index in range(self.bins):
            count = self.bin_counts[index].item()
            accuracy = self.bin_correct[index].item() / count if count else None
            confidence = self.bin_confidence[index].item() / count if count else None
            if count:
                ece += count * abs(accuracy - confidence)
            rows.append(dict(lower=index / self.bins, upper=(index + 1) / self.bins,
                             count=count, accuracy=accuracy, confidence=confidence))
        return dict(examples=self.count, nll=self.nll / self.count if self.count else None,
                    brier=self.brier / self.count if self.count else None,
                    ece=ece / self.count if self.count else None, bins=rows)
