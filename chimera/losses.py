"""Pertes d'alignement entre modalités, avec positifs multiples facultatifs."""

import torch
from torch.nn import functional as F


def quantile_pinball_loss(forecast, targets, quantiles):
    """Perte moyenne des quantiles sur les cibles finies, accumulée au moins en FP32."""
    dtype = torch.promote_types(torch.float32, torch.promote_types(forecast.dtype, targets.dtype))
    prediction = forecast.to(dtype=dtype)
    observed = torch.isfinite(targets)
    target = torch.where(observed, targets, 0).to(dtype=dtype)
    residual = target.unsqueeze(-1) - prediction
    q = prediction.new_tensor(quantiles)
    pinball = torch.maximum(q * residual, (q - 1) * residual)
    return (pinball * observed.unsqueeze(-1)).sum() / (observed.sum().clamp_min(1) * len(quantiles))


def symmetric_contrastive_loss(similarity, class_labels=None):
    """InfoNCE symétrique ; les classes connues peuvent partager leurs positifs.

    La diagonale reste positive, y compris pour les classes ignorées (-100).
    L'extension utilise la moyenne des log-probabilités des positifs, comme
    l'objectif supervisé de https://arxiv.org/abs/2004.11362, entre deux modalités.
    """
    scores = similarity.float()
    identity = torch.arange(scores.size(0), device=scores.device)
    if class_labels is None:
        return (F.cross_entropy(scores, identity) + F.cross_entropy(scores.T, identity)) / 2
    known = class_labels != -100
    positive = torch.eye(scores.size(0), device=scores.device, dtype=torch.bool)
    positive |= (class_labels[:, None] == class_labels[None, :]) & known[:, None] & known[None, :]
    weights = positive.float() / positive.sum(-1, keepdim=True)
    forward = -(weights * F.log_softmax(scores, dim=-1)).sum(-1).mean()
    backward = -(weights * F.log_softmax(scores.T, dim=-1)).sum(-1).mean()
    return (forward + backward) / 2
