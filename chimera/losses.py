"""Pertes d'alignement entre modalités, avec positifs multiples facultatifs."""

import torch
from torch.nn import functional as F


def quantile_pinball_loss(forecast, targets, quantiles):
    """Perte moyenne des quantiles sur les cibles finies, accumulée au moins en FP32."""
    dtype = torch.promote_types(torch.float32, torch.promote_types(forecast.dtype, targets.dtype))
    observed = torch.isfinite(targets)
    # Une prévision sur une cible absente ne contribue ni à la perte ni à son gradient.
    prediction = torch.where(observed.unsqueeze(-1), forecast, 0).to(dtype=dtype)
    target = torch.where(observed, targets, 0).to(dtype=dtype)
    magnitude = torch.maximum(prediction.detach().abs().amax(), target.detach().abs().amax()).item()
    if dtype == torch.float32:
        # Un résidu peut atteindre deux fois l'amplitude et chaque élément
        # peut contribuer à la somme avant sa division par le nombre de cibles.
        limit = torch.finfo(torch.float32).max / (4 * max(forecast.numel(), 1))
        if magnitude > limit:
            prediction, target = prediction.double(), target.double()
            dtype = torch.float64
    # Même en FP64, la différence ou la somme peuvent déborder avant la
    # division finale. La demi-différence évite de perdre les petits résidus
    # d'autres exemples, même si une prévision est proche du maximum FP64.
    limit = torch.finfo(torch.float64).max / (4 * max(forecast.numel(), 1))
    extreme = dtype == torch.float64 and magnitude > limit
    residual = (target.unsqueeze(-1) * 0.5 - prediction * 0.5 if extreme
                else target.unsqueeze(-1) - prediction)
    q = prediction.new_tensor(quantiles)
    pinball = torch.maximum(q * residual, (q - 1) * residual)
    denominator = observed.sum().clamp_min(1) * len(quantiles)
    if extreme:
        # La somme est positive : diviser d'abord uniquement si nécessaire,
        # puis rétablir le facteur deux une fois la moyenne calculée.
        maximum = pinball.detach().amax().item()
        if maximum > torch.finfo(torch.float64).max / (2 * max(forecast.numel(), 1)):
            return (pinball / denominator).sum() * 2
        return (pinball.sum() / denominator) * 2
    return pinball.sum() / denominator



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
