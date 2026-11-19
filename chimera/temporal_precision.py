"""Choix de précision pour les statistiques des historiques temporels."""

import math

import torch


def temporal_work_dtype(series, valid):
    """Éviter le débordement des sommes et variances avant la normalisation.

    La borne tient compte du pire écart à la moyenne (deux fois l'amplitude)
    et du nombre maximal de termes dans la somme des carrés.
    """
    observed = valid.unsqueeze(-1) & torch.isfinite(series)
    magnitude = torch.where(observed, series.abs(), 0).amax().item()
    length = series.size(1)
    float_limit = math.sqrt(torch.finfo(torch.float32).max / (4 * length))
    if magnitude <= float_limit:
        return torch.float32
    double_limit = math.sqrt(torch.finfo(torch.float64).max / (4 * length))
    if magnitude > double_limit:
        raise ValueError("Amplitude de série trop élevée pour calculer une variance finie")
    return torch.float64
