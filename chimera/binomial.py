"""Intervalles de Wilson pour des observations binaires indépendantes."""

import math
from statistics import NormalDist


def wilson_interval(successes, trials, confidence=.95):
    """Intervalle bilatéral approché d'une proportion binomiale, ou None si n=0."""
    if type(successes) is not int or type(trials) is not int or trials < 0 or not 0 <= successes <= trials:
        raise ValueError("Les effectifs binomiaux doivent être des entiers cohérents")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Le niveau de confiance doit être strictement compris entre 0 et 1")
    if not trials:
        return None
    z = -NormalDist().inv_cdf((1 - confidence) / 2)
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return [max(0., center - radius), min(1., center + radius)]
