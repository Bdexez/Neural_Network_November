"""Sélection des entrées et couverture déterministe des modalités disponibles."""

import itertools
import random


MODALITIES = ("input_ids", "images", "series")
TARGETS = ("labels", "class_labels", "forecast_targets")


def keep_modalities(batch, names):
    """Conserver les cibles de sortie et retirer les masques des entrées omises."""
    result = batch.copy()
    for name in MODALITIES:
        if name in result and name not in names:
            del result[name]
            if name == "input_ids":
                result.pop("attention_mask", None)
                result.pop("labels", None)
            elif name == "series":
                result.pop("series_mask", None)
    return result


def balanced_modalities(batch, index, seed=0):
    """Parcourir chaque sous-ensemble supervisé une fois par cycle mélangé.

    L'index global des micro-lots suffit pour reconstruire le choix après une
    reprise ou sur plusieurs rangs DDP. Le RNG d'entraînement n'est pas consommé.
    """
    if type(index) is not int or index < 0:
        raise ValueError("L'index de modalités doit être un entier positif ou nul")
    names = [name for name in MODALITIES if name in batch]
    choices = [subset for count in range(1, len(names) + 1)
               for subset in itertools.combinations(names, count)
               if any(target in keep_modalities(batch, subset) for target in TARGETS)]
    if not choices:
        raise ValueError("Le lot doit contenir une entrée et une cible supervisée")
    cycle, position = divmod(index, len(choices))
    random.Random(f"chimera-modalities:{seed}:{cycle}").shuffle(choices)
    return keep_modalities(batch, choices[position])
