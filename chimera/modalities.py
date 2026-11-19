"""Sélection des entrées et couverture déterministe des modalités disponibles."""

import itertools
import random

import torch


MODALITIES = ("input_ids", "images", "series")
TARGETS = ("labels", "class_labels", "forecast_targets")


def has_observed_targets(batch, *, prediction_steps=1):
    """Vérifier les cibles qui contribuent effectivement aux pertes supervisées."""
    labels = batch.get("labels")
    if labels is not None:
        valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
        for distance in range(1, min(prediction_steps + 1, labels.size(1))):
            if ((labels[:, distance:] != -100) & valid[:, distance:] & valid[:, :-distance]).any().item():
                return True
    classes = batch.get("class_labels")
    if classes is not None and (classes != -100).any().item():
        return True
    future = batch.get("forecast_targets")
    return future is not None and torch.isfinite(future).any().item()


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


def balanced_modalities(batch, index, seed=0, *, prediction_steps=1):
    """Parcourir chaque sous-ensemble supervisé une fois par cycle mélangé.

    L'index global des micro-lots suffit pour reconstruire le choix après une
    reprise ou sur plusieurs rangs DDP. Le RNG d'entraînement n'est pas consommé.
    """
    if type(index) is not int or index < 0:
        raise ValueError("L'index de modalités doit être un entier positif ou nul")
    names = [name for name in MODALITIES if name in batch]
    # Seule la suppression du texte peut retirer une cible ; évaluer les deux
    # possibilités une fois pour éviter des synchronisations GPU par sous-ensemble.
    with_text = has_observed_targets(batch, prediction_steps=prediction_steps)
    without_text = has_observed_targets(keep_modalities(batch, ("images", "series")),
                                        prediction_steps=prediction_steps)
    choices = [subset for count in range(1, len(names) + 1)
               for subset in itertools.combinations(names, count)
               if with_text and "input_ids" in subset or without_text]
    if not choices:
        raise ValueError("Le lot doit contenir une entrée et une cible supervisée")
    cycle, position = divmod(index, len(choices))
    random.Random(f"chimera-modalities:{seed}:{cycle}").shuffle(choices)
    return keep_modalities(batch, choices[position])
