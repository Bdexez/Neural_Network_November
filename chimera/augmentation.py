"""Perturbations des historiques, sans modifier les cibles ni le RNG du réseau."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class SeriesAugmentation:
    crop_probability: float = 0.0
    missing_rate: float = 0.0
    mask_probability: float = 0.5
    min_history: int = 8

    def __post_init__(self):
        for name in ("crop_probability", "missing_rate", "mask_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} doit être fini et appartenir à [0, 1]")
        if type(self.min_history) is not int or self.min_history < 1:
            raise ValueError("min_history doit être un entier strictement positif")

    def to_dict(self):
        return asdict(self)

    def __call__(self, batch, *, index, seed):
        if type(index) is not int or index < 0:
            raise ValueError("L'index d'augmentation doit être un entier positif ou nul")
        if "series" not in batch or not (self.crop_probability or self.missing_rate * self.mask_probability):
            return batch
        series = batch["series"]
        if series.ndim != 3 or not series.is_floating_point():
            raise ValueError("series doit être flottant [B, T, F]")
        valid = batch.get("series_mask", torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)).bool()
        if valid.shape != series.shape[:2] or not valid.any(-1).all():
            raise ValueError("series_mask doit garder au moins une position valide par exemple")
        stream_seed = (seed + index * 1000003) % (2**63)
        values = series.clone()
        result = dict(batch)
        if self.crop_probability:
            generator = torch.Generator(device=series.device).manual_seed(stream_seed)
            count = valid.sum(-1)
            minimum = count.clamp(max=self.min_history)
            lengths = minimum + (torch.rand(series.size(0), generator=generator, device=series.device)
                                 * (count - minimum + 1)).long()
            selected = torch.rand(series.size(0), generator=generator, device=series.device) < self.crop_probability
            lengths = torch.where(selected, lengths, count)
            reverse_rank = valid.flip(1).long().cumsum(1).flip(1)
            valid = valid & (reverse_rank <= lengths[:, None])
            result["series_mask"] = valid
            values.masked_fill_(~valid.unsqueeze(-1), float("nan"))
        if self.missing_rate and self.mask_probability:
            # Un second flux garde le masquage indépendant du tirage des longueurs.
            generator = torch.Generator(device=series.device).manual_seed((stream_seed + 1) % (2**63))
            selected = torch.rand(series.size(0), generator=generator, device=series.device) < self.mask_probability
            missing = torch.rand(series.shape, generator=generator, device=series.device) < self.missing_rate
            missing &= selected[:, None, None]
            values.masked_fill_(missing & valid.unsqueeze(-1), float("nan"))
        result["series"] = values
        return result
