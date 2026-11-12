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


@dataclass(frozen=True)
class SeriesAffineAugmentation:
    """Changer les unités et le niveau, pour l'historique et ses futurs cibles."""

    probability: float = 0.0
    min_scale: float = 0.1
    max_scale: float = 10.0
    max_shift: float = 20.0

    def __post_init__(self):
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probability doit être finie et appartenir à [0, 1]")
        if not all(math.isfinite(value) for value in (self.min_scale, self.max_scale, self.max_shift)):
            raise ValueError("Les bornes affines doivent être finies")
        if not 0 < self.min_scale <= self.max_scale or self.max_shift < 0:
            raise ValueError("0 < min_scale <= max_scale et max_shift >= 0 requis")

    def to_dict(self):
        return asdict(self)

    def __call__(self, batch, *, index, seed):
        if type(index) is not int or index < 0:
            raise ValueError("L'index d'augmentation doit être un entier positif ou nul")
        if "series" not in batch or not self.probability:
            return batch
        series = batch["series"]
        if series.ndim != 3 or not series.is_floating_point():
            raise ValueError("series doit être flottant [B, T, F]")
        future = batch.get("forecast_targets")
        if future is not None and (future.ndim != 3 or future.shape[::2] != series.shape[::2]
                                   or future.device != series.device or not future.is_floating_point()):
            raise ValueError("forecast_targets doit être flottant [B, H, F] sur le device des séries")
        generator = torch.Generator(device=series.device).manual_seed((seed + index * 1000003) % (2**63))
        shape = (series.size(0), series.size(2))
        gain = torch.rand(shape, generator=generator, device=series.device)
        gain = (math.log(self.min_scale) + gain * (math.log(self.max_scale) - math.log(self.min_scale))).exp()
        shift = (2 * torch.rand(shape, generator=generator, device=series.device) - 1) * self.max_shift
        selected = torch.rand(series.size(0), 1, generator=generator, device=series.device) < self.probability
        gain, shift = torch.where(selected, gain, 1), torch.where(selected, shift, 0)
        result = dict(batch)
        for key in ("series", "forecast_targets"):
            if key in batch:
                transformed = batch[key] * gain[:, None] + shift[:, None]
                if (torch.isfinite(batch[key]) & ~torch.isfinite(transformed)).any():
                    raise ValueError("L'augmentation affine dépasse la précision des données")
                result[key] = transformed
        return result
