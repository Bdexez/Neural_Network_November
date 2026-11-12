"""Correction affine de prévision à partir des observations les plus récentes."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ForecastAdapterContext:
    history: torch.Tensor
    anchor: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class ForecastAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config.forecast_adapter_context
        self.reference = config.forecast_context
        self.median_index = config.quantiles.index(0.5)
        self.proj = nn.Linear(2 * self.context + 4 + config.forecast_horizon, config.forecast_horizon)

    def prepare(self, series, valid, mean, scale):
        series = series.float()
        observed = torch.isfinite(series) & valid.unsqueeze(-1)
        clean = torch.where(observed, series, 0.0)
        positions = torch.arange(series.size(1), device=series.device)[None].expand_as(valid)
        last = positions[..., None].expand_as(series).masked_fill(~observed, -1).max(1).values
        anchor = clean.gather(1, last.clamp_min(0)[:, None]).squeeze(1)
        anchor = torch.where(last >= 0, anchor, mean)
        # Les indices croissants gardent les derniers pas valides, sans comprimer
        # le temps par un redimensionnement de l'historique.
        recent = positions.masked_fill(~valid, -1).sort(-1).values[:, -self.context:]
        recent = F.pad(recent, (self.context - recent.size(1), 0), value=-1)
        indices = recent.clamp_min(0)[..., None].expand(-1, -1, series.size(-1))
        flags = observed.gather(1, indices) & (recent >= 0).unsqueeze(-1)
        normalized = (clean.gather(1, indices) - anchor[:, None]) / scale[:, None]
        normalized = torch.where(flags, normalized, 0.0)
        length = valid.sum(-1).float().clamp_min(1)
        summary = torch.stack((mean.asinh(), scale.log(), anchor.asinh(),
                               (length[:, None].expand_as(mean) / self.reference).log()), dim=-1)
        history = torch.cat((normalized.transpose(1, 2), flags.float().transpose(1, 2), summary), dim=-1)
        gate = (1 - length / self.reference).clamp(0, 1)
        return ForecastAdapterContext(history, anchor, scale, gate)

    def features(self, forecast, context):
        median = forecast[..., self.median_index].transpose(1, 2)
        previous = (median - context.anchor[..., None]) / context.scale[..., None]
        return torch.cat((context.history, previous), dim=-1)

    def forward(self, forecast, context):
        features = self.features(forecast, context)
        correction = self.proj(features.to(self.proj.weight.dtype)).transpose(1, 2)
        correction = correction * context.scale[:, None] * context.gate[:, None, None]
        # Une translation commune conserve l'ordre des quantiles et n'intervient
        # ni dans les logits de texte ni dans les logits de classification.
        return forecast + correction.unsqueeze(-1)
