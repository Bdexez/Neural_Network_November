"""Correction affine de prévision à partir des observations les plus récentes."""

from dataclasses import dataclass
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F

from .modalities import TARGETS


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
        active = context.gate > 0
        if not active.any():
            return forecast
        features = self.features(forecast, context)[active]
        update = self.proj(features.to(self.proj.weight.dtype)).transpose(1, 2)
        update = update * context.scale[active, None] * context.gate[active, None, None]
        correction = forecast.new_zeros(forecast.shape[:-1])
        correction[active] = update.to(correction.dtype)
        # Une translation commune conserve l'ordre des quantiles et n'intervient
        # ni dans les logits de texte ni dans les logits de classification.
        return forecast + correction.unsqueeze(-1)


def fit_forecast_adapter(model, batches, *, ridge=0.001, amp="none"):
    """Ajuster uniquement la couche affine, par statistiques suffisantes en FP64.

    L'objectif est l'erreur quadratique de la médiane dans les unités des cibles,
    avec pénalisation L2 des coefficients et biais non pénalisé. Les cibles
    manquantes sont traitées séparément pour chaque horizon.
    """
    if model.forecast_adapter is None:
        raise ValueError("Activer forecast_adapter_context avant l'ajustement")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge doit être fini et strictement positif")
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    adapter = model.forecast_adapter
    horizon, dimension = adapter.proj.out_features, adapter.proj.in_features + 1
    gram = torch.zeros(horizon, dimension, dimension, dtype=torch.float64)
    cross = torch.zeros(horizon, dimension, dtype=torch.float64)
    counts = torch.zeros(horizon, dtype=torch.long)
    examples = active_examples = 0
    captured = {}
    # L'entrée du module précède sa correction existante : un réajustement
    # remplace ses coefficients, sans apprendre par-dessus sa propre sortie.
    hook = adapter.register_forward_pre_hook(lambda module, args: captured.update(forecast=args[0], context=args[1]))
    was_training = model.training
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                if "series" not in batch or "forecast_targets" not in batch:
                    raise ValueError("L'ajustement nécessite series et forecast_targets dans chaque lot")
                inputs = {key: value.to(device) for key, value in batch.items() if key not in TARGETS}
                captured.clear()
                with autocast():
                    model(**inputs)
                forecast, context = captured["forecast"], captured["context"]
                target = batch["forecast_targets"].to(device)
                if target.shape != forecast.shape[:-1]:
                    raise ValueError("forecast_targets doit partager la forme [B, H, F] des prévisions")
                features = adapter.features(forecast, context)
                if not torch.isfinite(forecast).all() or not torch.isfinite(features).all():
                    raise ValueError("Prévisions et caractéristiques finies requises pour l'ajustement")
                active = context.gate > 0
                examples += target.size(0)
                active_examples += active.sum().item()
                features = features[active].double().cpu()
                features = torch.cat((features, torch.ones(*features.shape[:2], 1, dtype=torch.float64)), dim=-1)
                multiplier = (context.scale[active] * context.gate[active, None]).double().cpu()
                design = (features * multiplier[..., None]).reshape(-1, dimension)
                residual = (target.double() - forecast[..., adapter.median_index].double()).transpose(1, 2)[active].cpu()
                residual = residual.reshape(-1, horizon)
                for step in range(horizon):
                    observed = torch.isfinite(residual[:, step])
                    x, y = design[observed], residual[observed, step]
                    gram[step].add_(x.T @ x)
                    cross[step].add_(x.T @ y)
                    counts[step] += observed.sum()
        if (counts == 0).any():
            raise ValueError("Chaque horizon doit avoir une cible observée avec un historique plus court que forecast_context")
        penalty = torch.eye(dimension, dtype=torch.float64) * ridge
        penalty[-1, -1] = 0
        coefficients = torch.linalg.solve(gram / counts[:, None, None] + penalty,
                                          (cross / counts[:, None]).unsqueeze(-1)).squeeze(-1)
        weight = coefficients[:, :-1].to(device=adapter.proj.weight.device, dtype=adapter.proj.weight.dtype)
        bias = coefficients[:, -1].to(device=adapter.proj.bias.device, dtype=adapter.proj.bias.dtype)
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Les coefficients ajustés ne sont pas finis dans la précision du modèle")
        with torch.no_grad():
            adapter.proj.weight.copy_(weight)
            adapter.proj.bias.copy_(bias)
        return dict(method="ridge", ridge=ridge, examples=examples, active_examples=active_examples,
                    observed_targets_by_horizon=counts.tolist(), context=adapter.context, amp=amp,
                    updated_parameters=sum(p.numel() for p in adapter.parameters()))
    finally:
        hook.remove()
        model.train(was_training)
