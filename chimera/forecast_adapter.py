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
    long_missing_gate: torch.Tensor | None = None


class ForecastAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config.forecast_adapter_context
        self.reference = config.forecast_context
        self.median_index = config.quantiles.index(0.5)
        self.relative = config.forecast_adapter_relative
        self.missing_gate = config.forecast_adapter_missing_gate
        self.proj = nn.Linear(2 * self.context + 4 + config.forecast_horizon, config.forecast_horizon)
        self.missing_proj = (nn.Linear(self.proj.in_features, self.proj.out_features)
                             if config.forecast_adapter_separate_missing else None)

    def prepare(self, series, valid, mean, scale):
        series = series.to(dtype=mean.dtype)
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
        duration = (length[:, None].expand_as(mean) / self.reference).log()
        if self.relative:
            age = (valid[..., None] & (positions[..., None] > last[:, None])).sum(1) / length[:, None]
            summary = torch.stack(((mean - anchor) / scale, observed.sum(1) / length[:, None], age, duration), dim=-1)
        else:
            summary = torch.stack((mean.asinh(), scale.log(), anchor.asinh(), duration), dim=-1)
        history = torch.cat((normalized.transpose(1, 2), flags.float().transpose(1, 2), summary), dim=-1)
        gate = (1 - length / self.reference).clamp(0, 1)[:, None].expand_as(mean)
        long_missing_gate = None
        if self.missing_gate:
            count = observed.sum(1)
            missing_fraction = 1 - count / length[:, None]
            if self.missing_proj is not None:
                long_missing_gate = torch.where((count > 0) & (length[:, None] >= self.reference),
                                                missing_fraction, 0)
            else:
                gate = torch.where(count > 0, torch.maximum(gate, missing_fraction), 0)
        return ForecastAdapterContext(history, anchor, scale, gate, long_missing_gate)

    def features(self, forecast, context):
        median = forecast[..., self.median_index].transpose(1, 2)
        previous = (median - context.anchor[..., None]) / context.scale[..., None]
        return torch.cat((context.history, previous), dim=-1)

    def forward(self, forecast, context):
        active = (context.gate > 0).any(-1)
        if context.long_missing_gate is not None:
            active |= (context.long_missing_gate > 0).any(-1)
        if not active.any():
            return forecast
        features = self.features(forecast, context)[active]
        update = self.proj(features.to(self.proj.weight.dtype)).transpose(1, 2)
        update = update * context.scale[active, None] * context.gate[active, None]
        if self.missing_proj is not None:
            missing_update = self.missing_proj(features.to(self.missing_proj.weight.dtype)).transpose(1, 2)
            update = update + missing_update * context.scale[active, None] * context.long_missing_gate[active, None]
        correction = forecast.new_zeros(forecast.shape[:-1])
        correction[active] = update.to(correction.dtype)
        # Une translation commune conserve l'ordre des quantiles et n'intervient
        # ni dans les logits de texte ni dans les logits de classification.
        return forecast + correction.unsqueeze(-1)


def fit_forecast_adapter(model, batches, *, ridge=0.001, amp="none"):
    """Ajuster uniquement les projections affines, par statistiques suffisantes en FP64.

    L'objectif est l'erreur quadratique de la médiane dans les unités des cibles,
    avec pénalisation L2 des coefficients et biais non pénalisé. Les cibles
    manquantes sont traitées séparément pour chaque horizon. Dans le mode à
    deux branches disjointes, une branche sans cible conserve ses coefficients.
    """
    if model.forecast_adapter is None:
        raise ValueError("Activer forecast_adapter_context avant l'ajustement")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge doit être fini et strictement positif")
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    adapter = model.forecast_adapter
    horizon, dimension = adapter.proj.out_features, adapter.proj.in_features + 1
    primary = "length_or_missing" if adapter.missing_gate and adapter.missing_proj is None else "length"
    branches = [(primary, adapter.proj)]
    if adapter.missing_proj is not None:
        branches.append(("long_missing", adapter.missing_proj))
    gram = torch.zeros(len(branches), horizon, dimension, dimension, dtype=torch.float64)
    cross = torch.zeros(len(branches), horizon, dimension, dtype=torch.float64)
    counts = torch.zeros(len(branches), horizon, dtype=torch.long)
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
                gates = [context.gate]
                if adapter.missing_proj is not None:
                    gates.append(context.long_missing_gate)
                active = torch.stack([(gate > 0).any(-1) for gate in gates]).any(0)
                examples += target.size(0)
                active_examples += active.sum().item()
                features = features[active].double().cpu()
                features = torch.cat((features, torch.ones(*features.shape[:2], 1, dtype=torch.float64)), dim=-1)
                residual = (target.double() - forecast[..., adapter.median_index].double()).transpose(1, 2)[active].cpu()
                residual = residual.reshape(-1, horizon)
                for branch, gate in enumerate(gates):
                    multiplier = (context.scale[active] * gate[active]).double().cpu()
                    design = (features * multiplier[..., None]).reshape(-1, dimension)
                    active_variables = (gate[active] > 0).reshape(-1).cpu()
                    for step in range(horizon):
                        observed = torch.isfinite(residual[:, step]) & active_variables
                        x, y = design[observed], residual[observed, step]
                        gram[branch, step].add_(x.T @ x)
                        cross[branch, step].add_(x.T @ y)
                        counts[branch, step] += observed.sum()
        if not counts.any():
            raise ValueError("Chaque horizon doit avoir une cible observée avec un historique plus court que forecast_context ou incomplet si la porte de valeurs manquantes est active")
        penalty = torch.eye(dimension, dtype=torch.float64) * ridge
        penalty[-1, -1] = 0
        updates = []
        for index, (name, projection) in enumerate(branches):
            if not counts[index].any():
                continue
            if (counts[index] == 0).any():
                raise ValueError(f"Chaque horizon de la branche {name} doit avoir une cible observée")
            coefficients = torch.linalg.solve(gram[index] / counts[index, :, None, None] + penalty,
                                              (cross[index] / counts[index, :, None]).unsqueeze(-1)).squeeze(-1)
            weight = coefficients[:, :-1].to(device=projection.weight.device, dtype=projection.weight.dtype)
            bias = coefficients[:, -1].to(device=projection.bias.device, dtype=projection.bias.dtype)
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError("Les coefficients ajustés ne sont pas finis dans la précision du modèle")
            updates.append((projection, weight, bias))
        with torch.no_grad():
            for projection, weight, bias in updates:
                projection.weight.copy_(weight)
                projection.bias.copy_(bias)
        return dict(method="ridge", ridge=ridge, examples=examples, active_examples=active_examples,
                    observed_targets_by_horizon=counts.sum(0).tolist(), context=adapter.context, amp=amp,
                    features="relative" if adapter.relative else "absolute",
                    missing_gate=adapter.missing_gate,
                    separate_missing=adapter.missing_proj is not None,
                    branches={name: dict(observed_targets_by_horizon=counts[index].tolist(),
                                         fitted=bool(counts[index].any()))
                              for index, (name, _) in enumerate(branches)},
                    updated_parameters=sum(p.numel() for projection, _, _ in updates for p in projection.parameters()),
                    total_adapter_parameters=sum(p.numel() for p in adapter.parameters()))
    finally:
        hook.remove()
        model.train(was_training)
