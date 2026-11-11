"""Mesures pondérées par les cibles observées, et diagnostics des prévisions."""

from contextlib import nullcontext
import math

import torch


def evaluate(model, batches, amp="none"):
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts = {}, {}
    correct = class_count = forecast_count = 0
    absolute = squared = covered = width = 0.0
    horizon_error = torch.zeros(model.config.forecast_horizon, dtype=torch.float64)
    horizon_count = torch.zeros_like(horizon_error)
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )

    def accumulate(name, value, weight):
        totals[name] = totals.get(name, 0.0) + float(value) * weight
        counts[name] = counts.get(name, 0) + weight

    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                n = next(iter(batch.values())).size(0)
                with autocast():
                    output = model(**batch)
                weights = {}
                if "labels" in batch:
                    labels = batch["labels"]
                    valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
                    weights["text"] = ((labels[:, 1:] != -100) & valid[:, 1:] & valid[:, :-1]).sum().item()
                if "class_labels" in batch:
                    labels = batch["class_labels"]
                    observed = labels != -100
                    weights["classification"] = observed.sum().item()
                    correct += ((output.classification_logits.argmax(-1) == labels) & observed).sum().item()
                    class_count += weights["classification"]
                if "forecast_targets" in batch:
                    target = batch["forecast_targets"]
                    observed = torch.isfinite(target)
                    weights["forecast"] = observed.sum().item()
                    forecast_count += weights["forecast"]
                    median = output.forecast[..., model.config.quantiles.index(0.5)]
                    error = torch.where(observed, median - target, 0).double()
                    absolute += error.abs().sum().item()
                    squared += error.square().sum().item()
                    horizon_error += error.abs().sum((0, 2)).cpu()
                    horizon_count += observed.sum((0, 2)).cpu()
                    if len(model.config.quantiles) > 1:
                        lower, upper = output.forecast[..., 0], output.forecast[..., -1]
                        covered += ((target >= lower) & (target <= upper) & observed).sum().item()
                        width += torch.where(observed, upper - lower, 0).double().sum().item()
                if output.loss is not None:
                    accumulate("loss", output.loss.item(), n)
                for name, value in output.losses.items():
                    accumulate(name, value.item(), weights.get(name, n))
        result = {name: total / max(counts[name], 1) for name, total in totals.items()}
        if "text" in counts:
            result["text_tokens"] = counts["text"]
            result["byte_perplexity"] = math.exp(min(result["text"], 20)) if counts["text"] else None
        if "classification" in counts:
            result["classified_examples"] = class_count
            result["accuracy"] = correct / class_count if class_count else None
        if "forecast" in counts:
            result["forecast_observations"] = forecast_count
            result["forecast_mae"] = absolute / forecast_count if forecast_count else None
            result["forecast_rmse"] = math.sqrt(squared / forecast_count) if forecast_count else None
            result["forecast_mae_by_horizon"] = [
                error.item() / count.item() if count else None
                for error, count in zip(horizon_error, horizon_count)
            ]
            if len(model.config.quantiles) > 1:
                result["forecast_interval_coverage"] = covered / forecast_count if forecast_count else None
                result["forecast_interval_width"] = width / forecast_count if forecast_count else None
                result["forecast_nominal_coverage"] = model.config.quantiles[-1] - model.config.quantiles[0]
        return result
    finally:
        model.train(was_training)
