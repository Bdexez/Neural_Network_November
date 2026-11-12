"""Mesures pondérées par les cibles observées, et diagnostics des prévisions."""

from contextlib import nullcontext
import math

import torch


def evaluate(model, batches, amp="none", *, calibration=None):
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    if calibration is not None:
        calibration.validate(model.config, amp=amp)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts = {}, {}
    correct = class_count = forecast_count = 0
    absolute = squared = covered = width = 0.0
    horizon_error = torch.zeros(model.config.forecast_horizon, dtype=torch.float64)
    horizon_count = torch.zeros_like(horizon_error)
    interval_covered = torch.zeros_like(horizon_error)
    interval_width = torch.zeros_like(horizon_error)
    calibrated_covered = torch.zeros_like(horizon_error)
    calibrated_width = torch.zeros_like(horizon_error)
    calibrated_score = 0.0
    raw_score = 0.0
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )

    def accumulate(name, value, weight):
        totals[name] = totals.get(name, 0.0) + float(value) * weight
        counts[name] = counts.get(name, 0) + weight

    def by_horizon(values):
        return [total.item() / count.item() if count else None
                for total, count in zip(values, horizon_count)]

    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                if calibration is not None:
                    calibration.validate(model.config, batch)
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
                        interval_covered += ((target >= lower) & (target <= upper) & observed).sum((0, 2)).cpu()
                        interval_width += torch.where(observed, upper - lower, 0).double().sum((0, 2)).cpu()
                        alpha = 1 - (model.config.quantiles[-1] - model.config.quantiles[0])
                        score = upper - lower + 2 / alpha * (
                            (lower - target).clamp_min(0) + (target - upper).clamp_min(0))
                        raw_score += torch.where(observed, score, 0).double().sum().item()
                        if calibration is not None:
                            interval = calibration.interval(output.forecast)
                            low, high = interval.unbind(-1)
                            calibrated_covered += ((target >= low) & (target <= high) & observed).sum((0, 2)).cpu()
                            calibrated_width += torch.where(observed, high - low, 0).double().sum((0, 2)).cpu()
                            score = high - low + 2 / (1 - calibration.coverage) * (
                                (low - target).clamp_min(0) + (target - high).clamp_min(0))
                            calibrated_score += torch.where(observed, score, 0).double().sum().item()
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
            result["forecast_mae_by_horizon"] = by_horizon(horizon_error)
            if len(model.config.quantiles) > 1:
                result["forecast_interval_coverage"] = covered / forecast_count if forecast_count else None
                result["forecast_interval_width"] = width / forecast_count if forecast_count else None
                result["forecast_nominal_coverage"] = model.config.quantiles[-1] - model.config.quantiles[0]
                result["forecast_interval_score"] = raw_score / forecast_count if forecast_count else None
                result["forecast_interval_coverage_by_horizon"] = by_horizon(interval_covered)
                result["forecast_interval_width_by_horizon"] = by_horizon(interval_width)
                if calibration is not None:
                    result["calibrated_interval_coverage"] = calibrated_covered.sum().item() / forecast_count if forecast_count else None
                    result["calibrated_interval_width"] = calibrated_width.sum().item() / forecast_count if forecast_count else None
                    result["calibrated_interval_score"] = calibrated_score / forecast_count if forecast_count else None
                    result["calibrated_nominal_coverage"] = calibration.coverage
                    result["calibrated_interval_coverage_by_horizon"] = by_horizon(calibrated_covered)
                    result["calibrated_interval_width_by_horizon"] = by_horizon(calibrated_width)
        return result
    finally:
        model.train(was_training)
