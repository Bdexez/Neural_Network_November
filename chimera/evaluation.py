"""Mesures pondérées par les cibles observées, et diagnostics des prévisions."""

from contextlib import nullcontext
from collections import Counter
import math

import torch

from .routing_metrics import summarize_expert_usage
from .probability_metrics import ProbabilityMetrics
from .series_probability import log_probabilities


def evaluate(model, batches, amp="none", *, calibration=None, series_classifier=None, series_probability=None, series_set=None):
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    if calibration is not None:
        calibration.validate(model.config, amp=amp)
    if series_classifier is not None:
        series_classifier.validate(model.config, amp=amp)
    if series_probability is not None:
        if series_classifier is None:
            raise ValueError("La calibration de probabilités nécessite une tête temporelle")
        series_probability.validate(series_classifier, amp=amp)
    if series_set is not None:
        if series_classifier is None:
            raise ValueError("Les ensembles conformes nécessitent une tête temporelle")
        series_set.validate(series_classifier, series_probability, amp=amp)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts = {}, {}
    auxiliary_totals, auxiliary_counts = {}, {}
    expert_totals, expert_tokens = None, 0
    correct = class_count = forecast_count = 0
    series_class_correct = series_class_count = 0
    series_class_confusion = Counter() if series_classifier is not None else None
    probability_raw = ProbabilityMetrics() if series_probability is not None else None
    probability_calibrated = ProbabilityMetrics() if series_probability is not None else None
    set_examples = set_size = set_singletons = set_labeled = set_covered = 0
    set_labeled_singletons = set_correct_singletons = 0
    set_class_counts = torch.zeros(model.config.num_classes, dtype=torch.long)
    set_class_covered = torch.zeros_like(set_class_counts)
    complete_forecasts = raw_joint_covered = calibrated_joint_covered = 0
    absolute = squared = covered = width = 0.0
    horizon_error = torch.zeros(model.config.forecast_horizon, dtype=torch.float64)
    horizon_count = torch.zeros_like(horizon_error)
    feature_count = torch.zeros(model.config.series_features, dtype=torch.float64)
    feature_error = torch.zeros_like(feature_count)
    cell_count = torch.zeros(model.config.forecast_horizon, model.config.series_features, dtype=torch.float64)
    cell_error = torch.zeros_like(cell_count)
    feature_width = torch.zeros_like(feature_count)
    calibrated_feature_width = torch.zeros_like(feature_count)
    feature_covered = torch.zeros_like(feature_count)
    calibrated_feature_covered = torch.zeros_like(feature_count)
    feature_below = torch.zeros_like(feature_count)
    feature_above = torch.zeros_like(feature_count)
    calibrated_feature_below = torch.zeros_like(feature_count)
    calibrated_feature_above = torch.zeros_like(feature_count)
    cell_covered = torch.zeros_like(cell_count)
    calibrated_cell_covered = torch.zeros_like(cell_count)
    quantile_hits = torch.zeros(len(model.config.quantiles), dtype=torch.float64)
    quantile_pinball = torch.zeros_like(quantile_hits)
    feature_quantile_hits = torch.zeros(model.config.series_features, len(model.config.quantiles), dtype=torch.float64)
    feature_quantile_pinball = torch.zeros_like(feature_quantile_hits)
    interval_covered = torch.zeros_like(horizon_error)
    interval_width = torch.zeros_like(horizon_error)
    calibrated_covered = torch.zeros_like(horizon_error)
    calibrated_width = torch.zeros_like(horizon_error)
    calibrated_score = 0.0
    raw_score = 0.0
    multi_token_count = 0
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )

    def accumulate(name, value, weight):
        totals[name] = totals.get(name, 0.0) + float(value) * weight
        counts[name] = counts.get(name, 0) + weight

    def by_horizon(values):
        return [total.item() / count.item() if count else None
                for total, count in zip(values, horizon_count)]

    def by_feature(values):
        return [total.item() / count.item() if count else None
                for total, count in zip(values, feature_count)]

    def by_cell(values):
        return [[total.item() / count.item() if count else None for total, count in zip(row, counts)]
                for row, counts in zip(values, cell_count)]

    def by_feature_quantile(values):
        return [[total.item() / count.item() if count else None for total in row]
                for row, count in zip(values, feature_count)]

    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                if calibration is not None:
                    calibration.validate(model.config, batch)
                n = next(iter(batch.values())).size(0)
                with autocast():
                    output = model(**batch)
                    series_scores = series_classifier.scores(model, batch) if series_classifier is not None else None
                set_mask = None
                if series_set is not None:
                    probs = (series_probability.probabilities(series_scores) if series_probability
                             else log_probabilities(series_scores).exp())
                    set_mask = series_set.predict(probs, series_scores.argmax(-1))
                    sizes = set_mask.sum(-1)
                    set_examples += sizes.numel()
                    set_size += sizes.sum().item()
                    set_singletons += (sizes == 1).sum().item()
                usage = getattr(output, "expert_usage", None)
                routed = getattr(output, "routed_tokens", None)
                if usage is not None and routed is not None:
                    count = routed.item()
                    weighted = usage.double().cpu() * count
                    expert_totals = weighted if expert_totals is None else expert_totals + weighted
                    expert_tokens += count
                weights = {}
                if "labels" in batch:
                    labels = batch["labels"]
                    valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
                    weights["text"] = ((labels[:, 1:] != -100) & valid[:, 1:] & valid[:, :-1]).sum().item()
                    for distance in range(2, min(model.config.prediction_steps + 1, labels.size(1))):
                        multi_token_count += ((labels[:, distance:] != -100) & valid[:, distance:]
                                              & valid[:, :-distance]).sum().item()
                if "class_labels" in batch:
                    labels = batch["class_labels"]
                    observed = labels != -100
                    weights["classification"] = observed.sum().item()
                    correct += ((output.classification_logits.argmax(-1) == labels) & observed).sum().item()
                    class_count += weights["classification"]
                    if set_mask is not None and observed.any():
                        selected = set_mask[observed]
                        observed_labels = labels[observed]
                        hits = selected.gather(1, observed_labels.unsqueeze(1)).squeeze(1)
                        singletons = selected.sum(-1) == 1
                        set_labeled += observed_labels.numel()
                        set_covered += hits.sum().item()
                        set_labeled_singletons += singletons.sum().item()
                        set_correct_singletons += (hits & singletons).sum().item()
                        set_class_counts += torch.bincount(observed_labels.cpu(), minlength=model.config.num_classes)
                        set_class_covered += torch.bincount(observed_labels[hits].cpu(), minlength=model.config.num_classes)
                    if series_scores is not None:
                        predictions = series_scores.argmax(-1)
                        series_class_correct += ((predictions == labels) & observed).sum().item()
                        series_class_count += observed.sum().item()
                        pairs = labels[observed] * model.config.num_classes + predictions[observed]
                        distinct, frequencies = pairs.unique(return_counts=True)
                        for pair, frequency in zip(distinct.cpu().tolist(), frequencies.cpu().tolist()):
                            series_class_confusion[divmod(pair, model.config.num_classes)] += frequency
                        if series_probability is not None and observed.any():
                            probability_raw.update(log_probabilities(series_scores[observed]), labels[observed])
                            probability_calibrated.update(series_probability.log_probabilities(series_scores[observed]),
                                                          labels[observed])
                if "forecast_targets" in batch:
                    target = batch["forecast_targets"]
                    observed = torch.isfinite(target)
                    weights["forecast"] = observed.sum().item()
                    forecast_count += weights["forecast"]
                    target = target.double()
                    predictions = output.forecast.double()
                    median = predictions[..., model.config.quantiles.index(0.5)]
                    error = torch.where(observed, median - target, 0)
                    absolute += error.abs().sum().item()
                    squared += error.square().sum().item()
                    horizon_error += error.abs().sum((0, 2)).cpu()
                    horizon_count += observed.sum((0, 2)).cpu()
                    feature_count += observed.sum((0, 1)).cpu()
                    feature_error += error.abs().sum((0, 1)).cpu()
                    cell_count += observed.sum(0).cpu()
                    cell_error += error.abs().sum(0).cpu()
                    levels = predictions.new_tensor(model.config.quantiles)
                    residual = torch.where(observed.unsqueeze(-1), target.unsqueeze(-1) - predictions, 0)
                    hits = (target.unsqueeze(-1) <= predictions) & observed.unsqueeze(-1)
                    pinball = torch.maximum(levels * residual, (levels - 1) * residual)
                    quantile_hits += hits.sum((0, 1, 2)).cpu()
                    quantile_pinball += pinball.sum((0, 1, 2)).cpu()
                    feature_quantile_hits += hits.sum((0, 1)).cpu()
                    feature_quantile_pinball += pinball.sum((0, 1)).cpu()
                    if len(model.config.quantiles) > 1:
                        lower, upper = predictions[..., 0], predictions[..., -1]
                        complete = observed.flatten(1).all(-1)
                        complete_forecasts += complete.sum().item()
                        inside = (target >= lower) & (target <= upper)
                        raw_joint_covered += (inside.flatten(1).all(-1) & complete).sum().item()
                        covered += ((target >= lower) & (target <= upper) & observed).sum().item()
                        width += torch.where(observed, upper - lower, 0).double().sum().item()
                        interval_covered += ((target >= lower) & (target <= upper) & observed).sum((0, 2)).cpu()
                        feature_covered += (inside & observed).sum((0, 1)).cpu()
                        cell_covered += (inside & observed).sum(0).cpu()
                        feature_below += ((target < lower) & observed).sum((0, 1)).cpu()
                        feature_above += ((target > upper) & observed).sum((0, 1)).cpu()
                        interval_width += torch.where(observed, upper - lower, 0).double().sum((0, 2)).cpu()
                        feature_width += torch.where(observed, upper - lower, 0).sum((0, 1)).cpu()
                        alpha = 1 - (model.config.quantiles[-1] - model.config.quantiles[0])
                        score = upper - lower + 2 / alpha * (
                            (lower - target).clamp_min(0) + (target - upper).clamp_min(0))
                        raw_score += torch.where(observed, score, 0).double().sum().item()
                        if calibration is not None:
                            interval = calibration.interval(output.forecast)
                            low, high = interval.double().unbind(-1)
                            inside = (target >= low) & (target <= high)
                            calibrated_joint_covered += (inside.flatten(1).all(-1) & complete).sum().item()
                            calibrated_covered += ((target >= low) & (target <= high) & observed).sum((0, 2)).cpu()
                            calibrated_feature_covered += (inside & observed).sum((0, 1)).cpu()
                            calibrated_cell_covered += (inside & observed).sum(0).cpu()
                            calibrated_feature_below += ((target < low) & observed).sum((0, 1)).cpu()
                            calibrated_feature_above += ((target > high) & observed).sum((0, 1)).cpu()
                            calibrated_width += torch.where(observed, high - low, 0).double().sum((0, 2)).cpu()
                            calibrated_feature_width += torch.where(observed, high - low, 0).sum((0, 1)).cpu()
                            if calibration.scope == "marginal":
                                alpha = 1 - calibration.coverage
                                if calibration.lower_tail_fraction in (None, 0.5):
                                    score = high - low + 2 / alpha * (
                                        (low - target).clamp_min(0) + (target - high).clamp_min(0))
                                else:
                                    fraction = calibration.lower_tail_fraction
                                    score = (high - low + (low - target).clamp_min(0) / (alpha * fraction)
                                             + (target - high).clamp_min(0) / (alpha * (1 - fraction)))
                                calibrated_score += torch.where(observed, score, 0).double().sum().item()
                if output.loss is not None:
                    accumulate("loss", output.loss.item(), n)
                for name, value in output.losses.items():
                    if name == "multi_token" and getattr(output, "auxiliary_text_losses", None) is not None:
                        for distance, (head_loss, count) in output.auxiliary_text_losses.items():
                            count = count.item()
                            auxiliary_totals[distance] = auxiliary_totals.get(distance, 0.0) + head_loss.item() * count
                            auxiliary_counts[distance] = auxiliary_counts.get(distance, 0) + count
                        continue
                    accumulate(name, value.item(), weights.get(name, n))
        result = {name: total / counts[name] if counts[name] else None for name, total in totals.items()}
        if expert_tokens:
            result.update(summarize_expert_usage(expert_totals, expert_tokens))
        if auxiliary_counts:
            per_horizon = [auxiliary_totals[distance] / count for distance, count in auxiliary_counts.items() if count]
            result["multi_token"] = sum(per_horizon) / len(per_horizon) if per_horizon else None
        if "text" in counts:
            result["text_tokens"] = counts["text"]
            if model.config.prediction_steps > 1:
                result["multi_token_targets"] = multi_token_count
                result.setdefault("multi_token", None)
            result["byte_perplexity"] = math.exp(min(result["text"], 20)) if counts["text"] else None
        if "classification" in counts:
            result["classified_examples"] = class_count
            result["accuracy"] = correct / class_count if class_count else None
        if series_classifier is not None:
            result["series_classifier_examples"] = series_class_count
            result["series_classifier_accuracy"] = series_class_correct / series_class_count if series_class_count else None
            result["series_classifier_confusion"] = [dict(actual=actual, predicted=predicted, count=count)
                                                      for (actual, predicted), count in sorted(series_class_confusion.items())]
        if series_probability is not None:
            result["series_classifier_probability_metrics"] = dict(
                uncalibrated=probability_raw.result(), calibrated=probability_calibrated.result())
            result["series_classifier_inverse_temperature"] = series_probability.inverse_temperature
        if series_set is not None:
            result.update(series_prediction_set_method=series_set.method,
                          series_prediction_set_scope=series_set.scope,
                          series_prediction_set_nominal_coverage=series_set.coverage,
                          series_prediction_set_examples=set_examples,
                          series_prediction_set_labeled_examples=set_labeled,
                          series_prediction_set_coverage=set_covered / set_labeled if set_labeled else None,
                          series_prediction_set_mean_size=set_size / set_examples if set_examples else None,
                          series_prediction_set_singleton_rate=set_singletons / set_examples if set_examples else None,
                          series_prediction_set_singleton_accuracy=(set_correct_singletons / set_labeled_singletons
                                                                    if set_labeled_singletons else None),
                          series_prediction_set_counts_by_class=set_class_counts.tolist(),
                          series_prediction_set_coverage_by_class=[hits.item() / count.item() if count else None
                              for hits, count in zip(set_class_covered, set_class_counts)])
        if "loss" in result:
            # Chaque tâche est d'abord agrégée selon son nombre de cibles ; les
            # termes auxiliaires conservent leur agrégation par lot existante.
            supervised = ("text", "classification", "forecast", "multi_token")
            terms = (("text", "text_weight"), ("classification", "classification_weight"),
                     ("forecast", "forecast_weight"), ("multi_token", "mtp_weight"),
                     ("balance", "balance_weight"), ("router_z", "router_z_weight"),
                     ("alignment", "alignment_weight"))
            result["observed_loss"] = (
                sum(getattr(model.config, weight) * result[name] for name, weight in terms
                    if result.get(name) is not None)
                if any(result.get(name) is not None for name in supervised) else None)
        if "forecast" in counts:
            result["forecast_observations"] = forecast_count
            result["forecast_observations_by_horizon"] = horizon_count.long().tolist()
            result["forecast_observations_by_feature"] = feature_count.long().tolist()
            result["forecast_observations_by_horizon_feature"] = cell_count.long().tolist()
            result["forecast_mae"] = absolute / forecast_count if forecast_count else None
            result["forecast_rmse"] = math.sqrt(squared / forecast_count) if forecast_count else None
            result["forecast_mae_by_horizon"] = by_horizon(horizon_error)
            result["forecast_mae_by_feature"] = by_feature(feature_error)
            result["forecast_mae_by_horizon_feature"] = by_cell(cell_error)
            result["forecast_quantile_levels"] = list(model.config.quantiles)
            result["forecast_quantile_coverage"] = [value.item() / forecast_count if forecast_count else None
                                                    for value in quantile_hits]
            result["forecast_pinball_by_quantile"] = [value.item() / forecast_count if forecast_count else None
                                                      for value in quantile_pinball]
            result["forecast_quantile_coverage_by_feature"] = by_feature_quantile(feature_quantile_hits)
            result["forecast_pinball_by_feature_quantile"] = by_feature_quantile(feature_quantile_pinball)
            if len(model.config.quantiles) > 1:
                result["forecast_complete_examples"] = complete_forecasts
                result["forecast_joint_interval_coverage"] = (
                    raw_joint_covered / complete_forecasts if complete_forecasts else None)
                result["forecast_interval_coverage"] = covered / forecast_count if forecast_count else None
                result["forecast_interval_width"] = width / forecast_count if forecast_count else None
                result["forecast_nominal_coverage"] = model.config.quantiles[-1] - model.config.quantiles[0]
                result["forecast_interval_score"] = raw_score / forecast_count if forecast_count else None
                result["forecast_interval_coverage_by_horizon"] = by_horizon(interval_covered)
                result["forecast_interval_coverage_by_feature"] = by_feature(feature_covered)
                result["forecast_interval_coverage_by_horizon_feature"] = by_cell(cell_covered)
                result["forecast_interval_below_by_feature"] = by_feature(feature_below)
                result["forecast_interval_above_by_feature"] = by_feature(feature_above)
                result["forecast_interval_width_by_horizon"] = by_horizon(interval_width)
                result["forecast_interval_width_by_feature"] = by_feature(feature_width)
                if calibration is not None:
                    result["calibrated_interval_coverage"] = calibrated_covered.sum().item() / forecast_count if forecast_count else None
                    result["calibrated_interval_width"] = calibrated_width.sum().item() / forecast_count if forecast_count else None
                    result["calibrated_interval_score"] = (
                        calibrated_score / forecast_count if forecast_count and calibration.scope == "marginal" else None)
                    result["calibrated_nominal_coverage"] = calibration.coverage
                    result["calibrated_coverage_scope"] = calibration.scope
                    result["calibrated_score_scale"] = calibration.score_scale
                    if calibration.lower_tail_fraction is not None:
                        result["calibrated_lower_tail_fraction"] = calibration.lower_tail_fraction
                    if calibration.score_scale == "width":
                        result["calibrated_scale_floor"] = calibration.scale_floor
                    result["calibrated_joint_interval_coverage"] = (
                        calibrated_joint_covered / complete_forecasts if complete_forecasts else None)
                    result["calibrated_interval_coverage_by_horizon"] = by_horizon(calibrated_covered)
                    result["calibrated_interval_coverage_by_feature"] = by_feature(calibrated_feature_covered)
                    result["calibrated_interval_coverage_by_horizon_feature"] = by_cell(calibrated_cell_covered)
                    result["calibrated_interval_below_by_feature"] = by_feature(calibrated_feature_below)
                    result["calibrated_interval_above_by_feature"] = by_feature(calibrated_feature_above)
                    result["calibrated_interval_width_by_horizon"] = by_horizon(calibrated_width)
                    result["calibrated_interval_width_by_feature"] = by_feature(calibrated_feature_width)
        return result
    finally:
        model.train(was_training)
