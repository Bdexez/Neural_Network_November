"""Pondération des pertes supervisées par cibles observées sur une étape."""

import torch


def target_counts(batch, prediction_steps):
    """Compter texte, classes, prévisions, puis chaque horizon de texte auxiliaire."""
    device = next(iter(batch.values())).device
    counts = torch.zeros(3 + max(prediction_steps - 1, 0), dtype=torch.long, device=device)
    labels = batch.get("labels")
    if labels is not None:
        valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
        for distance in range(1, min(prediction_steps + 1, labels.size(1))):
            index = 0 if distance == 1 else distance + 1
            counts[index] = ((labels[:, distance:] != -100) & valid[:, distance:]
                             & valid[:, :-distance]).sum()
    if "class_labels" in batch:
        counts[1] = (batch["class_labels"] != -100).sum()
    if "forecast_targets" in batch:
        counts[2] = torch.isfinite(batch["forecast_targets"]).sum()
    return counts


def observed_objective(output, local_counts, total_counts, config, *, world, accumulation):
    """Reconstituer la perte ; DDP moyenne les gradients entre les rangs."""
    losses = output.losses
    loss = (config.balance_weight * losses["balance"] + config.router_z_weight * losses["router_z"]) / accumulation
    if "alignment" in losses:
        loss = loss + config.alignment_weight * losses["alignment"] / accumulation
    for index, (name, weight) in enumerate((("text", config.text_weight),
                                            ("classification", config.classification_weight),
                                            ("forecast", config.forecast_weight))):
        if name in losses:
            factor = world * local_counts[index] / total_counts[index].clamp_min(1)
            loss = loss + weight * losses[name] * factor
    active_horizons = (total_counts[3:] > 0).sum().clamp_min(1)
    for distance, (head_loss, count) in output.auxiliary_text_losses.items():
        factor = world * count / total_counts[distance + 1].clamp_min(1) / active_horizons
        loss = loss + config.mtp_weight * head_loss * factor
    return loss
