"""Entraînement conjoint, AMP, accumulation, DDP et sauvegarde atomique."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import ChimeraConfig
from .data import JointDataset, collate_records, synthetic_batch
from .model import Chimera


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, *, optimizer=None, scaler=None, step=0, trainer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "config": model.config.to_dict(), "model": model.state_dict(),
        "step": step, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "trainer": trainer or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Version de checkpoint inconnue")
    model = Chimera(ChimeraConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def learning_rate(step, total_steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup - 1, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1))))


def drop_modalities(batch, probability):
    """Apprendre aussi avec des modalités absentes, sans supprimer toutes les entrées."""
    names = [name for name in ("input_ids", "images", "series") if name in batch]
    kept = [name for name in names if random.random() >= probability]
    if not kept:
        kept = [random.choice(names)]
    result = batch.copy()
    for name in set(names) - set(kept):
        del result[name]
        if name == "input_ids":
            result.pop("attention_mask", None)
            result.pop("labels", None)
        elif name == "series":
            result.pop("series_mask", None)
    # Un corpus ne contenant que du texte doit conserver ses cibles.
    if not any(name in result for name in ("labels", "class_labels", "forecast_targets")):
        return batch
    return result


def evaluate(model, batches, amp="none"):
    was_training = model.training
    model.eval()
    totals, count = {}, 0
    device = next(model.parameters()).device
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )
    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast():
                    output = model(**batch)
                n = next(iter(batch.values())).size(0)
                if output.loss is not None:
                    totals["loss"] = totals.get("loss", 0) + output.loss.item() * n
                for name, value in output.losses.items():
                    totals[name] = totals.get(name, 0) + value.item() * n
                if "labels" in batch and "text" in output.losses:
                    valid = batch.get("attention_mask", torch.ones_like(batch["labels"], dtype=torch.bool)).bool()
                    targets = batch["labels"][:, 1:]
                    tokens = ((targets != -100) & valid[:, 1:] & valid[:, :-1]).sum().item()
                    totals["text_token_loss"] = totals.get("text_token_loss", 0) + output.losses["text"].item() * tokens
                    totals["text_token_count"] = totals.get("text_token_count", 0) + tokens
                if "class_labels" in batch:
                    labels = batch["class_labels"]
                    valid = labels != -100
                    totals["class_correct"] = totals.get("class_correct", 0) + (
                        (output.classification_logits.argmax(-1) == labels) & valid).sum().item()
                    totals["class_count"] = totals.get("class_count", 0) + valid.sum().item()
                if "forecast_targets" in batch:
                    target = batch["forecast_targets"]
                    observed = torch.isfinite(target)
                    median = output.forecast[..., model.config.quantiles.index(0.5)]
                    error = torch.where(observed, (median - target).abs(), 0)
                    totals["forecast_abs_error"] = totals.get("forecast_abs_error", 0) + error.sum().item()
                    totals["forecast_count"] = totals.get("forecast_count", 0) + observed.sum().item()
                count += n
        result = {key: value / max(count, 1) for key, value in totals.items()
                  if key not in ("class_correct", "class_count", "forecast_abs_error", "forecast_count",
                                 "text_token_loss", "text_token_count")}
        if "text_token_count" in totals:
            result["text"] = totals["text_token_loss"] / max(totals["text_token_count"], 1)
        if "class_count" in totals:
            result["accuracy"] = totals["class_correct"] / max(totals["class_count"], 1)
        if "forecast_count" in totals:
            result["forecast_mae"] = totals["forecast_abs_error"] / max(totals["forecast_count"], 1)
        if "text" in result:
            result["byte_perplexity"] = math.exp(min(result["text"], 20))
        return result
    finally:
        model.train(was_training)
