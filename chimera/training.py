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
from .evaluation import evaluate
from .modules import RotaryEmbedding


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


def load_checkpoint(path, device="cpu", *, mmap=True):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if payload.get("format_version") != 1:
        raise ValueError("Version de checkpoint inconnue")
    with torch.device("meta"):
        model = Chimera(ChimeraConfig.from_dict(payload["config"]))
    model.load_state_dict(payload["model"], strict=True, assign=True)
    # assign remplace les Parameters ; rétablir explicitement le poids partagé.
    model.lm_head.weight = model.embedding.weight
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = module.frequencies(device="cpu")
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




def train(args, config):
    if args.steps < 1 or args.batch_size < 1 or args.accumulation < 1:
        raise ValueError("steps, batch-size et accumulation doivent être > 0")
    if args.lr <= 0 or not math.isfinite(args.lr) or args.grad_clip <= 0:
        raise ValueError("lr et grad-clip doivent être finis et > 0")
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("modality-dropout doit appartenir à [0, 1[")
    if args.warmup < 0 or args.weight_decay < 0 or args.save_every < 1 or args.log_every < 1:
        raise ValueError("warmup/weight-decay doivent être >= 0 et save-every > 0")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device)
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    if args.amp == "fp16" and device.type != "cuda":
        raise ValueError("fp16 nécessite CUDA ; utiliser bf16 ou none sur CPU")
    seed_everything(args.seed)
    if args.resume:
        model, payload = load_checkpoint(args.resume, device)
        if "optimizer" not in payload:
            raise ValueError("Une reprise nécessite le checkpoint complet latest.pt avec son optimiseur")
        config = model.config
        start = payload["step"]
        previous = payload.get("trainer", {})
        # Une reprise exacte conserve l'ordonnancement et le flux des données.
        for name in ("steps", "batch_size", "accumulation", "seed", "lr", "warmup",
                     "weight_decay", "amp", "data", "world_size", "modality_dropout"):
            current = world if name == "world_size" else getattr(args, name)
            if name in previous and previous[name] != current:
                raise ValueError(f"Reprise : {name} diffère du checkpoint ({previous[name]!r})")
    else:
        model, payload, start = Chimera(config).to(device), {}, 0
    temporal_parameters = list(model.temporal.forecast_skip.parameters())
    temporal_ids = {id(p) for p in temporal_parameters}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in temporal_ids], "lr_scale": 1.0},
        {"params": temporal_parameters, "lr_scale": 10.0},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    if args.resume and world == 1:
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                      find_unused_parameters=True)
               if world > 1 else model)
    sampler = loader = iterator = None
    if args.data:
        dataset = JointDataset(args.data, config)
        sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed)
                   if world > 1 else None)
        # Un générateur dédié isole l'itération du DataLoader du RNG du dropout.
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            sampler=sampler, shuffle=False,
                            generator=torch.Generator().manual_seed(args.seed))
        epoch, skip = divmod(start * args.accumulation, len(loader))
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
    else:
        epoch = 0
    output_dir = Path(args.output)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "config.json")
    autocast = (lambda: nullcontext()) if args.amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)
    )
    trainer = {key: getattr(args, key) for key in (
        "steps", "batch_size", "accumulation", "seed", "lr", "warmup", "weight_decay", "amp", "data",
        "modality_dropout"
    )}
    trainer["world_size"] = world
    model.train()
    started = time.monotonic()
    end_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    if end_step < start or end_step < 1:
        raise ValueError("stop-after doit être >= à l'étape reprise et > 0")
    for step in range(start, end_step):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup) * group["lr_scale"]
        metrics, expert_usage = {}, None
        for micro in range(args.accumulation):
            if iterator is None:
                batch = synthetic_batch(config, batch_size=args.batch_size,
                                        seed=args.seed + (step * args.accumulation + micro) * world + rank,
                                        device=device)
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(device) for key, value in batch.items()}
            if args.modality_dropout:
                batch = drop_modalities(batch, args.modality_dropout)
            sync = (wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext())
            with sync, autocast():
                result = wrapped(**batch)
                if result.loss is None:
                    raise ValueError("Aucune cible d'entraînement dans le batch")
                loss = result.loss / args.accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perte non finie à l'étape {step + 1}")
            scaler.scale(loss).backward()
            metrics["loss"] = metrics.get("loss", 0) + result.loss.item() / args.accumulation
            for key, value in result.losses.items():
                metrics[key] = metrics.get(key, 0) + value.detach().item() / args.accumulation
            expert_usage = result.expert_usage
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if rank == 0:
            row = dict(step=step + 1, **metrics, grad_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                       elapsed_seconds=round(time.monotonic() - started, 3),
                       expert_usage=expert_usage.cpu().tolist())
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if step == start or (step + 1) % args.log_every == 0 or step + 1 == end_step:
                print(json.dumps({key: value for key, value in row.items() if key != "expert_usage"}), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == end_step:
                save_checkpoint(output_dir / "latest.pt", model, optimizer=optimizer, scaler=scaler,
                                step=step + 1, trainer=trainer)
    if rank == 0:
        if args.val_data:
            val = DataLoader(JointDataset(args.val_data, config), batch_size=args.batch_size,
                             collate_fn=collate_records)
        else:
            # Seed disjoint : ceci reste un test synthétique, pas un benchmark réel.
            val = [synthetic_batch(config, batch_size=args.batch_size, seed=1000000 + args.seed)]
        report = evaluate(model, val, args.amp)
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"evaluation": report}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
