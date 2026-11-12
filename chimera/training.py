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
from .augmentation import SeriesAugmentation
from .data import JointDataset, collate_records, synthetic_batch
from .model import Chimera
from .modalities import MODALITIES, TARGETS, balanced_modalities, keep_modalities
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
    names = [name for name in MODALITIES if name in batch]
    kept = [name for name in names if random.random() >= probability]
    if not kept:
        kept = [random.choice(names)]
    result = keep_modalities(batch, kept)
    # Un corpus ne contenant que du texte doit conserver ses cibles.
    if not any(name in result for name in TARGETS):
        return batch
    return result


def step_optimizer(model, optimizer, scaler, grad_clip):
    """Laisser GradScaler ignorer un overflow, au lieu d'arrêter l'entraînement."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
    return float(norm) if torch.isfinite(norm) else None, skipped


def train(args, config):
    if min(args.steps, args.batch_size, args.accumulation, args.save_every, args.log_every, args.eval_every) < 1:
        raise ValueError("Les nombres d'étapes, de lots et les fréquences doivent être > 0")
    for name in ("lr", "grad_clip", "weight_decay", "min_delta"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ("lr", "grad_clip") and value == 0):
            raise ValueError(f"Valeur invalide pour {name}")
    if args.warmup < 0 or args.patience < 0 or args.workers < 0:
        raise ValueError("warmup, patience et workers doivent être >= 0")
    if not 0 <= args.modality_dropout < 1:
        raise ValueError("modality-dropout doit appartenir à [0, 1[")
    if args.patience and not args.val_data:
        raise ValueError("L'arrêt anticipé nécessite un jeu indépendant fourni par --val-data")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(args.device)
    if args.amp == "fp16" and device.type != "cuda":
        raise ValueError("fp16 nécessite CUDA ; utiliser bf16 ou none sur CPU")
    if world > 1:
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_everything(args.seed)
    initialization = None
    if args.resume or args.init_from:
        model, source = load_checkpoint(args.resume or args.init_from, device)
        config = model.config
        if args.resume:
            if "optimizer" not in source:
                raise ValueError("Une reprise nécessite le checkpoint complet latest.pt avec son optimiseur")
            payload, previous, start = source, source.get("trainer", {}), source["step"]
        else:
            initialization = dict(checkpoint=str(Path(args.init_from).resolve()), step=source["step"],
                                  seed=source.get("trainer", {}).get("seed"))
            payload, previous, start = {}, {}, 0
        del source
    else:
        model, payload, previous, start = Chimera(config).to(device), {}, {}, 0
    previous_policy = previous.get("modality_policy", "dropout")
    args.modality_policy = args.modality_policy or previous_policy
    if args.resume and args.modality_policy != previous_policy:
        raise ValueError(f"Reprise : modality_policy diffère du checkpoint ({previous_policy!r})")
    augmentation_options = {}
    previous_augmentation = previous.get("series_augmentation", {})
    for name, default in SeriesAugmentation().to_dict().items():
        saved = previous_augmentation.get(name, default)
        option = f"series_{name}"
        value = getattr(args, option)
        value = saved if value is None else value
        if args.resume and value != saved:
            raise ValueError(f"Reprise : {option} diffère du checkpoint ({saved!r})")
        augmentation_options[name] = value
    augment_series = SeriesAugmentation(**augmentation_options)
    args.shuffle = (previous.get("shuffle", False) if args.resume else True) if args.shuffle is None else args.shuffle
    keys = ("steps", "batch_size", "accumulation", "seed", "lr", "warmup", "weight_decay",
            "amp", "data", "modality_dropout", "modality_policy", "shuffle", "val_data", "eval_every", "patience", "min_delta")
    trainer = {key: getattr(args, key) for key in keys}
    trainer["initialization"] = previous.get("initialization", initialization)
    trainer["series_augmentation"] = augment_series.to_dict()
    trainer["world_size"] = world
    dataset = JointDataset(args.data, config) if args.data else None
    validation = JointDataset(args.val_data, config) if args.val_data else None
    trainer["data_fingerprint"] = dataset.fingerprint if dataset is not None else None
    trainer["validation_fingerprint"] = validation.fingerprint if validation is not None else None
    if args.resume:
        for name, current in trainer.items():
            if name in previous and previous[name] != current:
                raise ValueError(f"Reprise : {name} diffère du checkpoint ({previous[name]!r})")
    state = dict(previous.get("validation_state", {"best_loss": float("inf"), "bad_evaluations": 0, "best_step": None}))
    trainer["validation_state"] = state
    end_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    if end_step < start or end_step < 1:
        raise ValueError("stop-after doit être >= à l'étape reprise et > 0")
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
    del payload
    wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                                      find_unused_parameters=True) if world > 1 else model)
    sampler = loader = iterator = None
    epoch = 0
    if dataset is not None:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=args.seed, shuffle=args.shuffle)
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records,
                            sampler=sampler, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed))
        epoch, skip = divmod(start * args.accumulation, len(loader))
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
    output_dir = Path(args.output)
    val = None
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "config.json")
        val = (DataLoader(validation, batch_size=args.batch_size, collate_fn=collate_records,
                          num_workers=args.workers, generator=torch.Generator().manual_seed(args.seed + 1))
               if validation is not None else [synthetic_batch(config, batch_size=args.batch_size,
                                                               seed=1000000 + args.seed)])
    autocast = (lambda: nullcontext()) if args.amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if args.amp == "bf16" else torch.float16)
    )
    model.train()
    started, last_step, last_evaluated = time.monotonic(), start, None
    stopped_early = bool(args.patience and state["bad_evaluations"] >= args.patience)
    if stopped_early:
        end_step = start
    report = None
    for step in range(start, end_step):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup) * group["lr_scale"]
        metrics, expert_usage, modality_batches, series_statistics = {}, None, {}, {}
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
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {key: value.to(device) for key, value in batch.items()}
            index = (step * args.accumulation + micro) * world + rank
            batch = augment_series(batch, index=index, seed=args.seed)
            if args.modality_policy == "balanced":
                batch = balanced_modalities(batch, index, args.seed)
            elif args.modality_dropout:
                batch = drop_modalities(batch, args.modality_dropout)
            signature = "+".join(name for name in MODALITIES if name in batch)
            modality_batches[signature] = modality_batches.get(signature, 0) + 1
            if rank == 0 and "series" in batch:
                series = batch["series"]
                valid = batch.get("series_mask", torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)).bool()
                valid_steps = valid.sum().item()
                statistics = dict(examples=series.size(0), valid_steps=valid_steps,
                                  possible_values=valid_steps * series.size(-1),
                                  observed_values=(torch.isfinite(series) & valid.unsqueeze(-1)).sum().item())
                for name, value in statistics.items():
                    series_statistics[name] = series_statistics.get(name, 0) + value
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
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
        norm, skipped = step_optimizer(model, optimizer, scaler, args.grad_clip)
        last_step = step + 1
        if rank == 0:
            row = dict(step=last_step, **metrics, grad_norm=norm, optimizer_step_skipped=skipped,
                       lr=optimizer.param_groups[0]["lr"], elapsed_seconds=round(time.monotonic() - started, 3),
                       expert_usage=expert_usage.cpu().tolist(), modality_batches=modality_batches,
                       series_statistics=series_statistics)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == start or last_step % args.log_every == 0 or last_step == end_step:
                print(json.dumps({key: value for key, value in row.items() if key != "expert_usage"}), flush=True)
        # Ne pas ajouter une validation de contrôle à un simple --stop-after :
        # sa reprise doit conserver le même calendrier d'arrêt anticipé.
        validation_event = last_step % args.eval_every == 0 or last_step == args.steps
        if validation_event:
            if rank == 0:
                report = evaluate(model, val, args.amp)
                if "loss" not in report or not math.isfinite(report["loss"]):
                    raise ValueError("Le jeu de validation doit fournir une perte supervisée finie")
                last_evaluated = last_step
                improved = report["loss"] < state["best_loss"]
                significant = report["loss"] < state.get("reference_loss", state["best_loss"]) - args.min_delta
                if improved:
                    state.update(best_loss=report["loss"], best_step=last_step)
                if significant:
                    state.update(reference_loss=report["loss"], bad_evaluations=0)
                else:
                    state["bad_evaluations"] += 1
                if improved:
                    save_checkpoint(output_dir / "best.pt", model, optimizer=optimizer, scaler=scaler,
                                    step=last_step, trainer=trainer)
                stopped_early = bool(args.patience and state["bad_evaluations"] >= args.patience)
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=last_step, improved=improved, **report), allow_nan=False) + "\n")
                print(json.dumps({"validation_step": last_step, "loss": report["loss"],
                                  "best_step": state["best_step"]}), flush=True)
            if world > 1:
                decision = torch.tensor(int(stopped_early), device=device)
                dist.broadcast(decision, src=0)
                stopped_early = bool(decision.item())
        if rank == 0 and (last_step % args.save_every == 0 or last_step == end_step or stopped_early):
            save_checkpoint(output_dir / "latest.pt", model, optimizer=optimizer, scaler=scaler,
                            step=last_step, trainer=trainer)
        if stopped_early:
            break
    if rank == 0:
        if last_evaluated != last_step:
            report = evaluate(model, val, args.amp)
        report = dict(report, step=last_step, best_step=state["best_step"], stopped_early=stopped_early)
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"evaluation": report}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
