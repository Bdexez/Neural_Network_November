"""Entraînement conjoint, AMP, accumulation, DDP et sauvegarde atomique."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import warnings

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import ChimeraConfig
from .augmentation import SeriesAffineAugmentation, SeriesAugmentation
from .data import JointDataset, collate_records, synthetic_batch
from .model import Chimera
from .modalities import MODALITIES, balanced_modalities, has_observed_targets, keep_modalities
from .observed_objective import observed_objective, target_counts
from .routing_metrics import summarize_expert_usage
from .evaluation import evaluate
from .modules import RotaryEmbedding


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state(device):
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def collect_rng_states(device, world, rank):
    """Tous les rangs participent ; seul le rang 0 reçoit la liste ordonnée."""
    state = capture_rng_state(device)
    if world == 1:
        return None
    states = [None] * world if rank == 0 else None
    dist.gather_object(state, states, dst=0)
    return states


def save_checkpoint(path, model, *, optimizer=None, scaler=None, step=0, trainer=None, rank_rng=None):
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
    if rank_rng is not None:
        payload["rank_rng"] = rank_rng
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu", *, mmap=True, overrides=None):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if payload.get("format_version") != 1:
        raise ValueError("Version de checkpoint inconnue")
    original_config = ChimeraConfig.from_dict(payload["config"])
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Les changements de configuration doivent être un objet JSON")
    config = ChimeraConfig.from_dict(original_config.to_dict() | (overrides or {}))
    with torch.device("meta"):
        model = Chimera(config)
    state = dict(payload["model"])
    statistics_key = "temporal.statistics_proj.weight"
    if config.temporal_statistics and not original_config.temporal_statistics:
        if statistics_key in state:
            raise ValueError("La projection statistique contredit la configuration du checkpoint")
        # Ce seul nouveau paramètre est initialisé à zéro ; toute autre
        # incompatibilité du checkpoint reste une erreur de chargement strict.
        state[statistics_key] = torch.zeros(model.temporal.statistics_proj.weight.shape,
                                            dtype=state["embedding.weight"].dtype, device="cpu")
    elif original_config.temporal_statistics and not config.temporal_statistics:
        state.pop(statistics_key)
    adapter_keys = ("forecast_adapter.proj.weight", "forecast_adapter.proj.bias")
    missing_keys = ("forecast_adapter.missing_proj.weight", "forecast_adapter.missing_proj.bias")
    if (config.forecast_adapter_context != original_config.forecast_adapter_context
            or config.forecast_adapter_relative != original_config.forecast_adapter_relative
            or config.forecast_adapter_missing_gate != original_config.forecast_adapter_missing_gate
            or config.forecast_adapter_separate_missing != original_config.forecast_adapter_separate_missing):
        same_length_branch = (config.forecast_adapter_context == original_config.forecast_adapter_context > 0
                              and config.forecast_adapter_relative == original_config.forecast_adapter_relative
                              and (not original_config.forecast_adapter_missing_gate
                                   or original_config.forecast_adapter_separate_missing)
                              and (not config.forecast_adapter_missing_gate or config.forecast_adapter_separate_missing))
        old_keys = (*adapter_keys, *(missing_keys if original_config.forecast_adapter_separate_missing else ()))
        new_keys = (*adapter_keys, *(missing_keys if config.forecast_adapter_separate_missing else ()))
        if original_config.forecast_adapter_context:
            if not original_config.forecast_adapter_separate_missing and any(key in state for key in missing_keys):
                raise ValueError("L'adaptateur contredit la configuration du checkpoint")
            for key in old_keys:
                if not same_length_branch or key in missing_keys:
                    state.pop(key)  # Refuser aussi un ancien adaptateur incomplet.
        elif any(key in state for key in (*adapter_keys, *missing_keys)):
            raise ValueError("L'adaptateur contredit la configuration du checkpoint")
        if config.forecast_adapter_context:
            expected = model.forecast_adapter.state_dict()
            for key in new_keys:
                if not same_length_branch or key in missing_keys:
                    relative = key.removeprefix("forecast_adapter.")
                    state[key] = torch.zeros(expected[relative].shape, dtype=state["embedding.weight"].dtype, device="cpu")
    model.load_state_dict(state, strict=True, assign=True)
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


def drop_modalities(batch, probability, *, prediction_steps=1):
    """Apprendre aussi avec des modalités absentes, sans supprimer toutes les entrées."""
    names = [name for name in MODALITIES if name in batch]
    kept = [name for name in names if random.random() >= probability]
    if not kept:
        kept = [random.choice(names)]
    result = keep_modalities(batch, kept)
    # Les champs présents peuvent ne contenir que des cibles ignorées.
    if not has_observed_targets(result, prediction_steps=prediction_steps):
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


class DistributedTrainingModel(torch.nn.Module):
    """Ne présenter à DDP que la perte participant au backward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *, local_counts=None, total_counts=None, accumulation=1, world=1, **batch):
        output = self.model(**batch)
        loss = (output.loss if total_counts is None else
                observed_objective(output, local_counts, total_counts, self.model.config,
                                   world=world, accumulation=accumulation))
        return (loss, {name: value.detach() for name, value in output.losses.items()},
                output.expert_usage.detach(), output.routed_tokens.detach())


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
        overrides = json.loads(Path(args.model_overrides).read_text(encoding="utf-8")) if args.model_overrides else None
        model, source = load_checkpoint(args.resume or args.init_from, device, overrides=overrides)
        config = model.config
        if args.resume:
            if "optimizer" not in source:
                raise ValueError("Une reprise nécessite le checkpoint complet latest.pt avec son optimiseur")
            payload, previous, start = source, source.get("trainer", {}), source["step"]
        else:
            initialization = dict(checkpoint=str(Path(args.init_from).resolve()), step=source["step"],
                                  seed=source.get("trainer", {}).get("seed"), overrides=overrides or {})
            payload, previous, start = {}, {}, 0
        del source
    else:
        model, payload, previous, start = Chimera(config).to(device), {}, {}, 0
    previous_policy = previous.get("modality_policy", "dropout")
    args.modality_policy = args.modality_policy or previous_policy
    if args.resume and args.modality_policy != previous_policy:
        raise ValueError(f"Reprise : modality_policy diffère du checkpoint ({previous_policy!r})")
    previous_monitor = previous.get("monitor", "loss")
    args.monitor = getattr(args, "monitor", None) or previous_monitor
    if args.resume and args.monitor != previous_monitor:
        raise ValueError(f"Reprise : monitor diffère du checkpoint ({previous_monitor!r})")
    maximize = args.monitor == "accuracy"
    previous_normalization = previous.get("loss_normalization", "batch")
    args.loss_normalization = getattr(args, "loss_normalization", None) or previous_normalization
    if args.loss_normalization not in ("batch", "observed"):
        raise ValueError("loss_normalization doit être batch ou observed")
    if args.resume and args.loss_normalization != previous_normalization:
        raise ValueError(f"Reprise : loss_normalization diffère du checkpoint ({previous_normalization!r})")
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
    affine_options = {}
    previous_affine = previous.get("series_affine_augmentation", {})
    for name, default in SeriesAffineAugmentation().to_dict().items():
        saved = previous_affine.get(name, default)
        option = f"series_affine_{name}"
        value = getattr(args, option, None)
        value = saved if value is None else value
        if args.resume and value != saved:
            raise ValueError(f"Reprise : {option} diffère du checkpoint ({saved!r})")
        affine_options[name] = value
    augment_affine = SeriesAffineAugmentation(**affine_options)
    args.shuffle = (previous.get("shuffle", False) if args.resume else True) if args.shuffle is None else args.shuffle
    keys = ("steps", "batch_size", "accumulation", "seed", "lr", "warmup", "weight_decay",
            "amp", "data", "modality_dropout", "modality_policy", "shuffle", "val_data", "eval_every", "patience", "min_delta", "monitor", "loss_normalization")
    trainer = {key: getattr(args, key) for key in keys}
    trainer["initialization"] = previous.get("initialization", initialization)
    trainer["series_augmentation"] = augment_series.to_dict()
    trainer["series_affine_augmentation"] = augment_affine.to_dict()
    trainer["world_size"] = world
    dataset = JointDataset(args.data, config) if args.data else None
    validation = JointDataset(args.val_data, config) if args.val_data else None
    trainer["data_fingerprint"] = dataset.fingerprint if dataset is not None else None
    trainer["validation_fingerprint"] = validation.fingerprint if validation is not None else None
    trainer["data_images_fingerprint"] = dataset.image_fingerprint if dataset is not None else None
    trainer["validation_images_fingerprint"] = validation.image_fingerprint if validation is not None else None
    if args.resume:
        for name, current in trainer.items():
            if name in previous and previous[name] != current:
                raise ValueError(f"Reprise : {name} diffère du checkpoint ({previous[name]!r})")
            if (name in ("data_images_fingerprint", "validation_images_fingerprint") and name not in previous
                    and current is not None and rank == 0):
                warnings.warn(f"Checkpoint ancien : {name} absent, le contenu initial des images est invérifiable.",
                              RuntimeWarning)
    initial_best = float("-inf") if maximize else float("inf")
    state = dict(previous.get("validation_state", {"best_loss": initial_best, "bad_evaluations": 0, "best_step": None}))
    trainer["validation_state"] = state
    end_step = min(args.steps, args.stop_after) if args.stop_after is not None else args.steps
    if end_step < start or end_step < 1:
        raise ValueError("stop-after doit être >= à l'étape reprise et > 0")
    temporal_parameters = list(model.temporal.forecast_skip.parameters())
    if model.temporal.statistics_proj is not None:
        temporal_parameters.extend(model.temporal.statistics_proj.parameters())
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
    if args.resume:
        if world == 1:
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
            if device.type == "cuda" and payload["cuda_rng"]:
                torch.cuda.set_rng_state_all(payload["cuda_rng"])
        elif "rank_rng" in payload:
            states = payload["rank_rng"]
            if not isinstance(states, list) or len(states) != world:
                raise ValueError("États aléatoires distribués incompatibles avec le nombre de processus")
            state_for_rank = states[rank]
            torch.set_rng_state(state_for_rank["torch"])
            random.setstate(state_for_rank["python"])
            if device.type == "cuda":
                if state_for_rank["cuda"] is None:
                    if rank == 0:
                        warnings.warn("Checkpoint sans état aléatoire CUDA : la reprise ne sera pas exacte.", RuntimeWarning)
                else:
                    torch.cuda.set_rng_state(state_for_rank["cuda"], device)
            elif state_for_rank["cuda"] is not None and rank == 0:
                warnings.warn("Checkpoint CUDA repris sur CPU : la reprise ne sera pas exacte.", RuntimeWarning)
        elif rank == 0:
            warnings.warn("Checkpoint distribué ancien : les états aléatoires par rang ne sont pas disponibles ; "
                          "la reprise ne sera pas exacte.", RuntimeWarning)
    del payload
    wrapped = (DistributedDataParallel(DistributedTrainingModel(model), device_ids=[local_rank] if device.type == "cuda" else None,
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
        metrics, modality_batches, series_statistics = {}, {}, {}
        expert_totals = torch.zeros(config.n_layers, config.n_experts, device=device, dtype=torch.float64)
        expert_tokens = torch.zeros((), device=device, dtype=torch.long)
        def prepare_microbatch(micro):
            nonlocal epoch, iterator
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
            supervised = has_observed_targets(batch, prediction_steps=config.prediction_steps)
            if not supervised and args.loss_normalization == "batch":
                raise ValueError("Le batch ne contient aucune cible supervisée observée")
            if supervised and args.modality_policy == "balanced":
                batch = balanced_modalities(batch, index, args.seed, prediction_steps=config.prediction_steps)
            elif supervised and args.modality_dropout:
                batch = drop_modalities(batch, args.modality_dropout, prediction_steps=config.prediction_steps)
            # Le futur ne doit changer d'unités que si l'historique permettant
            # de connaître ces unités est encore présent dans les entrées.
            original_series = batch.get("series")
            batch = augment_affine(batch, index=index, seed=args.seed + 2000000)
            signature = "+".join(name for name in MODALITIES if name in batch)
            modality_batches[signature] = modality_batches.get(signature, 0) + 1
            if rank == 0 and "series" in batch:
                series = batch["series"]
                valid = batch.get("series_mask", torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)).bool()
                valid_steps = valid.sum().item()
                statistics = dict(examples=series.size(0), valid_steps=valid_steps,
                                  possible_values=valid_steps * series.size(-1),
                                  observed_values=(torch.isfinite(series) & valid.unsqueeze(-1)).sum().item(),
                                  affine_changed_examples=(torch.isfinite(original_series) & valid.unsqueeze(-1)
                                                           & (series != original_series)).any(dim=(1, 2)).sum().item())
                for name, value in statistics.items():
                    series_statistics[name] = series_statistics.get(name, 0) + value
            return batch

        batches = (prepare_microbatch(micro) for micro in range(args.accumulation))
        total_counts = None
        if args.loss_normalization == "observed":
            batches = list(batches)
            local_counts = [target_counts(batch, config.prediction_steps) for batch in batches]
            total_counts = torch.stack(local_counts).sum(0)
            if world > 1:
                dist.all_reduce(total_counts, op=dist.ReduceOp.SUM)
            if not total_counts.any().item():
                raise ValueError("L'étape ne contient aucune cible supervisée observée")
        else:
            local_counts = [None] * args.accumulation
        for micro, (batch, counts) in enumerate(zip(batches, local_counts)):
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
            with sync, autocast():
                if world > 1 and total_counts is not None:
                    result = wrapped(local_counts=counts, total_counts=total_counts,
                                     accumulation=args.accumulation, world=world, **batch)
                else:
                    result = wrapped(**batch)
                if world > 1:
                    batch_loss, batch_losses, batch_usage, batch_routed = result
                else:
                    batch_loss, batch_losses, batch_usage, batch_routed = (
                        result.loss, result.losses, result.expert_usage, result.routed_tokens)
                if batch_loss is None and total_counts is None:
                    raise ValueError("Aucune cible d'entraînement dans le batch")
                if total_counts is None:
                    loss = batch_loss / args.accumulation
                elif world > 1:
                    loss = batch_loss
                else:
                    loss = observed_objective(result, counts, total_counts, config,
                                              world=world, accumulation=args.accumulation)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Perte non finie à l'étape {step + 1}")
            scaler.scale(loss).backward()
            logged_loss = batch_loss.item() / args.accumulation if total_counts is None else loss.item()
            metrics["loss"] = metrics.get("loss", 0) + logged_loss
            for key, value in batch_losses.items():
                metrics[key] = metrics.get(key, 0) + value.detach().item() / args.accumulation
            expert_totals += batch_usage.detach().double() * batch_routed
            expert_tokens += batch_routed
        norm, skipped = step_optimizer(model, optimizer, scaler, args.grad_clip)
        last_step = step + 1
        routing = torch.cat((expert_totals.flatten(), expert_tokens.reshape(1).double()))
        if world > 1:
            dist.reduce(routing, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            routing_summary = summarize_expert_usage(routing[:-1].view(config.n_layers, config.n_experts),
                                                      int(routing[-1].item()))
            row = dict(step=last_step, **metrics, grad_norm=norm, optimizer_step_skipped=skipped,
                       lr=optimizer.param_groups[0]["lr"], elapsed_seconds=round(time.monotonic() - started, 3),
                       **routing_summary, modality_batches=modality_batches,
                       series_statistics=series_statistics)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == start or last_step % args.log_every == 0 or last_step == end_step:
                print(json.dumps({key: value for key, value in row.items() if key != "expert_usage"}), flush=True)
        # Ne pas ajouter une validation de contrôle à un simple --stop-after :
        # sa reprise doit conserver le même calendrier d'arrêt anticipé.
        validation_event = last_step % args.eval_every == 0 or last_step == args.steps
        if validation_event:
            improved = False
            if rank == 0:
                report = evaluate(model, val, args.amp)
                observed = sum(report.get(name, 0) for name in
                               ("text_tokens", "multi_token_targets", "classified_examples", "forecast_observations"))
                if not observed:
                    raise ValueError("Le jeu de validation ne contient aucune cible supervisée observée")
                score = report.get(args.monitor)
                if score is None or not math.isfinite(score):
                    raise ValueError(f"La mesure de validation {args.monitor} est absente ou non finie")
                last_evaluated = last_step
                reference = state.get("reference_loss", state["best_loss"])
                improved = score > state["best_loss"] if maximize else score < state["best_loss"]
                significant = score > reference + args.min_delta if maximize else score < reference - args.min_delta
                if improved:
                    state.update(best_loss=score, best_step=last_step)
                if significant:
                    state.update(reference_loss=score, bad_evaluations=0)
                else:
                    state["bad_evaluations"] += 1
                stopped_early = bool(args.patience and state["bad_evaluations"] >= args.patience)
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(step=last_step, improved=improved, monitor=args.monitor,
                                                 monitor_value=score, **report), allow_nan=False) + "\n")
                print(json.dumps({"validation_step": last_step, "loss": report.get("loss"),
                                  "monitor": args.monitor, "monitor_value": score,
                                  "best_step": state["best_step"]}), flush=True)
            if world > 1:
                decision = torch.tensor([int(stopped_early), int(improved)], device=device)
                dist.broadcast(decision, src=0)
                stopped_early, improved = map(bool, decision.tolist())
            if improved:
                rank_rng = collect_rng_states(device, world, rank)
                if rank == 0:
                    save_checkpoint(output_dir / "best.pt", model, optimizer=optimizer, scaler=scaler,
                                    step=last_step, trainer=trainer, rank_rng=rank_rng)
        if last_step % args.save_every == 0 or last_step == end_step or stopped_early:
            rank_rng = collect_rng_states(device, world, rank)
            if rank == 0:
                save_checkpoint(output_dir / "latest.pt", model, optimizer=optimizer, scaler=scaler,
                                step=last_step, trainer=trainer, rank_rng=rank_rng)
        if stopped_early:
            break
    if rank == 0:
        if last_evaluated != last_step:
            report = evaluate(model, val, args.amp)
        report = dict(report, step=last_step, best_step=state["best_step"], stopped_early=stopped_early,
                      monitor=args.monitor, best_monitor_value=state["best_loss"] if state["best_step"] is not None else None)
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"evaluation": report}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
