"""Alignement contrastif entre les paires présentes sur plusieurs rangs."""

import math
import struct

import torch
from torch import distributed as dist
from torch.nn import functional as F

from .losses import symmetric_contrastive_loss


class _GatherWithGrad(torch.autograd.Function):
    """All-gather différentiable ; le backward somme les contributions des rangs."""

    @staticmethod
    def forward(ctx, tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(output, tensor.contiguous(), group=group)
        return torch.stack(output, dim=0)

    @staticmethod
    def backward(ctx, gradient):
        # Un all-reduce fonctionne également sur Gloo, qui n'offre pas toujours
        # reduce_scatter. Le groupe est séparé de celui du réducteur DDP.
        gradient = gradient.contiguous()
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=ctx.group)
        return gradient[ctx.rank], None


def _rectangular_alignment(gathered, metadata, first, second, temperature, mode):
    """Aligner les vues qui ont une paire de ligne ou une identité commune."""
    left_rows = (metadata[:, first] != 0).nonzero(as_tuple=True)[0]
    right_rows = (metadata[:, second] != 0).nonzero(as_tuple=True)[0]
    if left_rows.numel() == 0 or right_rows.numel() == 0:
        return None
    left_ids, right_ids = metadata[left_rows, 4], metadata[right_rows, 4]
    positive = (left_rows[:, None] == right_rows[None, :])
    positive |= ((left_ids[:, None] == right_ids[None, :])
                 & (left_ids[:, None] >= 0) & (right_ids[None, :] >= 0))
    left_valid, right_valid = positive.any(-1), positive.any(0)
    left_rows, right_rows = left_rows[left_valid], right_rows[right_valid]
    if max(left_rows.numel(), right_rows.numel()) < 2 or not left_rows.numel() or not right_rows.numel():
        return None
    positive = positive[left_valid][:, right_valid]
    if mode == "class":
        left_labels, right_labels = metadata[left_rows, 3], metadata[right_rows, 3]
        positive |= ((left_labels[:, None] == right_labels[None, :])
                     & (left_labels[:, None] != -100) & (right_labels[None, :] != -100))
    left, right = gathered[left_rows, first], gathered[right_rows, second]
    with torch.autocast(device_type=left.device.type, enabled=False):
        scores = (left @ right.T / temperature).float()
        forward_weights = positive.float() / positive.sum(-1, keepdim=True)
        backward_weights = positive.T.float() / positive.T.sum(-1, keepdim=True)
        forward = -(forward_weights * F.log_softmax(scores, dim=-1)).sum(-1).mean()
        backward = -(backward_weights * F.log_softmax(scores.T, dim=-1)).sum(-1).mean()
    return (forward + backward) / 2


def distributed_alignment_loss(vectors, class_labels, temperature, mode, *, group, text_view="joint", sample_ids=None,
                               cross_rank_views=False):
    """Calculer les paires globales tout en conservant les gradients locaux.

    Les vecteurs, déjà normalisés, sont indexés par modalité (0, 1, 2).
    Tous les rangs du groupe doivent appeler cette fonction dans le même ordre.
    """
    if not vectors or any(key not in (0, 1, 2) for key in vectors):
        raise ValueError("Au moins une modalité connue est requise")
    sample = next(iter(vectors.values()))
    if sample.ndim != 2 or not sample.is_floating_point() or sample.size(0) < 1:
        raise ValueError("Les représentations doivent avoir la forme [B, D]")
    batch, dim = sample.shape
    if any(value.shape != sample.shape or value.device != sample.device or value.dtype != sample.dtype
           for value in vectors.values()):
        raise ValueError("Les représentations doivent partager forme, dtype et device")
    if class_labels is not None and (class_labels.shape != (batch,) or class_labels.dtype != torch.long
                                     or class_labels.device != sample.device):
        raise ValueError("class_labels doit avoir la forme [B] et le même device")
    if sample_ids is not None and (sample_ids.shape != (batch,) or sample_ids.dtype != torch.long
                                   or sample_ids.device != sample.device or (sample_ids < -1).any()):
        raise ValueError("sample_ids doit être long [B], avec -1 pour les identités inconnues")
    if type(cross_rank_views) is not bool:
        raise ValueError("cross_rank_views doit être booléen")
    if (mode not in ("paired", "class") or text_view not in ("joint", "isolated")
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Mode, vue textuelle ou température d'alignement invalide")
    dtype_codes = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3, torch.float64: 4}
    if sample.dtype not in dtype_codes:
        raise ValueError("Dtype de représentation non pris en charge")
    world = dist.get_world_size(group)
    temperature_bits = struct.unpack("!q", struct.pack("!d", float(temperature)))[0]
    specification = torch.tensor([batch, dim, dtype_codes[sample.dtype], int(mode == "class"),
                                  int(text_view == "isolated"), temperature_bits, int(cross_rank_views)],
                                 dtype=torch.long, device=sample.device)
    specifications = [torch.empty_like(specification) for _ in range(world)]
    dist.all_gather(specifications, specification, group=group)
    # Détecter les désaccords avant les collectives dont les tailles doivent coïncider.
    rows = [value.tolist() for value in specifications]
    if any(row[1:] != rows[0][1:] for row in rows[1:]):
        raise ValueError("Dimensions, dtype ou paramètres d'alignement différents selon les rangs")
    max_batch = max(row[0] for row in rows)
    zeros = torch.zeros_like(sample)
    packed = torch.stack([vectors.get(index, zeros) for index in range(3)], dim=1)
    if max_batch > batch:
        packed = F.pad(packed, (0, 0, 0, 0, 0, max_batch - batch))
    # Même un rang dont les paramètres sont gelés doit participer au backward.
    if torch.is_grad_enabled() and not packed.requires_grad:
        packed = packed.detach().requires_grad_(True)
    gathered = _GatherWithGrad.apply(packed, group).reshape(world * max_batch, 3, dim)
    metadata = torch.zeros(batch, 5, dtype=torch.long, device=sample.device)
    for index in vectors:
        metadata[:, index] = 1
    metadata[:, 3] = class_labels if class_labels is not None else -100
    metadata[:, 4] = sample_ids if sample_ids is not None else -1
    if max_batch > batch:
        metadata = F.pad(metadata, (0, 0, 0, max_batch - batch))
    all_metadata = [torch.empty_like(metadata) for _ in range(world)]
    dist.all_gather(all_metadata, metadata, group=group)
    all_metadata = torch.cat(all_metadata, dim=0)
    has_ids = ((all_metadata[:, 4] >= 0) & (all_metadata[:, :3] != 0).any(-1)).any().item()
    losses = []
    for first in range(3):
        for second in range(first + 1, 3):
            if has_ids and cross_rank_views:
                loss = _rectangular_alignment(gathered, all_metadata, first, second, temperature, mode)
                if loss is not None:
                    losses.append(loss)
                continue
            valid = (all_metadata[:, first] != 0) & (all_metadata[:, second] != 0)
            if valid.sum().item() < 2:
                continue
            left, right = gathered[valid, first], gathered[valid, second]
            with torch.autocast(device_type=sample.device.type, enabled=False):
                similarity = left @ right.T / temperature
            labels = all_metadata[valid, 3] if mode == "class" else None
            identities = all_metadata[valid, 4] if has_ids else None
            losses.append(symmetric_contrastive_loss(similarity, labels, sample_ids=identities))
    # Lien explicite au gather même sans paire : tous les rangs participent
    # alors au backward de la collective avec un gradient nul.
    return torch.stack(losses).mean() + gathered.sum() * 0 if losses else gathered.sum() * 0
