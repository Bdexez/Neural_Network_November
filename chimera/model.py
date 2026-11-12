from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ChimeraConfig
from .losses import symmetric_contrastive_loss
from .modules import (
    DecoderBlock, LayerCache, LatentResampler, QuantileForecastHead,
    RMSNorm, TemporalEncoder, VisionEncoder,
)


@dataclass
class ModelCache:
    layers: list[LayerCache]
    latents: torch.Tensor
    offset: int
    series_mean: torch.Tensor | None = None
    series_scale: torch.Tensor | None = None
    forecast_baseline: torch.Tensor | None = None


@dataclass
class ChimeraOutput:
    logits: torch.Tensor | None
    classification_logits: torch.Tensor
    forecast: torch.Tensor
    loss: torch.Tensor | None
    losses: dict[str, torch.Tensor]
    expert_usage: torch.Tensor
    cache: ModelCache | None = None


def masked_cross_entropy(logits, labels):
    # Un lot sans cible valide ne produit ni NaN ni apprentissage artificiel.
    values = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             labels.reshape(-1), ignore_index=-100, reduction="none")
    return values.sum() / (labels != -100).sum().clamp_min(1)


class Chimera(nn.Module):
    """Réseau conjoint : génération causale, classification, prévision probabiliste."""

    def __init__(self, config: ChimeraConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d)
        self.vision = VisionEncoder(config)
        self.temporal = TemporalEncoder(config)
        self.modality = nn.Parameter(torch.empty(2, d))
        self.resampler = LatentResampler(config)
        self.task_queries = nn.Parameter(torch.empty(1, 2, d))
        self.blocks = nn.ModuleList([DecoderBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(d, config.norm_eps)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.future_heads = nn.ModuleList([
            nn.Sequential(RMSNorm(d), nn.Linear(d, d, bias=False), nn.SiLU())
            for _ in range(config.prediction_steps - 1)
        ])
        self.fusion = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), RMSNorm(d))
        self.classifier = nn.Linear(d, config.num_classes)
        self.forecaster = QuantileForecastHead(config)
        self.alignment = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(3)])
        self.apply(self._initialize)
        nn.init.zeros_(self.temporal.forecast_skip.weight)
        for parameter in (self.modality, self.task_queries, self.resampler.latents,
                          self.resampler.null_context, self.vision.position):
            nn.init.normal_(parameter, std=0.02)
        for block in self.blocks:
            for expert in [*block.moe.experts, block.moe.shared]:
                nn.init.normal_(expert.down.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _validate(self, input_ids, images, series, attention_mask, series_mask, cache):
        c = self.config
        tensors = [x for x in (input_ids, images, series) if x is not None]
        if not tensors:
            raise ValueError("Au moins une modalité est requise : texte, images ou séries")
        batch = tensors[0].size(0)
        if batch < 1 or any(x.size(0) != batch for x in tensors):
            raise ValueError("Les modalités doivent partager un batch non vide")
        device = self.embedding.weight.device
        if any(x.device != device for x in tensors):
            raise ValueError("Les entrées et le modèle doivent être sur le même device")
        if input_ids is not None:
            if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.size(1) < 1:
                raise ValueError("input_ids doit être un tenseur long non vide [B, T]")
            if (input_ids < 0).any() or (input_ids >= c.vocab_size).any():
                raise ValueError("Identifiant de token hors vocabulaire")
            offset = 0 if cache is None else cache.offset
            if input_ids.size(1) + offset > c.max_seq_len:
                raise ValueError("La longueur texte dépasse max_seq_len")
        elif attention_mask is not None or cache is not None:
            raise ValueError("attention_mask et cache nécessitent input_ids")
        if images is not None:
            if images.shape[1:] != (c.image_channels, c.image_size, c.image_size):
                raise ValueError("images doit avoir la forme [B, canaux, image_size, image_size]")
            if not images.is_floating_point() or not torch.isfinite(images).all():
                raise ValueError("images doit contenir des nombres flottants finis")
        if series is not None:
            if series.ndim != 3 or series.size(-1) != c.series_features:
                raise ValueError("series doit avoir la forme [B, T, series_features]")
            if not series.is_floating_point() or not 1 <= series.size(1) <= c.max_series_len:
                raise ValueError("series doit être flottant et respecter max_series_len")
        elif series_mask is not None:
            raise ValueError("series_mask nécessite series")
        for mask, data, name in ((attention_mask, input_ids, "attention_mask"),
                                 (series_mask, series, "series_mask")):
            if mask is not None:
                if mask.shape != data.shape[:2] or mask.device != device:
                    raise ValueError(f"{name} doit partager la forme [B, T] et le device des données")
                if not ((mask == 0) | (mask == 1)).all():
                    raise ValueError(f"{name} doit être binaire")
                if not mask.bool().any(-1).all():
                    raise ValueError(f"{name} doit garder au moins une position par exemple")
        if cache is not None:
            if self.training:
                raise ValueError("Le cache est réservé à l'inférence en mode eval()")
            if images is not None or series is not None:
                raise ValueError("Le contexte multimodal est déjà dans le cache ; ne pas le réinjecter")
            if len(cache.layers) != c.n_layers or cache.latents.size(0) != batch:
                raise ValueError("Cache incompatible avec ce modèle ou ce batch")
            if cache.offset < 1 or any(
                layer.start_pos < 0 or layer.start_pos + layer.key.size(-2) != cache.offset
                or layer.key.shape != layer.value.shape
                or layer.key.size(0) != batch
                or layer.valid.shape != (batch, layer.key.size(-2))
                for layer in cache.layers
            ):
                raise ValueError("Positions incohérentes dans le cache")
        return batch

    def forward(
        self, input_ids=None, *, images=None, series=None, attention_mask=None,
        series_mask=None, labels=None, class_labels=None, forecast_targets=None,
        cache: ModelCache | None = None, use_cache=False,
    ) -> ChimeraOutput:
        batch = self._validate(input_ids, images, series, attention_mask, series_mask, cache)
        if use_cache and (self.training or input_ids is None):
            raise ValueError("use_cache nécessite eval() et des tokens texte")
        if cache is not None and not use_cache:
            raise ValueError("Passer use_cache=True pour continuer un cache")
        supervised = any(t is not None for t in (labels, class_labels, forecast_targets))
        if use_cache and supervised:
            raise ValueError("Le cache ne doit pas être utilisé pendant le calcul des pertes")
        if labels is not None and (input_ids is None or labels.shape != input_ids.shape):
            raise ValueError("labels doit partager la forme [B, T] de input_ids")
        if class_labels is not None and (class_labels.shape != (batch,) or class_labels.dtype != torch.long):
            raise ValueError("class_labels doit être long [B]")
        if forecast_targets is not None and forecast_targets.shape != (
            batch, self.config.forecast_horizon, self.config.series_features
        ):
            raise ValueError("forecast_targets doit avoir la forme [B, horizon, variables]")
        for target in (labels, class_labels, forecast_targets):
            if target is not None and target.device != self.embedding.weight.device:
                raise ValueError("Les cibles doivent être sur le même device que le modèle")
        if labels is not None and labels.dtype != torch.long:
            raise ValueError("labels doit contenir des entiers long ou -100")

        contexts, masks, align_vectors = [], [], []
        mean = scale = baseline = None
        if cache is None:
            if images is not None:
                image_tokens, image_valid = self.vision(images)
                contexts.append(image_tokens + self.modality[0])
                masks.append(image_valid)
                align_vectors.append((0, image_tokens.mean(1)))
            if series is not None:
                series_mask = (torch.ones(series.shape[:2], device=series.device, dtype=torch.bool)
                               if series_mask is None else series_mask.bool())
                series_tokens, temporal_valid, mean, scale, baseline = self.temporal(series, series_mask)
                contexts.append(series_tokens + self.modality[1])
                masks.append(temporal_valid)
                pooled_series = (series_tokens * temporal_valid.unsqueeze(-1)).sum(1)
                pooled_series = pooled_series / temporal_valid.sum(1, keepdim=True).clamp_min(1)
                align_vectors.append((1, pooled_series))
            latents = self.resampler(contexts, masks, batch)
        else:
            latents, mean, scale = cache.latents, cache.series_mean, cache.series_scale
            baseline = cache.forecast_baseline

        if input_ids is None:
            x = self.task_queries.expand(batch, -1, -1)
            valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            x = self.embedding(input_ids)
            valid = (torch.ones_like(input_ids, dtype=torch.bool)
                     if attention_mask is None else attention_mask.bool())
        offset = 0 if cache is None else cache.offset
        balance_losses, z_losses, usage, next_layers = [], [], [], []
        for index, block in enumerate(self.blocks):
            if self.config.gradient_checkpointing and self.training:
                # Capturer le bloc courant : aucune fermeture sur une variable de boucle mutable.
                def run(hidden, memory, current=block):
                    result = current(hidden, valid, memory)
                    return result[:4]
                x, balance, z_loss, expert_usage = checkpoint(run, x, latents, use_reentrant=False)
            else:
                x, balance, z_loss, expert_usage, layer_cache = block(
                    x, valid, latents, offset=offset,
                    cache=None if cache is None else cache.layers[index], use_cache=use_cache,
                )
                if use_cache:
                    next_layers.append(layer_cache)
            balance_losses.append(balance)
            z_losses.append(z_loss)
            usage.append(expert_usage)
        hidden = self.norm(x)
        logits = self.lm_head(hidden) if input_ids is not None else None
        if input_ids is None:
            class_pooled, forecast_pooled = hidden.unbind(1)
        else:
            last = torch.arange(valid.size(1), device=x.device)[None, :].expand_as(valid)
            last = last.masked_fill(~valid, -1).max(-1).values
            pooled = hidden[torch.arange(batch, device=x.device), last]
            class_pooled = forecast_pooled = pooled
            align_vectors.append((2, pooled))
        memory_pooled = latents.mean(1)
        class_pooled = self.fusion(torch.cat((class_pooled, memory_pooled), -1))
        forecast_pooled = self.fusion(torch.cat((forecast_pooled, memory_pooled), -1))
        class_logits = self.classifier(class_pooled)
        forecast = self.forecaster(forecast_pooled, mean, scale, baseline)

        losses = {
            "balance": torch.stack(balance_losses).mean(),
            "router_z": torch.stack(z_losses).mean(),
        }
        c = self.config
        total = c.balance_weight * losses["balance"] + c.router_z_weight * losses["router_z"]
        if labels is not None:
            targets = labels.masked_fill(~valid, -100)
            shifted = targets[:, 1:].masked_fill(~valid[:, :-1], -100)
            losses["text"] = masked_cross_entropy(logits[:, :-1], shifted)
            total = total + c.text_weight * losses["text"]
            mtp_losses = []
            for distance, head in enumerate(self.future_heads, start=2):
                if hidden.size(1) > distance:
                    future_logits = self.lm_head(head(hidden[:, :-distance]))
                    future_targets = targets[:, distance:].masked_fill(~valid[:, :-distance], -100)
                    mtp_losses.append(masked_cross_entropy(future_logits, future_targets))
            if mtp_losses:
                losses["multi_token"] = torch.stack(mtp_losses).mean()
                total = total + c.mtp_weight * losses["multi_token"]
        if class_labels is not None:
            losses["classification"] = masked_cross_entropy(class_logits, class_labels)
            total = total + c.classification_weight * losses["classification"]
        if forecast_targets is not None:
            observed = torch.isfinite(forecast_targets)
            safe_target = torch.where(observed, forecast_targets, 0.0)
            residual = safe_target.unsqueeze(-1) - forecast
            q = forecast.new_tensor(c.quantiles)
            pinball = torch.maximum(q * residual, (q - 1) * residual)
            losses["forecast"] = (pinball * observed.unsqueeze(-1)).sum() / (
                observed.sum().clamp_min(1) * len(c.quantiles)
            )
            total = total + c.forecast_weight * losses["forecast"]
        # Alignement contrastif de toutes les paires disponibles, jamais fictif avec B=1.
        if len(align_vectors) > 1 and batch > 1:
            vectors = [F.normalize(self.alignment[i](v).float(), dim=-1) for i, v in align_vectors]
            contrastive = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    with torch.autocast(device_type=x.device.type, enabled=False):
                        similarity = vectors[i] @ vectors[j].T / c.alignment_temperature
                    contrastive.append(symmetric_contrastive_loss(
                        similarity, class_labels if c.alignment_mode == "class" else None,
                    ))
            losses["alignment"] = torch.stack(contrastive).mean()
            total = total + c.alignment_weight * losses["alignment"]
        next_cache = (ModelCache(next_layers, latents, offset + hidden.size(1), mean, scale, baseline)
                      if use_cache else None)
        return ChimeraOutput(logits, class_logits, forecast, total if supervised else None,
                             losses, torch.stack(usage), next_cache)

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens=64, temperature=0.8, top_k=40,
                 top_p=0.95, eos_id=2, **context):
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("Le prompt doit avoir la forme [B, T] avec T > 0")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens doit être un entier positif ou nul")
        if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("temperature doit être >= 0 et top_p dans ]0, 1]")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k doit être un entier positif ou nul")
        if input_ids.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError("Prompt + génération dépassent max_seq_len")
        if eos_id is not None and not 0 <= eos_id < self.config.vocab_size:
            raise ValueError("eos_id hors vocabulaire")
        if "attention_mask" in context:
            mask = context["attention_mask"]
            if not mask[:, -1].bool().all():
                raise ValueError("La génération en batch nécessite du padding à gauche")
        was_training = self.training
        self.eval()
        try:
            result = input_ids
            finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
            cache = None
            for step in range(max_new_tokens):
                output = self(input_ids if step == 0 else result[:, -1:],
                              cache=cache, use_cache=True, **(context if step == 0 else {}))
                cache = output.cache
                scores = output.logits[:, -1].float()
                # Les tokens de contrôle réservés ne sont pas des caractères générables.
                scores[:, [0, 1, 3]] = -torch.inf
                if temperature == 0:
                    next_token = scores.argmax(-1)
                else:
                    scores = scores / temperature
                    if top_k:
                        threshold = scores.topk(min(top_k, scores.size(-1)), dim=-1).values[:, -1:]
                        scores = scores.masked_fill(scores < threshold, -torch.inf)
                    ordered, order = scores.sort(descending=True)
                    cumulative = ordered.softmax(-1).cumsum(-1)
                    remove = cumulative - ordered.softmax(-1) >= top_p
                    ordered = ordered.masked_fill(remove, -torch.inf)
                    scores = torch.full_like(scores, -torch.inf).scatter(-1, order, ordered)
                    next_token = torch.multinomial(scores.softmax(-1), 1).squeeze(-1)
                if eos_id is not None:
                    next_token = torch.where(finished, eos_id, next_token)
                    finished = finished | (next_token == eos_id)
                result = torch.cat((result, next_token[:, None]), -1)
                if eos_id is not None and finished.all():
                    break
            return result
        finally:
            self.train(was_training)
