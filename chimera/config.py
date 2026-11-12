from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass
class ChimeraConfig:
    vocab_size: int = 260
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    hidden_dim: int = 768
    n_experts: int = 8
    top_k: int = 2
    shared_experts: int = 1
    max_seq_len: int = 2048
    sliding_window: int = 256
    global_every: int = 4
    conv_kernel: int = 5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-6
    image_size: int = 64
    image_channels: int = 3
    patch_size: int = 8
    vision_layers: int = 2
    series_features: int = 3
    max_series_len: int = 256
    temporal_layers: int = 2
    temporal_statistics: bool = False
    n_latents: int = 16
    resampler_layers: int = 2
    num_classes: int = 6
    forecast_horizon: int = 12
    forecast_context: int = 32
    forecast_adapter_context: int = 0
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    prediction_steps: int = 2
    balance_weight: float = 0.01
    router_z_weight: float = 0.001
    mtp_weight: float = 0.1
    alignment_weight: float = 0.05
    alignment_mode: str = "paired"
    alignment_temperature: float = 0.1
    text_weight: float = 1.0
    classification_weight: float = 1.0
    forecast_weight: float = 1.0
    gradient_checkpointing: bool = False

    def __post_init__(self):
        integers = (
            "vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
            "hidden_dim", "n_experts", "top_k", "shared_experts", "max_seq_len",
            "sliding_window", "global_every", "conv_kernel", "image_size",
            "image_channels", "patch_size", "vision_layers", "series_features",
            "max_series_len", "temporal_layers", "n_latents", "resampler_layers",
            "num_classes", "forecast_horizon", "forecast_context", "prediction_steps",
        )
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} doit être un entier strictement positif")
        if self.vocab_size < 260:
            raise ValueError("vocab_size doit être >= 260 pour le tokenizer UTF-8")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads doit être un entier pair (RoPE)")
        if self.d_model % 4 or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model doit être divisible par 4 et n_heads par n_kv_heads")
        if self.top_k > self.n_experts:
            raise ValueError("top_k ne peut pas dépasser n_experts")
        if self.image_size % self.patch_size:
            raise ValueError("image_size doit être divisible par patch_size")
        if (type(self.forecast_adapter_context) is not int
                or not 0 <= self.forecast_adapter_context <= self.forecast_context):
            raise ValueError("forecast_adapter_context doit être un entier entre 0 et forecast_context")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout doit appartenir à [0, 1[")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
            raise ValueError("rope_theta doit être fini et > 1")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps doit être fini et > 0")
        if self.alignment_mode not in ("paired", "class"):
            raise ValueError("alignment_mode doit être paired ou class")
        if not math.isfinite(self.alignment_temperature) or self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature doit être finie et > 0")
        self.quantiles = tuple(self.quantiles)
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("Les quantiles doivent appartenir à ]0, 1[")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("Les quantiles doivent être strictement croissants")
        if 0.5 not in self.quantiles:
            raise ValueError("Le quantile médian 0.5 est requis")
        for name in ("balance_weight", "router_z_weight", "mtp_weight", "alignment_weight",
                     "text_weight", "classification_weight", "forecast_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} doit être fini et positif ou nul")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing doit être booléen")
        if type(self.temporal_statistics) is not bool:
            raise ValueError("temporal_statistics doit être booléen")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Clés de configuration inconnues : {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def preset(cls, name):
        presets = {
            "tiny": dict(d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                         hidden_dim=128, n_experts=4, max_seq_len=256,
                         sliding_window=32, global_every=2, image_size=32,
                         patch_size=4, vision_layers=1, temporal_layers=1,
                         n_latents=8, resampler_layers=1, max_series_len=128),
            "small": {},
            "workstation": dict(d_model=512, n_layers=12, n_heads=8, n_kv_heads=2,
                                hidden_dim=1536, n_experts=16, n_latents=32,
                                vision_layers=3, temporal_layers=3, resampler_layers=2,
                                gradient_checkpointing=True),
            "base": dict(d_model=768, n_layers=16, n_heads=12, n_kv_heads=4,
                         hidden_dim=2048, n_experts=16, image_size=224,
                         patch_size=16, vision_layers=4, temporal_layers=4,
                         n_latents=32, resampler_layers=3, max_seq_len=4096),
            "large": dict(d_model=2048, n_layers=32, n_heads=16, n_kv_heads=4,
                          hidden_dim=5632, n_experts=32, image_size=224,
                          patch_size=16, vision_layers=8, temporal_layers=6,
                          n_latents=64, resampler_layers=4, max_seq_len=8192,
                          gradient_checkpointing=True),
            "ultra": dict(d_model=4096, n_layers=64, n_heads=32, n_kv_heads=8,
                          hidden_dim=11008, n_experts=64, top_k=4,
                          shared_experts=2, image_size=448, patch_size=16,
                          vision_layers=12, temporal_layers=8, n_latents=128,
                          resampler_layers=6, max_seq_len=32768,
                          sliding_window=1024, gradient_checkpointing=True),
        }
        if name not in presets:
            raise ValueError(f"Preset inconnu : {name}. Choix : {', '.join(presets)}")
        return cls(**presets[name])
