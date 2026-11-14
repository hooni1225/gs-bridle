"""
Compatibility shim for video ViT.

Some code expects `model_video.vit` to provide:
  Block, PatchEmbed, get_sinusoid_encoding_table, _cfg

This repo provides most of them in `model_video.vvit`.
We re-export what exists and provide a minimal `_cfg` fallback.
"""

from .vvit import *  # noqa: F401,F403

def _cfg(url: str = "", **kwargs):
    """
    Minimal timm-like default config.
    Many projects use this to attach metadata; training usually doesn't depend on it.
    """
    cfg = dict(
        url=url,
        num_classes=kwargs.pop("num_classes", 1000),
        input_size=kwargs.pop("input_size", (3, 224, 224)),
        crop_pct=kwargs.pop("crop_pct", 0.9),
        interpolation=kwargs.pop("interpolation", "bicubic"),
        mean=kwargs.pop("mean", (0.485, 0.456, 0.406)),
        std=kwargs.pop("std", (0.229, 0.224, 0.225)),
    )
    cfg.update(kwargs)
    return cfg
