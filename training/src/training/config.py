"""Typed configuration dataclasses for the training pipeline.

Each class mirrors one YAML config block and provides:

- Type-checked fields with sensible defaults.
- A ``from_cfg`` classmethod that reads from an OmegaConf DictConfig or
  any object supporting ``getattr`` (the PyYAML SimpleNamespace fallback).
- A ``head_kwargs`` method that returns a plain ``dict`` ready to be
  unpacked into the corresponding constructor with ``**``.

Usage::

    arc_cfg = ArcFaceConfig.from_cfg(cfg)
    head = ArcFaceHead(in_dim=768, num_classes=37, **arc_cfg.head_kwargs())

    sl_cfg = SelflabelConfig.from_cfg(cfg)
    head = SelfLabelHead(embed_dim=768, **sl_cfg.head_kwargs())
"""

from __future__ import annotations

from dataclasses import dataclass


def _g(obj, key: str, default):
    """Attribute-or-key access that works for both dicts and namespace objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace head
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArcFaceConfig:
    """Config for :class:`~src.loss.arcface.ArcFaceHead`."""
    scale:       float = 30.0
    margin:      float = 0.5
    easy_margin: bool  = False

    @classmethod
    def from_cfg(cls, cfg) -> "ArcFaceConfig":
        acfg = getattr(cfg, "arcface", None)
        return cls(
            scale       = float(_g(acfg, "scale",       30.0)),
            margin      = float(_g(acfg, "margin",       0.5)),
            easy_margin = bool( _g(acfg, "easy_margin", False)),
        )

    def head_kwargs(self) -> dict:
        return {"scale": self.scale, "margin": self.margin, "easy_margin": self.easy_margin}


# ─────────────────────────────────────────────────────────────────────────────
# Self-labeling head (removed — only standard ArcFace supported)
# ─────────────────────────────────────────────────────────────────────────────
# Previously: SelflabelConfig, HierarchicalConfig

