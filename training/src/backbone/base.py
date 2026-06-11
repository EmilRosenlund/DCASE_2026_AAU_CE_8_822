"""Abstract base class for all audio backbone models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseBackbone(ABC, nn.Module):
    """Protocol every backbone must satisfy.

    Subclasses wrap a pretrained audio encoder and expose:
    - A standard ``forward(waveform) -> embedding`` interface.
    - Freeze / unfreeze helpers so the training loop is backbone-agnostic.
    - An ``output_dim`` property so downstream heads can size themselves.

    Waveform contract:
        Input:  ``(B, 1, T)`` or ``(B, T)`` — float32, 16 kHz
        Output: ``(B, output_dim)`` — float32 pooled embedding
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the pooled embedding vector."""
        ...

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding ``(B, output_dim)``."""
        ...

    @abstractmethod
    def freeze_all(self) -> None:
        """Freeze every parameter (no gradients)."""
        ...

    @abstractmethod
    def unfreeze_all(self) -> None:
        """Unfreeze every parameter (full fine-tune)."""
        ...

    @abstractmethod
    def unfreeze_top_n(self, n: int) -> None:
        """Freeze all layers, then unfreeze the top *n* transformer blocks."""
        ...

    def checkpoint_state(self) -> dict:
        """Return a dict to embed in the checkpoint file at save time.

        The default implementation saves the full ``state_dict`` under a
        generic key.  Backbone subclasses should override to produce the
        exact format their own loader expects at inference time (e.g.
        stripping internal attribute prefixes, embedding config dicts, etc.).
        """
        return {
            "backbone_type":       type(self).__name__,
            "backbone_state_dict": self.state_dict(),
        }
