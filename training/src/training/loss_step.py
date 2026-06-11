"""Pluggable loss-step interface.

A ``LossStep`` owns the loss head(s) and defines the full forward + loss
computation for one mini-batch.  The trainer loop is completely agnostic to
which objective is being optimised — adding a new one only requires:

1. A new ``LossStep`` subclass here.
2. Instantiating it and passing it to ``ArcFaceTrainer``.

Interface summary
-----------------
``__call__(backbone, batch, device)``
    Forward pass + loss.  Called inside the trainer's amp autocast context.
    Returns ``(loss_scalar, info_dict)`` where *info_dict* holds any float
    values you want logged to WandB / console.

``head_parameters() -> list``
    All ``nn.Parameter`` objects owned by this step's head(s).
    Used for the optimizer head-LR param group and gradient clipping.

``to_device(device)``
    Move all owned modules to *device*.

``wrap_ddp(rank, device_ids)``
    DDP-wrap owned modules.  Called by the trainer after ``to_device``.

``set_train() / set_eval()``
    Set owned modules to the appropriate PyTorch mode.

``eval_forward(backbone, batch, device) -> (loss, info_dict)``
    Called by ``_eval_epoch`` on the *validation* split.  Defaults to
    running the same pass as ``__call__`` but with a single-view batch.
    Override when the train and eval objectives differ (e.g. selflabel uses
    the supervised ArcFace branch for validation).

``eval_head  -> nn.Module | None``
    Convenience property: the supervised ArcFace / HierarchicalHead, or
    ``None`` if this step has no supervised signal (pure OT only).

``checkpoint_state() -> dict``
    Extra ``{key: state_dict}`` mappings to embed in the checkpoint file.
    The trainer always saves the backbone; the step saves its own heads.

``needs_two_views: bool``
    Tells the trainer to build a ``MultiViewEpochBuffer`` (two independently
    augmented views per sample) instead of a plain ``EpochBuffer``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LossStep(ABC):
    """Abstract pluggable loss step. See module docstring for the full contract."""

    needs_two_views: bool = False

    @abstractmethod
    def __call__(
        self,
        backbone: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ...

    @abstractmethod
    def head_parameters(self) -> List[nn.Parameter]:
        ...

    @abstractmethod
    def to_device(self, device: torch.device) -> None:
        ...

    def wrap_ddp(self, rank: int, device_ids: list) -> None:
        pass

    def set_train(self) -> None:
        pass

    def set_eval(self) -> None:
        pass

    def eval_forward(
        self,
        backbone: nn.Module,
        batch: tuple,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.__call__(backbone, batch, device)

    @property
    def eval_head(self) -> Optional[nn.Module]:
        return None

    def checkpoint_state(self) -> Dict[str, dict]:
        """Extra state dicts to embed in the checkpoint. Default: empty."""
        return {}

    def load_head_state(self, ckpt: Dict[str, dict]) -> None:
        """Inverse of checkpoint_state. Restores head weights from the checkpoint dict."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared single-head mixin
# ─────────────────────────────────────────────────────────────────────────────

class _SingleHeadLossStep(LossStep):
    """Mixin for loss steps that own exactly one ``nn.Module`` head."""

    def __init__(self, head: nn.Module, checkpoint_key: str = "head_state_dict") -> None:
        self._head           = head
        self._checkpoint_key = checkpoint_key

    def head_parameters(self):
        raw = self._head.module if isinstance(self._head, DDP) else self._head
        return list(raw.parameters())

    def to_device(self, device):
        self._head = self._head.to(device)

    def wrap_ddp(self, rank, device_ids):
        self._head = DDP(self._head, device_ids=device_ids, find_unused_parameters=True)

    def set_train(self):
        self._head.train()

    def set_eval(self):
        self._head.eval()

    @property
    def eval_head(self):
        return self._head

    def checkpoint_state(self):
        raw = self._head.module if isinstance(self._head, DDP) else self._head
        return {self._checkpoint_key: raw.state_dict()}

    def load_head_state(self, ckpt: Dict[str, dict]) -> None:
        if self._checkpoint_key in ckpt:
            logger.info(f"Restoring head state from checkpoint key: {self._checkpoint_key}")
            raw = self._head.module if isinstance(self._head, DDP) else self._head
            raw.load_state_dict(ckpt[self._checkpoint_key])


# ─────────────────────────────────────────────────────────────────────────────
# Standard ArcFace
# ─────────────────────────────────────────────────────────────────────────────

class StandardLossStep(_SingleHeadLossStep):
    def __init__(self, arcface_head: nn.Module) -> None:
        super().__init__(arcface_head)

    def __call__(self, backbone, batch, device):
        waveforms, labels = batch
        waveforms = waveforms.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)
        embeddings = backbone(waveforms)
        loss       = self._head(embeddings, labels)
        return loss, {}


class _NoEvalHeadError(Exception):
    pass
