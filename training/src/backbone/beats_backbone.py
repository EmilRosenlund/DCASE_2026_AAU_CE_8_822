"""BEATs backbone for full fine-tuning.

Self-contained — does not import from pipeline3.
Locates the BEATs source code via the repo layout:

    AAU_P8/
        beats/          ← BEATs source (BEATs.py, modules.py, …)
        training/       ← this file lives here
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Union

import torch
import torch.nn as nn

from .base import BaseBackbone
from src.utils import get_logger

logger = get_logger(__name__)

# BEATs source is two levels above this file: AAU_P8/beats/
_BEATS_DIR = Path(__file__).resolve().parents[3] / "beats"


def _load_beats(checkpoint: str):
    """Load a BEATs checkpoint and return the raw model on CPU."""
    beats_str = str(_BEATS_DIR)
    if beats_str not in sys.path:
        sys.path.insert(0, beats_str)

    from BEATs import BEATs, BEATsConfig  # type: ignore

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"BEATs checkpoint not found: {ckpt_path}\n"
            "Download BEATs_iter3_plus_AS2M.pt to the models/ directory."
        )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt["cfg"]
    cfg = BEATsConfig(cfg_dict)
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg_dict


class BEATsBackbone(BaseBackbone):
    """BEATs transformer backbone with configurable fine-tuning scope.

    Parameters
    ----------
    checkpoint:
        Absolute path to the BEATs *.pt checkpoint file.
    pool:
        Temporal pooling mode after the transformer:
        - ``"mean"``     → ``(B, 768)``
        - ``"mean+std"`` → ``(B, 1536)``
        - ``"first"``    → ``(B, 768)``  (CLS-like)
    fine_tune_layers:
        ``"all"`` — unfreeze every parameter (true full fine-tune).
        integer N — freeze all layers, then unfreeze the top N
                    transformer encoder blocks.
    """

    def __init__(
        self,
        checkpoint: str,
        pool: Literal["mean", "mean+std", "first"] = "mean",
        fine_tune_layers: Union[str, int] = "all",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.pool_mode = pool
        self._beats, self._beats_cfg_dict = _load_beats(checkpoint)
        self._embedding_dim = 768  # BEATs iter3+AS2M hidden size

        # Apply freeze/unfreeze policy
        if fine_tune_layers == "all":
            self.unfreeze_all()
            logger.info("BEATsBackbone — unfreezing ALL layers (full fine-tune)")
        elif isinstance(fine_tune_layers, int):
            self.unfreeze_top_n(fine_tune_layers)
            logger.info("BEATsBackbone — unfreezing top %d transformer blocks", fine_tune_layers)
        else:
            raise ValueError(f"fine_tune_layers must be 'all' or int, got {fine_tune_layers!r}")

        # Freeze the BEATs AudioSet predictor head — it is never called by
        # extract_features and would cause DDP to error on unreduced gradients.
        self._freeze_predictor()

        # Gradient checkpointing: recompute activations during backward to save VRAM
        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        logger.info(
            "BEATsBackbone | pool=%s | out_dim=%d | grad_ckpt=%s | trainable_params=%s",
            pool,
            self.output_dim,
            gradient_checkpointing,
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
        )

    # ──────────────────────────────────────────────────────────────────────
    # BaseBackbone interface
    # ──────────────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return self._embedding_dim * 2 if self.pool_mode == "mean+std" else self._embedding_dim

    def freeze_all(self) -> None:
        for p in self._beats.parameters():
            p.requires_grad_(False)
        self._beats.eval()

    def unfreeze_all(self) -> None:
        for p in self._beats.parameters():
            p.requires_grad_(True)
        self._beats.train()

    def unfreeze_top_n(self, n: int) -> None:
        """Freeze everything, then unfreeze the last *n* encoder blocks.

        BEATs encoder layers live at ``self._beats.encoder.layers`` (a
        ``nn.ModuleList``). If the model has a different attribute layout
        this method will fall back to unfreezing all layers with a warning.
        """
        self.freeze_all()

        encoder = getattr(self._beats, "encoder", None)
        layers = getattr(encoder, "layers", None) if encoder is not None else None

        if layers is None or not isinstance(layers, nn.ModuleList):
            logger.warning(
                "BEATsBackbone.unfreeze_top_n: could not locate encoder.layers — "
                "falling back to unfreezing ALL parameters."
            )
            self.unfreeze_all()
            return

        total = len(layers)
        n = min(n, total)
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        # Also unfreeze the final layer-norm / projection if present
        for attr in ("layer_norm", "post_extract_proj"):
            m = getattr(self._beats, attr, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def _freeze_predictor(self) -> None:
        """Freeze the BEATs AudioSet classification head.

        ``BEATs.extract_features`` never calls the predictor, so its
        parameters never receive gradients.  DDP would error waiting for
        their all-reduce.  Freezing them removes them from the DDP bucket.
        """
        frozen = 0
        for attr in ("predictor", "head", "fc"):
            m = getattr(self._beats, attr, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
                    frozen += 1
        if frozen:
            logger.info(
                "BEATsBackbone \u2014 froze %d predictor parameters (not used in extract_features)",
                frozen,
            )

    def _enable_gradient_checkpointing(self) -> None:
        """Wrap each transformer encoder block with gradient checkpointing.

        This recomputes activations during the backward pass instead of
        storing them, reducing peak VRAM by ~40–50% at the cost of ~30%
        extra compute.  Falls back gracefully if the encoder layout differs.
        """
        try:
            from torch.utils.checkpoint import checkpoint as ckpt_fn

            encoder = getattr(self._beats, "encoder", None)
            layers  = getattr(encoder, "layers", None) if encoder is not None else None
            if layers is None or not isinstance(layers, nn.ModuleList):
                logger.warning(
                    "BEATsBackbone: gradient checkpointing not applied — "
                    "could not locate encoder.layers."
                )
                return

            # Replace each layer's forward with a checkpointed version
            for i, layer in enumerate(layers):
                original_forward = layer.forward

                def make_ckpt_forward(fwd):
                    def checkpointed(*args, **kwargs):
                        # checkpoint requires at least one Tensor input
                        return ckpt_fn(fwd, *args, use_reentrant=False, **kwargs)
                    return checkpointed

                layer.forward = make_ckpt_forward(original_forward)

            logger.info(
                "BEATsBackbone — gradient checkpointing enabled on %d encoder layers",
                len(layers),
            )
        except Exception as e:
            logger.warning("BEATsBackbone: gradient checkpointing setup failed: %s", e)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pooled embedding.

        Parameters
        ----------
        waveform : ``(B, 1, T)`` or ``(B, T)`` at 16 kHz

        Returns
        -------
        embedding : ``(B, output_dim)``
        """
        # Normalise shape to (B, T)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = next(self._beats.parameters()).device
        waveform = waveform.to(device)
        padding_mask = torch.zeros(
            waveform.shape[0], waveform.shape[1],
            dtype=torch.bool, device=device,
        )
        frames, _ = self._beats.extract_features(waveform, padding_mask=padding_mask)
        # frames: (B, seq_len, 768)
        return self._pool(frames)

    def _pool(self, frames: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return frames.mean(dim=1)
        elif self.pool_mode == "mean+std":
            return torch.cat([frames.mean(dim=1), frames.std(dim=1)], dim=-1)
        elif self.pool_mode == "first":
            return frames[:, 0, :]
        raise ValueError(f"Unknown pool mode: {self.pool_mode!r}")
    
    def extract_frames(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return raw frame-level features (B, seq_len, 768) without pooling.

        Used by the trainer's _extract_embeddings so that the eval pooling
        strategy (Lp) can differ from the training pool mode.
        """
        """ logger.info("waveform dtype: %s, min: %.4f, max: %.4f, has_nan: %s, has_inf: %s",
            waveform.dtype,
            waveform.min().item(),
            waveform.max().item(),
            torch.isnan(waveform).any().item(),
            torch.isinf(waveform).any().item(),
        ) """
        
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        device       = next(self._beats.parameters()).device
        
        waveform = waveform.float().to(device)
        #waveform     = waveform.to(device)
        
        padding_mask = torch.zeros(
            waveform.shape[0], waveform.shape[1], dtype=torch.bool, device=device
        )
        
        frames, _ = self._beats.extract_features(waveform, padding_mask=padding_mask)
       
        return frames  # (B, T, 768)

    # ──────────────────────────────────────────────────────────────────────
    # Device handling — keep internal model in sync
    # ──────────────────────────────────────────────────────────────────────

    def to(self, *args, **kwargs):
        self._beats = self._beats.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def checkpoint_state(self) -> dict:
        """Checkpoint format expected by the BEATs inference loader."""
        model_state = {
            k[len("_beats."):]: v
            for k, v in self.state_dict().items()
            if k.startswith("_beats.")
        }
        return {
            "backbone_type": "beats",
            "cfg":           self._beats_cfg_dict,
            "model":         model_state,
        }
