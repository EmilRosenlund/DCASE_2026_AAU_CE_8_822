"""SSLAM backbone for fine-tuning.

Loads ``ta012/SSLAM_pretrain`` via HuggingFace Transformers and adapts it to
the ``BaseBackbone`` interface used by the training pipeline.

Key difference from ``BEATsBackbone``: SSLAM expects mel-filterbank input, so
waveforms are pre-processed before the network forward pass.
"""

from __future__ import annotations

from typing import Literal, Union

import torch
import torch.nn as nn
import torchaudio
import transformers

from .base import BaseBackbone
from src.utils import get_logger

logger = get_logger(__name__)

_MODEL_ID   = "ta012/SSLAM_pretrain"
_EMBED_DIM  = 768
# AudioSet normalisation constants (must match inference pipeline)
_FBANK_MEAN = -4.268
_FBANK_STD  =  4.569


def _patch_sslam_model(model):
    """No-op patch - SSLAM all-layer training requires disabling mixed precision and checkpointing."""
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Mel-filterbank helper (matches create_sslam_embeddings.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _waveform_to_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
) -> torch.Tensor:
    """Convert ``(B, T)`` waveforms to ``(B, 1, T', 128)`` fbank tensors.

    ``torchaudio.compliance.kaldi.fbank`` operates on one clip at a time, so
    we loop over the batch and pad to a common length.
    """
    cpu = waveform.cpu().float()  # kaldi ops run on CPU and require float32
    fbanks = []
    for wav in cpu:
        fb = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),          # (1, T)
            htk_compat=True,
            sample_frequency=sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )  # (T', 128)
        fbanks.append(fb)

    # Pad to the longest clip in the batch
    max_t = max(f.shape[0] for f in fbanks)
    padded = []
    for f in fbanks:
        if f.shape[0] < max_t:
            pad = f.new_zeros(max_t - f.shape[0], 128)
            f = torch.cat([f, pad], dim=0)
        padded.append(f)

    out = torch.stack(padded)          # (B, T', 128)
    out = (out - _FBANK_MEAN) / _FBANK_STD
    return out.unsqueeze(1)            # (B, 1, T', 128)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────────────────────────────────────

class SSLAMBackbone(BaseBackbone):
    """SSLAM transformer backbone with configurable fine-tuning scope.

    Parameters
    ----------
    pool:
        Temporal pooling over the ``(B, T', 768)`` frame features:
        - ``"mean"``     → ``(B, 768)``
        - ``"mean+std"`` → ``(B, 1536)``
        - ``"first"``    → ``(B, 768)``
    fine_tune_layers:
        ``"all"`` — unfreeze every parameter.
        integer N — freeze all, then unfreeze the top N transformer blocks.
    gradient_checkpointing:
        Wrap encoder blocks with gradient checkpointing to save VRAM.
    """

    def __init__(
        self,
        pool: Literal["mean", "mean+std", "first"] = "mean",
        fine_tune_layers: Union[str, int] = "all",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.pool_mode = pool

        logger.info("Loading SSLAM model from %s …", _MODEL_ID)
        self._sslam = transformers.AutoModel.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        ).eval()
        
        # Patch model for gradient compatibility when training all layers
        self._sslam = _patch_sslam_model(self._sslam)
        
        logger.info(
            "SSLAM loaded | transformers=%s", transformers.__version__
        )

        if fine_tune_layers == "all":
            self.unfreeze_all()
            logger.info("SSLAMBackbone — unfreezing ALL layers (full fine-tune)")
        elif isinstance(fine_tune_layers, int):
            self.unfreeze_top_n(fine_tune_layers)
            logger.info("SSLAMBackbone — unfreezing top %d transformer blocks", fine_tune_layers)
        else:
            raise ValueError(
                f"fine_tune_layers must be 'all' or int, got {fine_tune_layers!r}"
            )

        if gradient_checkpointing:
            self._enable_gradient_checkpointing()

        logger.info(
            "SSLAMBackbone | pool=%s | out_dim=%d | grad_ckpt=%s | trainable=%s",
            pool,
            self.output_dim,
            gradient_checkpointing,
            f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}",
        )

    # ── BaseBackbone interface ────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return _EMBED_DIM * 2 if self.pool_mode == "mean+std" else _EMBED_DIM

    def freeze_all(self) -> None:
        for p in self._sslam.parameters():
            p.requires_grad_(False)
        self._sslam.eval()

    def unfreeze_all(self) -> None:
        for p in self._sslam.parameters():
            p.requires_grad_(True)
        self._sslam.train()

    def unfreeze_top_n(self, n: int) -> None:
        """Freeze everything, then unfreeze the top *n* encoder blocks."""
        self.freeze_all()
        layers = self._find_encoder_layers()
        if layers is None:
            logger.warning(
                "SSLAMBackbone.unfreeze_top_n: could not locate encoder layers — "
                "falling back to unfreezing ALL."
            )
            self.unfreeze_all()
            return

        total = len(layers)
        for layer in layers[-min(n, total):]:
            for p in layer.parameters():
                p.requires_grad_(True)

        # Also unfreeze any final layer-norm / projection
        for attr in ("layer_norm", "norm", "post_extract_proj", "final_proj"):
            m = getattr(self._sslam, attr, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_encoder_layers(self):
        """Return the EAT transformer block list.

        EATModel (HuggingFace wrapper)
          └─ .model  (EAT)
               └─ .blocks  (nn.ModuleList of AltBlock)
        """
        eat = getattr(self._sslam, "model", None)
        blocks = getattr(eat, "blocks", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            return blocks
        return None

    def _enable_gradient_checkpointing(self) -> None:
        from torch.utils.checkpoint import checkpoint as ckpt_fn
        layers = self._find_encoder_layers()
        if layers is None:
            logger.warning(
                "SSLAMBackbone: gradient checkpointing not applied — "
                "could not locate encoder layers."
            )
            return
        for layer in layers:
            original_fwd = layer.forward

            def _wrap(fwd):
                def _ckpt(*args, **kwargs):
                    return ckpt_fn(fwd, *args, use_reentrant=False, **kwargs)
                return _ckpt

            layer.forward = _wrap(original_fwd)

        logger.info(
            "SSLAMBackbone — gradient checkpointing enabled on %d layers", len(layers)
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pooled embedding from raw waveform.

        Parameters
        ----------
        waveform : ``(B, 1, T)`` or ``(B, T)`` at 16 kHz

        Returns
        -------
        embedding : ``(B, output_dim)``
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        device = next(self._sslam.parameters()).device

        # Compute fbank on CPU (kaldi ops), then move to device
        fbank = _waveform_to_fbank(waveform).to(device)  # (B, 1, T', 128)
        
        # Ensure fbank is contiguous BEFORE passing to SSLAM
        # This prevents .view() issues in backward pass when training all layers
        fbank = fbank.contiguous()

        # extract_features: (B, T', 768)
        frames = self._sslam.extract_features(fbank)
        # Ensure contiguous for backward compatibility with view() operations
        frames = frames.contiguous()
        return self._pool(frames)

    def _pool(self, frames: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "mean":
            return frames.mean(dim=1)
        elif self.pool_mode == "mean+std":
            return torch.cat([frames.mean(dim=1), frames.std(dim=1)], dim=-1)
        elif self.pool_mode == "first":
            return frames[:, 0, :]
        raise ValueError(f"Unknown pool mode: {self.pool_mode!r}")

    # ── Device handling ───────────────────────────────────────────────────

    def to(self, *args, **kwargs):
        self._sslam = self._sslam.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def checkpoint_state(self) -> dict:
        """Checkpoint format expected by the SSLAM inference loader."""
        model_state = {
            k[len("_sslam."):]: v
            for k, v in self.state_dict().items()
            if k.startswith("_sslam.")
        }
        return {
            "backbone_type": "sslam",
            "model_id":      _MODEL_ID,
            "model":         model_state,
        }
