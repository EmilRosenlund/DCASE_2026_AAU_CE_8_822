"""Waveform augmentations for ArcFace fine-tuning.

All functions operate on ``(1, T)`` float32 tensors.
Each has a ``prob`` parameter; if the random draw fails, the input
is returned unchanged.
"""

from __future__ import annotations

import random
from typing import Optional

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentations
# ─────────────────────────────────────────────────────────────────────────────

def time_shift(
    waveform: torch.Tensor,
    max_fraction: float = 0.1,
    prob: float = 0.5,
) -> torch.Tensor:
    """Circular time shift by up to *max_fraction* of the clip length."""
    if random.random() >= prob:
        return waveform
    T = waveform.shape[-1]
    shift = random.randint(0, max(1, int(T * max_fraction)))
    return torch.roll(waveform, shifts=shift, dims=-1)


def add_gaussian_noise(
    waveform: torch.Tensor,
    snr_db_low: float = 15.0,
    snr_db_high: float = 35.0,
    prob: float = 0.3,
) -> torch.Tensor:
    """Add white Gaussian noise at a random SNR in [snr_db_low, snr_db_high]."""
    if random.random() >= prob:
        return waveform
    snr_db = random.uniform(snr_db_low, snr_db_high)
    signal_power = waveform.pow(2).mean().clamp(min=1e-8)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return (waveform + noise).clamp(-1.0, 1.0)


def mix_with_real_noise(
    waveform: torch.Tensor,
    noise_waveform: torch.Tensor,
    snr_db_low: float = 5.0,
    snr_db_high: float = 20.0,
    prob: float = 0.8,
) -> torch.Tensor:
    """Mix a clean clip with a pre-loaded noise clip at a random SNR.

    The noise clip is randomly cropped / tiled to match the signal length.
    """
    if random.random() >= prob:
        return waveform

    T = waveform.shape[-1]
    noise = noise_waveform

    # Tile noise if shorter than signal
    if noise.shape[-1] < T:
        repeats = (T // noise.shape[-1]) + 1
        noise = noise.repeat(1, repeats)

    # Random crop
    offset = random.randint(0, noise.shape[-1] - T)
    noise = noise[..., offset : offset + T]

    # Scale to target SNR
    sig_rms   = waveform.pow(2).mean().clamp(min=1e-8).sqrt()
    noise_rms = noise.pow(2).mean().clamp(min=1e-8).sqrt()
    snr_db    = random.uniform(snr_db_low, snr_db_high)
    target_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_rms / noise_rms)

    return (waveform + noise).clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Composed augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class WaveformAugmentor:
    """Apply a chain of waveform augmentations driven by a config dict.

    Parameters
    ----------
    aug_cfg:
        The ``augmentation`` block of the YAML config (as a plain dict or
        an ``OmegaConf`` DictConfig object — both support attribute access
        via ``getattr``).
    noise_waveforms:
        Optional list of pre-loaded noise tensors for real-noise mixing.
    """

    def __init__(self, aug_cfg, noise_waveforms: Optional[list] = None) -> None:
        self.cfg = aug_cfg
        self.noise_waveforms = noise_waveforms or []

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        # Real-noise mixing
        rn = _get(cfg, "real_noise")
        if rn and _get(rn, "enabled", True) and self.noise_waveforms:
            noise = random.choice(self.noise_waveforms)
            waveform = mix_with_real_noise(
                waveform,
                noise,
                snr_db_low=_get(rn, "snr_db_low", 5.0),
                snr_db_high=_get(rn, "snr_db_high", 20.0),
                prob=_get(rn, "prob", 0.8),
            )

        # Time shift
        ts = _get(cfg, "time_shift")
        if ts and _get(ts, "enabled", True):
            waveform = time_shift(
                waveform,
                max_fraction=_get(ts, "max_fraction", 0.1),
                prob=_get(ts, "prob", 0.5),
            )

        # Gaussian noise
        gn = _get(cfg, "gaussian_noise")
        if gn and _get(gn, "enabled", True):
            waveform = add_gaussian_noise(
                waveform,
                snr_db_low=_get(gn, "snr_db_low", 15.0),
                snr_db_high=_get(gn, "snr_db_high", 35.0),
                prob=_get(gn, "prob", 0.3),
            )

        return waveform


def _get(obj, key, default=None):
    """Attribute-or-key access that works for both dicts and OmegaConf nodes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
