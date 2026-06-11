"""
train_domain_autoencoder.py
───────────────────────────
Single-machine domain autoencoder.                                          project/simple_autoencoder/train_split_2ch_pre.py

Goal: encode one target machine's domains into distinct, separated clusters
in latent space. All other machines are collapsed to a single "other" class
(domain_label = 0) and are excluded from both clustering losses — the
autoencoder simply reconstructs them without trying to cluster them.

Domain label convention
  0          → "other" (every machine that is NOT the target machine)
  1 … K      → domain IDs for the target machine (re-indexed from 1)

The DomainClusteringLoss only operates on samples where domain_label > 0.

t-SNE plots are saved to /ceph/project/P8_DCASE/plots every N epochs.
"""

import sys
import os
import math
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── project imports ───────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(project_dir, "project", "utils"))

from dataloader import DCASE_Dataset
from model_architecture2 import DomainClusteringLoss

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "checkpoint_dir": "/ceph/project/P8_DCASE/models/2026/domain_autoencoder_bearingEmu_v2",
    "tsne_plot_dir":  "/ceph/project/P8_DCASE/plots/2026/bearingEmu_tsne",

    # Target machine — must match one of DCASE_Dataset.machines exactly:
    #   fan | valve | slider | ToyCar | ToyCarEmu | gearbox | bearing
    "target_machine": "bearingEmu",

    # Architecture
    "latent_dim": 256,
    "base_ch":    32,
    "dropout":    0.5,
    
    # Audio
    "sample_rate":       16_000,
    "max_audio_len_sec": 1.0,
    "train_crops_per_sample": 8, #

    # Optional RAM preload for faster repeated crops from the same files.
    "preload_audio_to_ram": True,
    "preload_train_only": True,
    "preload_max_ram_gb": 20.0,
    "preload_num_workers": 8,

    # Training
    "num_epochs":     300,
    "learning_rate":  1e-4,
    "weight_decay":   1e-4,
    "batch_size":     64,
    "num_workers":    4,
    "val_split":      0.20,
    "val_split_seed": 42,

    # Loss weights
    #   lambda_recon    – reconstruction fidelity (all samples)
    #   lambda_compact  – plain intra-cluster variance (set 0 to rely on proto only)
    #   lambda_separate – prototypical contrastive loss + hard centroid push
    #   lambda_repel    – push "other" samples away from target centroids
    "lambda_recon":    1.0,
    "lambda_compact":  0.0,
    "lambda_separate": 1.0,
    "lambda_repel":    0.8,

    # Softmax temperature for the prototypical loss (range 0.05–0.2).
    "proto_temperature": 0.1,

    # Minimum distance "other" samples must keep from any target centroid.
    "other_margin": 0.8,

    # Oversample target samples so clustering loss fires on most batches.
    "target_oversample": 10,

    # Temporal consistency loss: two non-overlapping crops of the same clip
    # should produce the same embedding.  Reduces within-domain fragmentation.
    # Set to 0.0 to disable.
    "lambda_consistency": 0.0,

    # Scheduler
    "scheduler_T_max":  300,
    "scheduler_eta_min": 1e-6,

    # t-SNE
    "tsne_every_n_epochs": 1,
    "tsne_max_samples":    3000,

    # AUC validation on the target machine test set
    "auc_val_n_neighbors": 2,
    "auc_val_reference_samples": 40_000,
    "auc_val_batch_size": 16,
    "auc_val_sliding_window": True,
    "auc_val_num_windows": 5,
    "auc_val_aggregation": "mean",
    # Recompute reference embeddings each epoch using the current model.
    # This makes train-time AUC directly comparable to offline inference.
    "auc_refresh_reference_each_epoch": True,

    # Early stopping
    "patience": 100,

    # Resume
    "resume_checkpoint": "/ceph/project/P8_DCASE/models/2026/domain_autoencoder_bearingEmu/best_model_auc.pt"  # Path to .pth to resume from, or empty string to start fresh.
}

local_env = False
if local_env:
    CONFIG["checkpoint_dir"] = r"C:\Users\emilr\Documents\GitHub\AAU_P8\project\models\domain_autoencoder"
    CONFIG["tsne_plot_dir"]   = r"C:\Users\emilr\Documents\GitHub\AAU_P8\project\plots"


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 stride: int = 2, padding: int = 3, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if residual and (in_ch != out_ch or stride != 1)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        if self.residual:
            skip = self.shortcut(x) if self.shortcut is not None else x
            out = out + skip
        return out


class DeconvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7,
                 stride: int = 2, padding: int = 3, output_padding: int = 1):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel,
            stride=stride, padding=padding,
            output_padding=output_padding, bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class StereoResidualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        ch = [3, base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.blocks = nn.ModuleList([
            ConvBlock1d(ch[i], ch[i + 1], kernel=7, stride=2,
                        padding=3, residual=(i > 0))
            for i in range(len(ch) - 1)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc_mu = nn.Linear(ch[-1], latent_dim)
        self.fc_log_var = nn.Linear(ch[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        h = self.pool(x).squeeze(-1)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = F.normalize(mu, p=2, dim=1)
        return z, mu, log_var


class StereoResidualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 output_samples: int = 80_000, out_channels: int = 3):
        super().__init__()
        self.output_samples = output_samples
        base_ch_top = base_ch * 16
        self.time_seed = math.ceil(output_samples / (2 ** 5))
        self.fc = nn.Linear(latent_dim, base_ch_top * self.time_seed)

        ch = [base_ch * 16, base_ch * 8, base_ch * 4, base_ch * 2, base_ch, out_channels]
        self.blocks = nn.ModuleList([
            DeconvBlock1d(ch[i], ch[i + 1], kernel=7, stride=2,
                          padding=3, output_padding=1)
            for i in range(len(ch) - 2)
        ])
        self.final_conv = nn.ConvTranspose1d(
            ch[-2], ch[-1], kernel_size=7,
            stride=2, padding=3, output_padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        x = self.fc(z)
        x = x.view(bsz, -1, self.time_seed)
        for block in self.blocks:
            x = block(x)
        x = torch.tanh(self.final_conv(x))
        if x.shape[-1] > self.output_samples:
            x = x[..., :self.output_samples]
        elif x.shape[-1] < self.output_samples:
            x = F.pad(x, (0, self.output_samples - x.shape[-1]))
        return x


class DomainAutoencoder(nn.Module):
    """
    Stereo autoencoder with explicit residual input.
    Input channels must be [ch1, ch2, ch1-ch2].
    """

    def __init__(self, latent_dim: int = 128, base_ch: int = 32,
                 sample_rate: int = 16_000, max_audio_sec: float = 5.0,
                 dropout: float = 0.0):
        super().__init__()
        output_samples = int(sample_rate * max_audio_sec)
        self.encoder = StereoResidualEncoder(
            latent_dim=latent_dim,
            base_ch=base_ch,
            dropout=dropout,
        )
        self.decoder = StereoResidualDecoder(
            latent_dim=latent_dim,
            base_ch=base_ch,
            output_samples=output_samples,
            out_channels=3,
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # Backward compatibility: allow mono/stereo tensors by expanding to 3 channels.
        if x.dim() == 3 and x.shape[1] == 1:
            ch1 = x[:, 0, :]
            ch2 = x[:, 0, :]
            res = ch1 - ch2
            x = torch.stack([ch1, ch2, res], dim=1)
        elif x.dim() == 3 and x.shape[1] == 2:
            ch1 = x[:, 0, :]
            ch2 = x[:, 1, :]
            res = ch1 - ch2
            x = torch.stack([ch1, ch2, res], dim=1)
        return x

    def forward(self, x: torch.Tensor):
        x = self._prepare_input(x)
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return {"z": z, "mu": mu, "log_var": log_var, "x_hat": x_hat}

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["z"]


class StereoReconstructionLoss(nn.Module):
    """Reconstruction loss for multi-channel waveforms (B, C, T)."""

    def __init__(self, fft_sizes: tuple = (512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(1)

        loss_mse = F.mse_loss(x_hat, x)

        bsz, channels, time_steps = x.shape
        x_flat = x.reshape(bsz * channels, time_steps)
        x_hat_flat = x_hat.reshape(bsz * channels, time_steps)

        loss_spec = torch.tensor(0.0, device=x.device)
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=x.device)
            x_stft = torch.stft(x_flat, n_fft=n_fft, hop_length=hop,
                                return_complex=True, window=window)
            x_hat_stft = torch.stft(x_hat_flat, n_fft=n_fft, hop_length=hop,
                                    return_complex=True, window=window)
            log_x = torch.log(x_stft.abs() + 1e-7)
            log_x_hat = torch.log(x_hat_stft.abs() + 1e-7)
            loss_spec = loss_spec + F.l1_loss(log_x_hat, log_x)

        return loss_mse + (loss_spec / len(self.fft_sizes))


class StereoTemporalConsistencyLoss(nn.Module):
    """Cosine consistency between first-half and second-half stereo crops."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, model: DomainAutoencoder, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        t_steps = x.shape[-1]
        mid = t_steps // 2
        crop_a = x[..., :mid]
        crop_b = x[..., mid:]

        z_a = model.get_embeddings(crop_a)
        z_b = model.get_embeddings(crop_b)
        cos_sim = (z_a * z_b).sum(dim=1)
        return (1.0 - cos_sim / self.temperature).mean()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DCASEAudioDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, sample_rate: int = 16_000,
                 window_sec: float = 5.0, mode: str = "train",
                 crops_per_sample: int = 1,
                 preload_to_ram: bool = False,
                 preload_max_ram_bytes: Optional[int] = None,
                 preload_num_workers: int = 1):
        self.samples = samples
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)
        self.mode = mode
        self.crops_per_sample = max(1, int(crops_per_sample)) if mode == "train" else 1
        self._dcase = DCASE_Dataset()
        self.preload_to_ram = bool(preload_to_ram)
        self.preload_num_workers = max(1, int(preload_num_workers))
        self.preload_max_ram_bytes = (
            int(preload_max_ram_bytes)
            if preload_max_ram_bytes is not None
            else None
        )
        self._audio_cache: Dict[str, np.ndarray] = {}
        self._cached_bytes = 0

        if self.preload_to_ram:
            self._preload_audio_cache()

    def __len__(self) -> int:
        return len(self.samples) * self.crops_per_sample

    def _crop(self, audio: np.ndarray) -> np.ndarray:
        win = self.window_samples
        if audio.ndim == 1:
            audio = audio[None, :]

        if audio.shape[-1] <= win:
            pad = win - audio.shape[-1]
            return np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        start = (
            np.random.randint(0, audio.shape[-1] - win)
            if self.mode == "train"
            else (audio.shape[-1] - win) // 2
        )
        return audio[:, start: start + win]

    def _load_stereo_audio(self, file_path: str):
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
            return audio, sr
        except Exception:
            return None, None

    def _resample_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.sample_rate:
            return audio
        if audio.ndim == 1:
            return librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return np.stack([
            librosa.resample(audio[ch], orig_sr=sr, target_sr=self.sample_rate)
            for ch in range(audio.shape[0])
        ], axis=0)

    def _prepare_base_audio(self, file_path: str) -> np.ndarray:
        audio, sr = self._load_stereo_audio(file_path)
        if audio is None:
            return np.zeros((2, self.window_samples), dtype=np.float32)
        audio = self._resample_audio(audio, sr).astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio

    def _preload_audio_cache(self) -> None:
        unique_paths = []
        seen = set()
        for path, _, _ in self.samples:
            if path in seen:
                continue
            seen.add(path)
            unique_paths.append(path)

        use_parallel = self.preload_num_workers > 1 and len(unique_paths) > 1
        loaded = 0
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.preload_num_workers) as executor:
                for path, audio in zip(unique_paths, executor.map(self._prepare_base_audio, unique_paths)):
                    n_bytes = int(audio.nbytes)

                    if (
                        self.preload_max_ram_bytes is not None
                        and self._cached_bytes + n_bytes > self.preload_max_ram_bytes
                    ):
                        logger.info(
                            f"[{self.mode}] RAM preload cap reached at "
                            f"{loaded}/{len(unique_paths)} files "
                            f"({self._cached_bytes / (1024 ** 3):.2f} GiB cached)."
                        )
                        break

                    self._audio_cache[path] = audio
                    self._cached_bytes += n_bytes
                    loaded += 1
        else:
            for path in unique_paths:
                audio = self._prepare_base_audio(path)
                n_bytes = int(audio.nbytes)

                if (
                    self.preload_max_ram_bytes is not None
                    and self._cached_bytes + n_bytes > self.preload_max_ram_bytes
                ):
                    logger.info(
                        f"[{self.mode}] RAM preload cap reached at "
                        f"{loaded}/{len(unique_paths)} files "
                        f"({self._cached_bytes / (1024 ** 3):.2f} GiB cached)."
                    )
                    break

                self._audio_cache[path] = audio
                self._cached_bytes += n_bytes
                loaded += 1

        logger.info(
            f"[{self.mode}] RAM preload: {loaded}/{len(unique_paths)} files "
            f"cached ({self._cached_bytes / (1024 ** 3):.2f} GiB)."
        )

    def _get_base_audio(self, file_path: str) -> np.ndarray:
        if not self.preload_to_ram:
            return self._prepare_base_audio(file_path)
        audio = self._audio_cache.get(file_path)
        if audio is not None:
            return audio
        return self._prepare_base_audio(file_path)

    def _to_three_channel_input(self, audio: np.ndarray) -> np.ndarray:
        # Accept mono/stereo and always convert to [ch1, ch2, ch1-ch2].
        if audio.ndim == 1:
            ch1 = audio.astype(np.float32)
            ch2 = audio.astype(np.float32)
        else:
            if audio.shape[0] == 1:
                ch1 = audio[0].astype(np.float32)
                ch2 = audio[0].astype(np.float32)
            else:
                ch1 = audio[0].astype(np.float32)
                ch2 = audio[1].astype(np.float32)
        residual = ch1 - ch2
        return np.stack([ch1, ch2, residual], axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        base_idx = idx // self.crops_per_sample
        file_path, machine_label, domain_label = self.samples[base_idx]

        audio = self._get_base_audio(file_path)
        audio = self._crop(audio)
        audio = self._to_three_channel_input(audio)
        return torch.from_numpy(audio), int(machine_label), int(domain_label)


class AUCValidationDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, sample_rate: int = 16_000,
                 window_sec: float = 5.0, use_sliding_windows: bool = False,
                 num_windows: int = 5):
        self.samples = samples
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)
        self._dcase = DCASE_Dataset()
        self.use_sliding_windows = bool(use_sliding_windows)
        self.num_windows = max(1, int(num_windows))

        if self.use_sliding_windows:
            self.window_to_sample_idx = []
            for sample_idx in range(len(self.samples)):
                for window_idx in range(self.num_windows):
                    self.window_to_sample_idx.append((sample_idx, window_idx))
        else:
            self.window_to_sample_idx = [(i, 0) for i in range(len(self.samples))]

    def __len__(self) -> int:
        return len(self.window_to_sample_idx)

    def _sliding_window_crop(self, audio: np.ndarray, window_idx: int) -> np.ndarray:
        win = self.window_samples
        if audio.shape[-1] <= win:
            pad = win - audio.shape[-1]
            return np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        total_length = audio.shape[-1]
        if self.num_windows <= 1:
            start = (total_length - win) // 2
        else:
            stride = max(1, (total_length - win) // (self.num_windows - 1))
            start = min(window_idx * stride, total_length - win)
        return audio[:, start:start + win]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_idx, window_idx = self.window_to_sample_idx[idx]
        file_path, label = self.samples[sample_idx]

        try:
            audio, sr = librosa.load(file_path, sr=None, mono=False)
        except Exception:
            audio, sr = None, None

        if audio is None:
            audio = np.zeros((2, self.window_samples), dtype=np.float32)
            sr = self.sample_rate

        if sr != self.sample_rate:
            if audio.ndim == 1:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            else:
                audio = np.stack([
                    librosa.resample(audio[ch], orig_sr=sr, target_sr=self.sample_rate)
                    for ch in range(audio.shape[0])
                ], axis=0)

        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)

        audio = audio.astype(np.float32)
        if self.use_sliding_windows:
            audio = self._sliding_window_crop(audio, window_idx)
        else:
            if audio.shape[-1] > self.window_samples:
                start = (audio.shape[-1] - self.window_samples) // 2
                audio = audio[:, start:start + self.window_samples]
            elif audio.shape[-1] < self.window_samples:
                pad = self.window_samples - audio.shape[-1]
                audio = np.pad(audio, ((0, 0), (0, pad)), mode="reflect")

        ch1 = audio[0]
        ch2 = audio[1]
        residual = ch1 - ch2
        audio_3ch = np.stack([ch1, ch2, residual], axis=0).astype(np.float32)
        return torch.from_numpy(audio_3ch), int(label), int(sample_idx)


def _auc_window_settings(cfg: dict) -> Tuple[bool, int]:
    """AUC validation uses overlapping windows with a minimum of 5 per clip."""
    num_windows = max(5, int(cfg.get("auc_val_num_windows", 5)))
    return True, num_windows


def _is_anomaly(filename: str) -> int:
    name = filename.lower()
    if "anomaly" in name or "abnormal" in name:
        return 1
    if "normal" in name:
        return 0
    return -1


def _find_machine_root(data_root: str, machine: str) -> Optional[str]:
    if not os.path.exists(data_root):
        return None
    return next((d for d in os.listdir(data_root) if d.lower() == machine.lower()), None)


def _path_has_machine_segment(path: str, machine: str) -> bool:
    machine_l = machine.lower()
    parts = os.path.normpath(path).replace("\\", "/").split("/")
    return any(part.lower() == machine_l for part in parts)


def discover_target_test_samples(machine: str, data_root: str) -> list:
    samples = []
    dcase = DCASE_Dataset()
    machine_root = _find_machine_root(data_root, machine)
    if machine_root is None:
        logger.warning(f"No data folder found for machine '{machine}' in {data_root}")
        return samples

    test_dir = None
    for candidate in ("test", "test_data"):
        path = os.path.join(data_root, machine_root, candidate)
        if os.path.isdir(path):
            test_dir = path
            break

    if test_dir is None:
        logger.warning(f"No test folder found for '{machine}'")
        return samples

    for fname in os.listdir(test_dir):
        if not fname.endswith(".wav"):
            continue
        label = _is_anomaly(fname)
        if label < 0:
            continue
        path = os.path.join(test_dir, fname)
        domain_str = dcase.build_domain(fname, machine)
        samples.append((path, 1, domain_str))

    logger.info(f"Target test: {len(samples)} files for '{machine}'")
    return samples


def discover_target_test_auc_samples(machine: str, data_root: str) -> list:
    samples = []
    machine_root = _find_machine_root(data_root, machine)
    if machine_root is None:
        return samples

    test_dir = None
    for candidate in ("test", "test_data"):
        path = os.path.join(data_root, machine_root, candidate)
        if os.path.isdir(path):
            test_dir = path
            break

    if test_dir is None:
        return samples

    for fname in os.listdir(test_dir):
        if not fname.endswith(".wav"):
            continue
        label = _is_anomaly(fname)
        if label < 0:
            continue
        samples.append((os.path.join(test_dir, fname), label))
    return samples


def aggregate_windowed_embeddings(model, dataloader, device, aggregation: str,
                                  return_labels: bool):
    model.eval()
    window_embeddings = {}
    with torch.no_grad():
        for audio, labels, sample_idxs in dataloader:
            audio = audio.to(device)
            z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
            emb_np = z.cpu().numpy()
            if isinstance(sample_idxs, torch.Tensor):
                sample_idxs = sample_idxs.tolist()
            for i, sample_idx in enumerate(sample_idxs):
                if sample_idx not in window_embeddings:
                    window_embeddings[sample_idx] = []
                window_embeddings[sample_idx].append(emb_np[i])

    if not window_embeddings:
        return (None, None) if return_labels else None

    samples = getattr(dataloader.dataset, "samples", None)
    emb_list = []
    lbl_list = []
    for sample_idx in sorted(window_embeddings.keys()):
        windows = np.array(window_embeddings[sample_idx])
        if aggregation == "max":
            agg_emb = np.max(windows, axis=0)
        else:
            agg_emb = np.mean(windows, axis=0)
        emb_list.append(agg_emb)
        if return_labels and samples is not None and sample_idx < len(samples):
            lbl_list.append(samples[sample_idx][1])

    emb = np.vstack(emb_list)
    if return_labels:
        labels = np.array(lbl_list, dtype=int)
        return emb, labels
    return emb


def generate_reference_embeddings(model, dataloader, device, cfg, limit: int = 40_000):
    use_windows, num_windows = _auc_window_settings(cfg)
    if use_windows:
        samples = getattr(dataloader.dataset, "samples", None)
        if not samples:
            return None
        ref_samples = [(path, 0) for path, *_ in samples][:limit]
        ref_ds = AUCValidationDataset(
            ref_samples,
            cfg["sample_rate"],
            cfg["max_audio_len_sec"],
            use_sliding_windows=True,
            num_windows=num_windows,
        )
        ref_loader = DataLoader(
            ref_ds,
            batch_size=cfg.get("auc_val_batch_size", 16),
            shuffle=False,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=True,
        )
        return aggregate_windowed_embeddings(
            model,
            ref_loader,
            device,
            aggregation=str(cfg.get("auc_val_aggregation", "mean")).lower(),
            return_labels=False,
        )

    model.eval()
    embeddings = []
    seen = 0
    with torch.no_grad():
        for audio, _, _ in dataloader:
            if seen >= limit:
                break
            audio = audio.to(device)
            z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
            embeddings.append(z.cpu().numpy())
            seen += len(audio)
    if not embeddings:
        return None
    return np.vstack(embeddings)[:limit]


def validate_auc(model, dataloader, reference_embeddings, device, cfg, rank: int = 0):
    if dataloader is None or reference_embeddings is None or len(reference_embeddings) == 0:
        return 0.0

    use_windows, _ = _auc_window_settings(cfg)
    if use_windows:
        q_emb, q_lbl = aggregate_windowed_embeddings(
            model,
            dataloader,
            device,
            aggregation=str(cfg.get("auc_val_aggregation", "mean")).lower(),
            return_labels=True,
        )
        if q_emb is None or q_lbl is None:
            return 0.0
    else:
        model.eval()
        query_embeddings, query_labels = [], []
        with torch.no_grad():
            for audio, labels in dataloader:
                audio = audio.to(device)
                z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
                query_embeddings.append(z.cpu().numpy())
                query_labels.append(labels.numpy())

        if not query_embeddings:
            return 0.0

        q_emb = np.vstack(query_embeddings)
        q_lbl = np.concatenate(query_labels)
    if len(np.unique(q_lbl)) < 2:
        logger.warning("AUC validation skipped because the target test set has only one class.")
        return 0.5

    n_neighbors = min(cfg.get("auc_val_n_neighbors", 2), len(reference_embeddings))
    n_neighbors = max(1, n_neighbors)

    scaler = StandardScaler()
    ref_scaled = scaler.fit_transform(reference_embeddings)
    q_scaled = scaler.transform(q_emb)

    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn_model.fit(ref_scaled)
    distances, _ = nn_model.kneighbors(q_scaled)
    scores = distances[:, 0]

    try:
        auc = roc_auc_score(q_lbl, scores)
    except ValueError as exc:
        logger.warning(f"AUC calculation failed: {exc}")
        return 0.5

    if rank == 0:
        n_norm = int((q_lbl == 0).sum())
        n_anom = int((q_lbl == 1).sum())
        logger.info(f"   AUC: {auc:.4f}  ({n_norm} normal / {n_anom} anomaly)")
    return auc


def build_datasets(cfg: dict, rank: int = 0):
    target = cfg["target_machine"].lower()
    dcase  = DCASE_Dataset()
    all_samples = dcase.discover_train()
    target_test_samples_raw = discover_target_test_samples(target, dcase.data_path)

    inv_domain_map = {v: k for k, v in dcase.domain_map.items()}

    target_domain_strings = sorted({
        inv_domain_map[dom_id]
        for path, _, dom_id in all_samples
        if _path_has_machine_segment(path, target) and dom_id in inv_domain_map
    })

    domain_str_to_new_id = {s: i + 1 for i, s in enumerate(target_domain_strings)}
    domain_label_to_name = {0: "other"}
    domain_label_to_name.update({v: k for k, v in domain_str_to_new_id.items()})

    target_test_samples = []
    for path, machine_label, domain_str in target_test_samples_raw:
        new_dom = domain_str_to_new_id.get(domain_str, 0)
        target_test_samples.append((path, machine_label, new_dom))

    if rank == 0:
        logger.info(f"Target machine  : '{target}'")
        logger.info(f"Target domains  : {len(target_domain_strings)}")
        for new_id, name in sorted(domain_label_to_name.items()):
            logger.info(f"  label {new_id:3d} → {name}")

    target_samples, other_samples, aug_samples = [], [], []

    for path, class_id, dom_id in all_samples:
        filename  = os.path.basename(path)
        is_target = _path_has_machine_segment(path, target)

        if is_target:
            dom_str       = inv_domain_map.get(dom_id, "")
            new_dom       = domain_str_to_new_id.get(dom_str, 0)
            machine_label = 1
        else:
            new_dom       = 0
            machine_label = 0

        entry = (path, machine_label, new_dom)

        if dcase.detect_augmentation(filename) != "original":
            aug_samples.append(entry)
        elif is_target:
            target_samples.append(entry)
        else:
            other_samples.append(entry)

    oversample    = cfg.get("target_oversample", 10)
    train_samples = target_samples * oversample + other_samples + aug_samples
    val_samples   = target_test_samples

    if rank == 0:
        logger.info(
            f"Train  — target: {len(target_samples)} x{oversample}, "
            f"other: {len(other_samples)}, aug: {len(aug_samples)}"
        )
        logger.info(
            f"Train crops/sample: {cfg.get('train_crops_per_sample', 1)} "
            f"(effective train size: {len(train_samples) * cfg.get('train_crops_per_sample', 1)})"
        )
        logger.info(f"Val    — target test: {len(target_test_samples)}")

    preload_enabled = bool(cfg.get("preload_audio_to_ram", False))
    preload_train_only = bool(cfg.get("preload_train_only", True))
    preload_max_ram_gb = cfg.get("preload_max_ram_gb", None)
    preload_max_ram_bytes = None
    if preload_max_ram_gb is not None:
        preload_max_ram_bytes = int(float(preload_max_ram_gb) * (1024 ** 3))

    if rank == 0 and preload_enabled:
        cap_str = (
            f"{float(preload_max_ram_gb):.2f} GiB"
            if preload_max_ram_gb is not None
            else "unbounded"
        )
        logger.info(
            f"RAM preload enabled (train_only={preload_train_only}, cap={cap_str})."
        )

    window_sec = cfg["max_audio_len_sec"]
    train_ds   = DCASEAudioDataset(
        train_samples,
        cfg["sample_rate"],
        window_sec,
        "train",
        crops_per_sample=cfg.get("train_crops_per_sample", 1),
        preload_to_ram=preload_enabled,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    preload_eval = preload_enabled and not preload_train_only
    val_ds     = DCASEAudioDataset(
        val_samples,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    reference_ds = DCASEAudioDataset(
        target_samples,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )
    auc_val_ds = AUCValidationDataset(
        discover_target_test_auc_samples(target, dcase.data_path),
        cfg["sample_rate"], window_sec,
        use_sliding_windows=cfg.get("auc_val_sliding_window", True),
        num_windows=cfg.get("auc_val_num_windows", 5),
    )

    # t-SNE dataset: all originals without oversampling for a faithful picture
    tsne_raw = target_test_samples + target_samples + other_samples[:len(target_test_samples) + len(target_samples)]
    tsne_ds  = DCASEAudioDataset(
        tsne_raw,
        cfg["sample_rate"],
        window_sec,
        "val",
        preload_to_ram=preload_eval,
        preload_max_ram_bytes=preload_max_ram_bytes,
        preload_num_workers=cfg.get("preload_num_workers", 1),
    )

    return train_ds, val_ds, tsne_ds, reference_ds, auc_val_ds, domain_label_to_name


# ─────────────────────────────────────────────────────────────────────────────
# t-SNE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def collect_embeddings(model, dataloader, device, max_samples: int = 3000):
    model.eval()
    zs, domains = [], []
    count = 0
    with torch.no_grad():
        for audio, _, domain_labels in dataloader:
            if count >= max_samples:
                break
            audio = audio.to(device)
            out   = model.module(audio) if hasattr(model, "module") else model(audio)
            zs.append(out["z"].cpu().numpy())
            domains.append(domain_labels.numpy())
            count += len(audio)
    if not zs:
        return None, None
    return np.vstack(zs)[:max_samples], np.concatenate(domains)[:max_samples]


def save_tsne_plot(model, dataloader, device, epoch: int,
                   domain_label_to_name: dict, plot_dir: str,
                   max_samples: int = 3000):
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Generating t-SNE plot for epoch {epoch} …")

    zs, domains = collect_embeddings(model, dataloader, device, max_samples)
    if zs is None:
        logger.warning("No embeddings collected — skipping t-SNE.")
        return

    present       = sorted(np.unique(domains).astype(int))
    present_names = [domain_label_to_name.get(d, f"domain {d}") for d in present]
    logger.info(f"   t-SNE covers {len(zs)} samples, {len(present)} labels: {present_names}")

    zs_scaled = StandardScaler().fit_transform(zs)
    tsne = TSNE(n_components=2, perplexity=min(30, len(zs) - 1),
                max_iter=1000, random_state=seed, init="pca")
    coords = tsne.fit_transform(zs_scaled)

    unique_domains   = sorted(np.unique(domains).astype(int))
    n_target_domains = sum(1 for d in unique_domains if d > 0)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(n_target_domains, 1))

    fig, ax = plt.subplots(figsize=(9, 7))

    if 0 in unique_domains:
        mask = domains == 0
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c="lightgrey", s=8, alpha=0.35,
                   label="other (domain 0)", zorder=1)

    colour_idx = 0
    for d in unique_domains:
        if d == 0:
            continue
        mask  = domains == d
        label = domain_label_to_name.get(d, f"domain {d}")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(colour_idx)], s=22, alpha=0.75,
                   label=label, zorder=2)
        colour_idx += 1

    ax.set_title(f"t-SNE — domain embeddings — Epoch {epoch}", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="upper right", markerscale=2, fontsize=7,
              ncol=max(1, n_target_domains // 20))
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    save_path = os.path.join(plot_dir, f"tsne_epoch_{epoch:04d}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"   t-SNE plot saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def log_loss_breakdown(epoch, batch_idx, total, recon, compact, separate, repel, consist):
    logger.info(
        f"Epoch {epoch} | Batch {batch_idx:4d} | "
        f"Total: {total:.4f}  Recon: {recon:.4f}  "
        f"Compact: {compact:.4f}  Separate: {separate:.4f}  "
        f"Repel: {repel:.4f}  Consist: {consist:.4f}"
    )


def run_epoch(model, dataloader, optimizer, device, epoch, cfg, rank,
              training: bool):
    model.train() if training else model.eval()

    recon_crit        = StereoReconstructionLoss().to(device)
    cluster_crit      = DomainClusteringLoss(
        temperature=cfg["proto_temperature"],
        other_margin=cfg["other_margin"],
    ).to(device)
    consistency_crit  = StereoTemporalConsistencyLoss().to(device)

    total_loss = 0.0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for batch_idx, (audio, machine_labels, domain_labels) in enumerate(dataloader):
            audio         = audio.to(device)
            domain_labels = domain_labels.to(device).long()

            if training:
                optimizer.zero_grad()

            out   = model(audio)
            z     = out["z"]        # (B, latent_dim)
            x_hat = out["x_hat"]    # (B, 3, T)
            x_in  = audio

            # Reconstruction loss: all samples (target + other)
            loss_recon = recon_crit(x_in, x_hat)

            # Temporal consistency loss: two crops of the same clip → same z.
            # Passed as (model, raw_audio) so the loss re-runs the encoder on
            # both halves internally.  The decoder is NOT run on the crops —
            # this is encoder-only and adds no extra decoder computation.
            if cfg.get("lambda_consistency", 0.0) > 0:
                loss_consistency = consistency_crit(
                    model.module if hasattr(model, "module") else model,
                    audio
                )
            else:
                loss_consistency = torch.tensor(0.0, device=device)

            # Clustering losses: pass the full z batch so the repulsion term
            # can act on "other" samples too.
            # DomainClusteringLoss returns exactly 3 values:
            #   loss_compact  – intra-cluster variance (monitoring)
            #   loss_separate – proto contrastive + hard centroid push
            #   loss_repel    – push "other" away from target centroids
            if (domain_labels > 0).sum() >= 2:
                loss_compact, loss_separate, loss_repel = cluster_crit(z, domain_labels)
            else:
                loss_compact  = torch.tensor(0.0, device=device)
                loss_separate = torch.tensor(0.0, device=device)
                loss_repel    = torch.tensor(0.0, device=device)

            loss = (
                cfg["lambda_recon"]        * loss_recon       +
                cfg["lambda_compact"]      * loss_compact     +
                cfg["lambda_separate"]     * loss_separate    +
                cfg["lambda_repel"]        * loss_repel       +
                cfg["lambda_consistency"]  * loss_consistency
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            if training and rank == 0 and (batch_idx + 1) % 20 == 0:
                log_loss_breakdown(
                    epoch, batch_idx + 1,
                    loss.item(), loss_recon.item(),
                    loss_compact.item(), loss_separate.item(),
                    loss_repel.item(), loss_consistency.item(),
                )

    return total_loss / max(len(dataloader), 1)


def training_loop(model, train_loader, val_loader, tsne_loader, auc_val_loader,
                  reference_loader, optimizer, scheduler, device, cfg, rank,
                  domain_label_to_name):
    ckpt_dir = cfg["checkpoint_dir"]
    plot_dir = cfg["tsne_plot_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir,  exist_ok=True)

    best_model_path  = os.path.join(ckpt_dir, "best_model_auc.pt")
    best_auc         = float("-inf")
    best_val_loss    = float("inf")
    patience_counter = 0
    stop             = False
    use_auc = (
        auc_val_loader is not None
        and reference_loader is not None
        and len(auc_val_loader.dataset) > 0
        and len(reference_loader.dataset) > 0
    )
    reference_embeddings_fixed = None
    if rank == 0 and use_auc and not cfg.get("auc_refresh_reference_each_epoch", True):
        reference_embeddings_fixed = generate_reference_embeddings(
            model,
            reference_loader,
            device,
            cfg,
            limit=cfg["auc_val_reference_samples"],
        )

    def save(epoch, suffix=""):
        path  = os.path.join(ckpt_dir, f"model_epoch_{epoch}{suffix}.pt")
        state = model.module.state_dict() if hasattr(model, "module") \
                else model.state_dict()
        torch.save(state, path)
        logger.info(f"   Checkpoint: {path}")

    for epoch in range(1, cfg["num_epochs"] + 1):
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        train_loss = run_epoch(model, train_loader, optimizer,
                               device, epoch, cfg, rank, training=True)
        val_loss   = run_epoch(model, val_loader,   None,
                               device, epoch, cfg, rank, training=False)
        current_auc = 0.0

        if rank == 0 and use_auc:
            if cfg.get("auc_refresh_reference_each_epoch", True):
                reference_embeddings = generate_reference_embeddings(
                    model,
                    reference_loader,
                    device,
                    cfg,
                    limit=cfg["auc_val_reference_samples"],
                )
            else:
                reference_embeddings = reference_embeddings_fixed

            current_auc = validate_auc(
                model, auc_val_loader, reference_embeddings,
                device, cfg, rank=rank,
            )

        if rank == 0:
            logger.info(
                f"Epoch {epoch}/{cfg['num_epochs']} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"AUC: {current_auc:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if epoch % 5 == 0:
                save(epoch)

            if epoch % cfg["tsne_every_n_epochs"] == 0:
                save_tsne_plot(model, tsne_loader, device, epoch,
                               domain_label_to_name, plot_dir,
                               max_samples=cfg["tsne_max_samples"])

            improved = False
            if use_auc:
                if current_auc > best_auc:
                    best_auc = current_auc
                    state = model.module.state_dict() if hasattr(model, "module") \
                            else model.state_dict()
                    torch.save(state, best_model_path)
                    logger.info(f"   ★ NEW BEST AUC: {best_auc:.4f}")
                    improved = True
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state = model.module.state_dict() if hasattr(model, "module") \
                            else model.state_dict()
                    torch.save(state, best_model_path)
                    logger.info(f"   ★ NEW BEST Val Loss: {best_val_loss:.4f}")
                    improved = True

            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"   No improvement for {patience_counter} epoch(s).")
                if patience_counter >= cfg["patience"]:
                    logger.info("   Early stopping triggered.")
                    stop = True

        if dist.is_initialized():
            stop_t = torch.tensor([int(stop)], device=device)
            dist.broadcast(stop_t, src=0)
            stop = bool(stop_t.item())
        if stop:
            break

        scheduler.step()

    if rank == 0:
        if use_auc:
            logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
        else:
            logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        logger.info(f"Best model: {best_model_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device     = torch.device("cuda", local_rank)
        if dist.get_rank() == 0:
            logger.info(f"DDP — world_size={dist.get_world_size()}")
    else:
        rank   = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Single-process — device={device}")

    cfg = CONFIG
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["tsne_plot_dir"],  exist_ok=True)

    if rank == 0:
        logger.info("CONFIG:")
        for k, v in cfg.items():
            logger.info(f"  {k}: {v}")

    train_ds, val_ds, tsne_ds, reference_ds, auc_val_ds, domain_label_to_name = build_datasets(cfg, rank=rank)
    cfg["num_domains"] = len(domain_label_to_name) - 1

    train_sampler = DistributedSampler(train_ds, shuffle=True) \
        if dist.is_initialized() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) \
        if dist.is_initialized() else None

    loader_num_workers = cfg["num_workers"]
    if cfg.get("preload_audio_to_ram", False) and os.name == "nt" and loader_num_workers > 0:
        logger.warning(
            "Windows uses spawn workers; RAM preload would be duplicated per worker. "
            "Setting num_workers=0 to avoid excessive memory use."
        )
        loader_num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=loader_num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], sampler=val_sampler,
        shuffle=False, num_workers=loader_num_workers,
        pin_memory=True, drop_last=False,
    )
    tsne_loader = DataLoader(
        tsne_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=loader_num_workers,
        pin_memory=True, drop_last=False,
    )
    reference_loader = DataLoader(
        reference_ds, batch_size=cfg["auc_val_batch_size"],
        shuffle=False, num_workers=loader_num_workers,
        pin_memory=True, drop_last=False,
    )
    auc_val_loader = DataLoader(
        auc_val_ds, batch_size=cfg["auc_val_batch_size"],
        shuffle=False, num_workers=loader_num_workers,
        pin_memory=True, drop_last=False,
    )

    model = DomainAutoencoder(
        latent_dim=cfg["latent_dim"],
        base_ch=cfg["base_ch"],
        sample_rate=cfg["sample_rate"],
        max_audio_sec=cfg["max_audio_len_sec"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)

    resume = cfg.get("resume_checkpoint", "")
    if resume and os.path.isfile(resume):
        sd = torch.load(resume, map_location=device)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded checkpoint: {resume}")

    if dist.is_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["scheduler_T_max"],
        eta_min=cfg["scheduler_eta_min"],
    )

    training_loop(
        model, train_loader, val_loader, tsne_loader, auc_val_loader,
        reference_loader, optimizer, scheduler, device, cfg, rank,
        domain_label_to_name,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()