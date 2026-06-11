"""
train_domain_autoencoder.py

Single-machine domain autoencoder.                                          project/simple_autoencoder/train_split_2ch_pre_self.py
"""
import sys
import os
import math
import logging
import random
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
from sklearn.cluster import KMeans


current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(project_dir, "project", "utils"))

from utils.dataloader import (
    DCASE_Dataset,
    DCASEAudioDataset,
    AUCValidationDataset,
    build_datasets as build_datasets_from_utils,
    generate_reference_embeddings,
    validate_auc,
    aggregate_windowed_embeddings,
    discover_target_test_samples,
    discover_target_test_auc_samples,
)
from model_parts import (
    DomainClusteringLoss,
    ConvBlock1d,
    DeconvBlock1d,
    StereoResidualEncoder,
    StereoResidualDecoder,
    DomainAutoencoder,
    StereoReconstructionLoss,
    StereoTemporalConsistencyLoss,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# CONFIG
# 

CONFIG = {
    "checkpoint_dir": "/ceph/project/P8_DCASE/models/2026/domain_autoencoder_slider_0_2s_self_10",
    "tsne_plot_dir":  "/ceph/project/P8_DCASE/plots/2026/slider_0_2s",

    
    #   fan | valve | slider | ToyCar | ToyCarEmu | gearbox | bearing
    "target_machine": "slider",

    # Architecture
    "latent_dim": 256,
    "base_ch":    32,
    "dropout":    0.5,
    
    # Audio
    "sample_rate":       16_000,
    "max_audio_len_sec": 0.2,
    "train_crops_per_sample": 40, #

    # Optional RAM preload for faster repeated crops from the same files.
    "preload_audio_to_ram": False,
    "preload_train_only": False,
    "preload_max_ram_gb": 20.0,

    # Training
    "num_epochs":     200,
    "learning_rate":  1e-4,
    "weight_decay":   1e-4,
    "batch_size":     64,
    "num_workers":    4,
    "val_split":      0.20,
    "val_split_seed": 42,

    # Loss weights
    "lambda_recon":    1.0,
    "lambda_compact":  0.0,
    "lambda_separate": 1.0,
    "lambda_repel":    0.8,

    # Softmax temperature for the prototypical loss (range 0.05 - 0.2)
    "proto_temperature": 0.2,

    # Minimum distance "other" samples must keep from any target centroid.
    "other_margin": 0.8,

    # Oversample target samples so clustering loss fires on most batches.
    "target_oversample": 10,

    # Unsupervised clustering (Option A: KMeans pseudo-labels)
    "unsupervised_clustering": True,
    "num_clusters": 10,
    "kmeans_refresh_every": 1,
    "kmeans_max_samples": 20_000,
    "kmeans_n_init": "auto",
    "kmeans_seed": 42,

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
    "resume_checkpoint": ""  # Path to .pth to resume from, or empty string to start fresh.
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


# Shared dataset discovery and validation helpers live in audio_encoder/utils/dataloader.py.


# 
# t-SNE VISUALISATION
# 

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
    logger.info(f"Generating t-SNE plot for epoch {epoch} ...")

    zs, domains = collect_embeddings(model, dataloader, device, max_samples)
    if zs is None:
        logger.warning("No embeddings collected ... skipping t-SNE.")
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


# 
# TRAINING
# 

def log_loss_breakdown(epoch, batch_idx, total, recon, compact, separate, repel, consist):
    logger.info(
        f"Epoch {epoch} | Batch {batch_idx:4d} | "
        f"Total: {total:.4f}  Recon: {recon:.4f}  "
        f"Compact: {compact:.4f}  Separate: {separate:.4f}  "
        f"Repel: {repel:.4f}  Consist: {consist:.4f}"
    )


def compute_kmeans_centroids(model, dataloader, device, cfg, rank: int = 0) -> Optional[torch.Tensor]:
    if not cfg.get("unsupervised_clustering", False):
        return None

    num_clusters = int(cfg.get("num_clusters", 10))
    max_samples = int(cfg.get("kmeans_max_samples", 20_000))
    kmeans_seed = int(cfg.get("kmeans_seed", 42))
    kmeans_n_init = cfg.get("kmeans_n_init", "auto")

    centroids_np = None
    if rank == 0:
        model.eval()
        embeddings = []
        seen = 0
        with torch.no_grad():
            for audio, machine_labels, _ in dataloader:
                target_mask = machine_labels == 1
                if target_mask.sum() == 0:
                    continue
                audio = audio[target_mask].to(device)
                z = model.module.get_embeddings(audio) if hasattr(model, "module") else model.get_embeddings(audio)
                embeddings.append(z.cpu().numpy())
                seen += z.shape[0]
                if seen >= max_samples:
                    break

        if not embeddings or seen < num_clusters:
            logger.warning(
                f"KMeans skipped (need >= {num_clusters} target samples, got {seen})."
            )
            centroids_np = None
        else:
            X = np.vstack(embeddings)[:max_samples]
            kmeans = KMeans(
                n_clusters=num_clusters,
                n_init=kmeans_n_init,
                random_state=kmeans_seed,
            )
            kmeans.fit(X)
            centroids_np = kmeans.cluster_centers_
            logger.info(
                f"KMeans centroids refreshed: k={num_clusters}, samples={X.shape[0]}"
            )

    if dist.is_initialized():
        shape = torch.tensor(
            centroids_np.shape if centroids_np is not None else (0, 0),
            device=device,
            dtype=torch.long,
        )
        dist.broadcast(shape, src=0)
        if shape[0] == 0 or shape[1] == 0:
            return None
        if rank == 0:
            centroids_t = torch.tensor(centroids_np, device=device, dtype=torch.float32)
        else:
            centroids_t = torch.empty((shape[0], shape[1]), device=device, dtype=torch.float32)
        dist.broadcast(centroids_t, src=0)
        return centroids_t

    return (
        torch.tensor(centroids_np, device=device, dtype=torch.float32)
        if centroids_np is not None
        else None
    )


def assign_pseudo_labels(
    z: torch.Tensor,
    machine_labels: torch.Tensor,
    centroids: Optional[torch.Tensor],
) -> torch.Tensor:
    device = z.device
    pseudo = torch.zeros(z.shape[0], dtype=torch.long, device=device)
    if centroids is None:
        return pseudo
    target_mask = machine_labels > 0
    if target_mask.sum() == 0:
        return pseudo
    dists = torch.cdist(z[target_mask], centroids)
    cluster_ids = torch.argmin(dists, dim=1) + 1
    pseudo[target_mask] = cluster_ids
    return pseudo


def run_epoch(model, dataloader, optimizer, device, epoch, cfg, rank,
              centroids: Optional[torch.Tensor], training: bool):
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
            machine_labels = machine_labels.to(device).long()
            domain_labels = domain_labels.to(device).long()

            if training:
                optimizer.zero_grad()

            out   = model(audio)
            z     = out["z"]        # (B, latent_dim)
            x_hat = out["x_hat"]    # (B, 3, T)
            x_in  = audio

            # Reconstruction loss: all samples (target + other)
            loss_recon = recon_crit(x_in, x_hat)

            # Temporal consistency loss: two crops of the same clip ÔåÆ same z.
            # Passed as (model, raw_audio) so the loss re-runs the encoder on
            # both halves internally.  The decoder is NOT run on the crops 
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
            #   loss_compact  - intra-cluster variance (monitoring)
            #   loss_separate - proto contrastive + hard centroid push
            #   loss_repel    - push "other" away from target centroids
            cluster_labels = domain_labels
            if cfg.get("unsupervised_clustering", False) and centroids is not None:
                cluster_labels = assign_pseudo_labels(z, machine_labels, centroids)

            if (cluster_labels > 0).sum() >= 2:
                loss_compact, loss_separate, loss_repel = cluster_crit(z, cluster_labels)
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

    centroids = None
    kmeans_refresh_every = int(cfg.get("kmeans_refresh_every", 1))

    for epoch in range(1, cfg["num_epochs"] + 1):
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)

        if cfg.get("unsupervised_clustering", False):
            if centroids is None or (epoch <= 3 and epoch % kmeans_refresh_every == 0):
                centroids = compute_kmeans_centroids(
                    model, train_loader, device, cfg, rank=rank
                )

        train_loss = run_epoch(model, train_loader, optimizer,
                               device, epoch, cfg, rank, centroids, training=True)
        val_loss   = run_epoch(model, val_loader,   None,
                               device, epoch, cfg, rank, centroids, training=False)
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
                    logger.info(f"   — NEW BEST AUC: {best_auc:.4f}")
                    improved = True
            else:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    state = model.module.state_dict() if hasattr(model, "module") \
                            else model.state_dict()
                    torch.save(state, best_model_path)
                    logger.info(f"   — NEW BEST Val Loss: {best_val_loss:.4f}")
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


# 
# ENTRY POINT
# 

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

    train_ds, val_ds, tsne_ds, reference_ds, auc_val_ds, domain_label_to_name = build_datasets_from_utils(cfg, rank=rank)
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
