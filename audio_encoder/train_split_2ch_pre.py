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
sys.path.append(os.path.join(current_dir, "utils"))
from dataloader import (
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "checkpoint_dir": "/ceph/project/P8_DCASE/models/2026/domain_autoencoder_bearingEmu_v2",
    "tsne_plot_dir":  "/ceph/project/P8_DCASE/plots/2026/bearingEmu_tsne",

    # Target machine — must match one of DCASE_Dataset.machines exactly:
    #   fan | valve | slider | ToyCar | ToyCarEmu | gearbox | bearing
    "target_machine": "bearing",

    # Architecture
    "latent_dim": 256,
    "base_ch":    32,
    "dropout":    0.5,
    
    # Audio
    "sample_rate":       16_000,
    "max_audio_len_sec": 1.0,
    "train_crops_per_sample": 8, #

    # Optional RAM preload for faster repeated crops from the same files.
    "preload_audio_to_ram": False,
    "preload_train_only": False,
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
    "resume_checkpoint": ""  # Path to .pth to resume from, or empty string to start fresh.
}

local_env = False
if local_env:
    CONFIG["checkpoint_dir"] = r"C:\Users\emilr\Documents\GitHub\AAU_P8\project\models\domain_autoencoder"
    CONFIG["tsne_plot_dir"]   = r"C:\Users\emilr\Documents\GitHub\AAU_P8\project\plots"

# Architecture and loss classes moved to audio_encoder/model_parts.py


# Dataset discovery and validation helpers moved to utils/dataloader.py


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