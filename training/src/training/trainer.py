"""ArcFace fine-tuning trainer with DistributedDataParallel support.

Launch via::

    torchrun --nproc_per_node=4 train_arcface.py --config arcface_config.yaml

Or single-GPU::

    python train_arcface.py --config arcface_config.yaml

Design
------
- The trainer is **loss-agnostic**: all forward/loss logic lives in a
  :class:`~src.training.loss_step.LossStep` passed in at construction.
- Validation uses the official DCASE score (harmonic mean of AUC + pAUC)
  instead of validation loss. Early stopping maximises this score.
- DDP: each rank gets its own shard via ``DistributedSampler`` or
  ``DomainBalancedBatchSampler``.
- Optimizer: AdamW with two param groups (lower LR for backbone, higher for heads).
- Scheduler: linear warmup → cosine annealing.
- Checkpoint: rank-0 saves ``best.pt`` whenever the DCASE monitor score improves.
- WandB: rank-0 only.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.backbone.base import BaseBackbone
from src.training.loss_step import LossStep, _NoEvalHeadError
from src.data.ram_dataset import (
    RawPool, EpochBuffer,
    DomainBalancedBatchSampler,
    LiveAugDataset, LiveMultiViewDataset,
)
from src.utils import get_logger
from src.evaluation.dcase_eval import dcase_score, dcase_score_multi_machine, DCASEResult

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup followed by cosine annealing."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping (maximise)
# ─────────────────────────────────────────────────────────────────────────────

class _EarlyStopper:
    """Track a monitored metric and signal when training should stop.

    Unlike a loss-based stopper this one **maximises** the metric (higher
    DCASE score = better).

    Parameters
    ----------
    patience:
        Consecutive epochs without improvement before stopping.
        ``0`` or negative disables early stopping.
    """

    def __init__(self, patience: int) -> None:
        self.patience          = patience
        self.best_score        = -float("inf")
        self.epochs_no_improve = 0
        self.improved          = False

    def step(self, monitor: float) -> bool:
        """Update state. Returns ``True`` when stopping should be triggered."""
        if monitor > self.best_score:
            self.best_score        = monitor
            self.epochs_no_improve = 0
            self.improved          = True
        else:
            self.epochs_no_improve += 1
            self.improved           = False

        if self.patience > 0 and self.epochs_no_improve >= self.patience:
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ArcFaceTrainer:
    """Full-featured trainer with SLURM-aware auto-resume and DCASE validation."""

    def __init__(
        self,
        backbone:         BaseBackbone,
        loss_step:        LossStep,
        raw_pool:         RawPool,
        train_samples:    list,
        val_pool:         RawPool | None,
        val_samples:      list | None,
        cfg,
        rank:             int                       = 0,
        world_size:       int                       = 1,
        run_dir:          str                       = "training/results/run",
        fold:             int                       = 0,
        wandb_run                                   = None,
        save_checkpoints: bool                      = True,
        stop_event:       Optional[threading.Event] = None,
    ) -> None:
        self.rank, self.world_size = rank, world_size
        self.is_main  = (rank == 0)
        self.run_dir  = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.cfg              = cfg
        self.raw_pool         = raw_pool
        self.train_samples    = train_samples
        self.val_pool         = val_pool
        self.val_samples      = val_samples
        self.fold             = fold
        self.save_checkpoints = save_checkpoints
        self.stop_event       = stop_event
        self.loss_step        = loss_step
        
        # Detect device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ── 1. Move to device ─────────────────────────────────────────────
        backbone = backbone.to(self.device)
        loss_step.to_device(self.device)

        # ── 2. Resume logic (pre-DDP / pre-optimizer) ─────────────────────
        self.start_epoch          = 1
        self.stopper              = _EarlyStopper(cfg.training.early_stopping_patience)
        self._wandb_id_from_ckpt  = None

        ckpt_dict = None
        if self.is_main:
            #ckpt_path = self.run_dir / "last.pt"
            ckpt_path = self.run_dir / "best.pt"
            if ckpt_path.exists():
                logger.info("Found local checkpoint at %s. Resuming...", ckpt_path)
                ckpt_dict = torch.load(ckpt_path, map_location=self.device)
            else:
                ckpt_dict = self._restore_from_wandb()

        if world_size > 1:
            obj_list = [ckpt_dict]
            dist.broadcast_object_list(obj_list, src=0)
            ckpt_dict = obj_list[0]

        if ckpt_dict:
            self._apply_checkpoint(backbone, loss_step, ckpt_dict)

        # ── 3. DDP wrapping ───────────────────────────────────────────────
        if world_size > 1:
            backbone  = DDP(backbone, device_ids=[rank], find_unused_parameters=True)
            loss_step.wrap_ddp(rank, device_ids=[rank])
        self.backbone = backbone

        # ── 4. Optimizer & scheduler ──────────────────────────────────────
        tcfg     = cfg.training
        bb_params = (
            self.backbone.module.parameters()
            if isinstance(self.backbone, DDP)
            else self.backbone.parameters()
        )
        self.optimizer = torch.optim.AdamW([
            {"params": list(bb_params),              "lr": tcfg.lr_backbone},
            {"params": loss_step.head_parameters(),  "lr": tcfg.lr_head},
        ], weight_decay=tcfg.weight_decay)

        if ckpt_dict and "optimizer_state_dict" in ckpt_dict:
            self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            logger.info("Restored optimizer state.")

        self.scheduler = _build_scheduler(self.optimizer, tcfg.warmup_epochs, tcfg.epochs)
        if ckpt_dict and "scheduler_state_dict" in ckpt_dict:
            self.scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])

        # ── 5. Mixed precision ────────────────────────────────────────────
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            tcfg.mixed_precision.lower()
        )
        # GradScaler only works on CUDA
        if torch.cuda.is_available() and tcfg.mixed_precision.lower() == "fp16":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        if ckpt_dict and "scaler_state_dict" in ckpt_dict:
            self.scaler.load_state_dict(ckpt_dict["scaler_state_dict"])

        # ── 6. WandB ──────────────────────────────────────────────────────
        self._wbl = WandBLogger(
            self._init_wandb(wandb_run, resume_id=self._wandb_id_from_ckpt)
        )

        # ── 7. Training hyper-params ──────────────────────────────────────
        self.batch_size       = tcfg.batch_size
        self.grad_clip        = tcfg.gradient_clip
        self.grad_accum_steps = max(1, int(getattr(tcfg, "grad_accum_steps", 1)))
        self.min_target_frac  = float(getattr(tcfg, "target_domain_min_frac", 0.0))
        self.pin_memory       = getattr(tcfg, "pin_memory", True)

    # ─────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        has_val  = self.val_pool is not None
        # Allow disabling validation via config
        skip_validation = getattr(self.cfg.training, "skip_validation", False)
        has_val = has_val and not skip_validation
        
        if self.is_main:
            logger.info("Validation enabled: %s", has_val)
            logger.info("Backbone type: %s", type(self.backbone).__name__)
        
        train_ds = (
            LiveMultiViewDataset(self.raw_pool)
            if self.loss_step.needs_two_views
            else LiveAugDataset(self.raw_pool)
        )

        try:
            for epoch in range(self.start_epoch, self.cfg.training.epochs + 1):
                epoch_start = time.time()
                avg_loss, task_losses = self._train_epoch(train_ds, epoch)

                dcase_result: Optional[DCASEResult] = None
                val_freq = getattr(self.cfg.training, "validation_frequency", 1)
                should_validate = has_val and (epoch % val_freq == 0)
                
                if should_validate:
                    if self.is_main:
                        logger.info("Starting validation (epoch %d)...", epoch)
                    try:
                        dcase_result = self._eval_dcase()
                        if self.is_main:
                            logger.info("Validation completed. Result: %s", dcase_result)
                    except Exception as e:
                        logger.error("Validation failed: %s", e, exc_info=True)
                        dcase_result = None
                else:
                    if self.is_main and has_val:
                        logger.debug("Validation skipped (frequency=%d, epoch=%d)", val_freq, epoch)

                self.scheduler.step()

                if self.is_main:
                    self._log_epoch(
                        epoch, avg_loss, task_losses,
                        dcase_result, time.time() - epoch_start,
                    )
                    monitor = dcase_result.monitor if dcase_result is not None else -avg_loss
                    should_stop = self.stopper.step(monitor)

                    if self.stopper.improved and self.save_checkpoints:
                        self._save_checkpoint(epoch, avg_loss, dcase_result, self.stopper, is_best=True)
                        self._save_checkpoint(epoch, avg_loss, dcase_result, self.stopper, is_best=False)
                    elif self.save_checkpoints:
                        self._save_checkpoint(epoch, avg_loss, dcase_result, self.stopper, is_best=False)
                    if should_stop:
                        logger.info("Early stopping triggered at epoch %d.", epoch)
                        if self.is_main:
                            (self.run_dir / "done.flag").touch()
                        break

                if self.world_size > 1:
                    dist.barrier()
                if self.stop_event and self.stop_event.is_set():
                    break
            else:
                if self.is_main:
                    (self.run_dir / "done.flag").touch()

        finally:
            if self.is_main:
                self._wbl.finish()

    # ─────────────────────────────────────────────────────────────────────
    # Training epoch
    # ─────────────────────────────────────────────────────────────────────

    def _train_epoch(self, buf, epoch):
        self.backbone.train()
        self.loss_step.set_train()
        loader = self._build_loader(buf, epoch, shuffle=True, drop_last=True)

        total_loss, n_batches = 0.0, 0
        info_sums, info_counts = {}, {}

        self.optimizer.zero_grad()
        for step, batch in enumerate(loader):
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                loss, info = self.loss_step(self.backbone, batch, self.device)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.grad_accum_steps
            n_batches  += 1

            for k, v in info.items():
                info_sums[k]   = info_sums.get(k, 0.0) + float(v)
                info_counts[k] = (
                    1 if k in ("n_source", "n_target")
                    else info_counts.get(k, 0) + 1
                )

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1 == len(loader)):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) + self.loss_step.head_parameters(),
                        self.grad_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            #break # TEMP: remove this line to run full epoch


        avg_loss   = total_loss / max(n_batches, 1)
        task_losses = {k: info_sums[k] / max(info_counts[k], 1) for k in info_sums}
        return avg_loss, task_losses

    def _extract_embedding(self, backbone, wav: torch.Tensor) -> torch.Tensor:
        """Extract embedding from backbone, handling different architectures.
        
        Parameters
        ----------
        backbone : nn.Module
            The backbone model (BEATs, SSLAM, etc.)
        wav : torch.Tensor
            Audio waveform tensor with shape (batch, samples)
            
        Returns
        -------
        torch.Tensor
            Embedding tensor with shape (batch, embedding_dim)
        """
        backbone_type = type(backbone).__name__.lower()
        
        # Handle BEATs backbone (has extract_frames, returns time-series features)
        if 'beats' in backbone_type and hasattr(backbone, 'extract_frames'):
            frames = backbone.extract_frames(wav)
            # Geometric mean pooling: (∏ frames)^(1/n)
            emb = torch.mean(frames.clamp(min=1e-6).pow(3), dim=1).pow(1./3)
            return emb
        
        # Handle SSLAM backbone (forward returns pooled embedding directly)
        elif 'sslam' in backbone_type:
            emb = backbone(wav)  # Returns (batch, embedding_dim)
            return emb
        
        # Generic fallback for any backbone with extract_features
        elif hasattr(backbone, 'extract_features'):
            features = backbone.extract_features(wav)
            # Simple mean pooling across time dimension
            if features.dim() == 3:  # (batch, time, features)
                emb = torch.mean(features, dim=1)
            else:
                emb = features
            return emb
        
        # Fallback: forward pass as-is
        else:
            logger.warning("Unknown backbone type: %s. Using forward pass.", backbone_type)
            with torch.no_grad():
                emb = backbone(wav)
            return emb





    def _eval_dcase(self) -> DCASEResult:
        logger.info("_eval_dcase: Starting validation evaluation...")
        self.backbone.eval()
        bb = self.backbone.module if isinstance(self.backbone, DDP) else self.backbone
        
        logger.info("_eval_dcase: Backbone type=%s", type(bb).__name__)

        if self.raw_pool is None or self.val_pool is None:
            logger.warning("_eval_dcase: No validation pools available")
            if self.world_size > 1:
                dist.barrier()
            return DCASEResult()

        train_ds = self.raw_pool.build_val_buffer()
        test_ds  = self.val_pool.build_val_buffer()
        logger.info("_eval_dcase: Built val buffers - train=%d, test=%d", len(train_ds), len(test_ds))
        
        # Apply stratified sampling if configured (for faster validation on local machines)
        sample_frac = getattr(self.cfg.training, "validation_sample_fraction", 1.0)
        if sample_frac < 1.0:
            import random
            from sklearn.model_selection import train_test_split
            
            # Extract indices grouped by (machine, anomaly) to preserve stratification
            strata_dict = {}
            for idx in range(len(train_ds)):
                try:
                    _, machine, anomaly, *_ = train_ds[idx]
                    key = (str(machine), int(anomaly))
                    if key not in strata_dict:
                        strata_dict[key] = []
                    strata_dict[key].append(idx)
                except:
                    pass
            
            # Sample from each stratum proportionally
            sampled_indices = []
            for key, indices in strata_dict.items():
                n_sample = max(1, int(len(indices) * sample_frac))
                sampled = random.sample(indices, n_sample)
                sampled_indices.extend(sampled)
            
            train_ds = torch.utils.data.Subset(train_ds, sampled_indices)
            
            # Same for test
            strata_dict_test = {}
            for idx in range(len(test_ds)):
                try:
                    _, machine, anomaly, *_ = test_ds[idx]
                    key = (str(machine), int(anomaly))
                    if key not in strata_dict_test:
                        strata_dict_test[key] = []
                    strata_dict_test[key].append(idx)
                except:
                    pass
            
            sampled_indices_test = []
            for key, indices in strata_dict_test.items():
                n_sample = max(1, int(len(indices) * sample_frac))
                sampled = random.sample(indices, n_sample)
                sampled_indices_test.extend(sampled)
            
            test_ds = torch.utils.data.Subset(test_ds, sampled_indices_test)
            logger.info("_eval_dcase: Applied stratified sampling (fraction=%.2f) - train=%d, test=%d", 
                       sample_frac, len(train_ds), len(test_ds))
        
        if len(test_ds) == 0:
            logger.warning("_eval_dcase: Test dataset is empty!")
            if self.world_size > 1:
                dist.barrier()
            return DCASEResult()

        eval_batch_size = self._find_eval_batch_size(train_ds, bb)

        if self.rank == 0:
            logger.info("Extracting embeddings for DCASE evaluation... (this may take a while)")
        # ── Each rank extracts its own shard ─────────────────────────────────
        def extract_embeddings_sharded(dataset, split_name) -> tuple:
            logger.info("extract_embeddings_sharded: Starting for %s split (size=%d)", split_name, len(dataset))
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            loader = DataLoader(dataset, batch_size=eval_batch_size, sampler=sampler, num_workers=0)
            logger.info("extract_embeddings_sharded: Built loader with batch_size=%d", eval_batch_size)

            all_embs, all_machines, all_anomalies, all_domains, all_sections = [], [], [], [], []
            for batch_idx, batch in enumerate(loader):
                wav, machine, anomaly, domain, section = batch
                if wav.ndim == 3:
                    wav = wav.squeeze(1)
                wav = wav.float().to(self.device)

                with torch.no_grad():
                    emb = self._extract_embedding(bb, wav)

                if torch.isnan(emb).any():
                    emb = torch.nan_to_num(emb, nan=0.0)

                all_embs.append(emb.cpu())
                all_machines.extend(machine)
                all_anomalies.append(anomaly)
                all_domains.extend(domain)
                all_sections.extend(section)
                
                if batch_idx % 10 == 0:
                    logger.info("extract_embeddings_sharded: %s batch %d/%d", split_name, batch_idx, len(loader))
            
            # Debug: check anomaly value types
            if all_anomalies:
                first_anom = all_anomalies[0]
                logger.info("extract_embeddings_sharded: Anomaly dtype=%s, shape=%s, sample_values=%s", 
                           type(first_anom), getattr(first_anom, 'shape', 'no_shape'), 
                           first_anom if not hasattr(first_anom, 'shape') else first_anom.flatten()[:3].tolist())

            logger.info("extract_embeddings_sharded: Completed %s - extracted %d embeddings", split_name, len(all_embs))

            local_embs = torch.cat(all_embs, dim=0) if all_embs else torch.zeros((0,))
            local_anomalies = (
                torch.cat(all_anomalies, dim=0) if all_anomalies else torch.zeros((0,), dtype=torch.long)
            )

            # ── Gather from all ranks ─────────────────────────────────────────
            if self.world_size > 1:
                # Use object gather so shards can have variable lengths across ranks.
                payload = {
                    "embs": local_embs.cpu().numpy(),
                    "anomalies": local_anomalies.cpu().numpy(),
                    "machines": all_machines,
                    "domains": all_domains,
                    "sections": all_sections,
                }
                gathered = [None] * self.world_size
                dist.all_gather_object(gathered, payload)

                embs = np.concatenate([g["embs"] for g in gathered], axis=0) if gathered else np.zeros((0,))
                anomalies = (
                    np.concatenate([g["anomalies"] for g in gathered], axis=0)
                    if gathered
                    else np.zeros((0,), dtype=int)
                )
                machines = [m for g in gathered for m in g["machines"]]
                domains = [d for g in gathered for d in g["domains"]]
                sections = [s for g in gathered for s in g["sections"]]
            else:
                embs      = local_embs.numpy()
                anomalies = local_anomalies.numpy()
                machines  = all_machines
                domains   = all_domains
                sections  = all_sections

            return (
                embs,
                np.array(machines),
                anomalies.astype(int),
                np.array(domains),
                np.array(sections),
            )

        train_embs, train_machines, train_anomalies, train_domains, train_sections = extract_embeddings_sharded(train_ds, "train")
        test_embs,  test_machines,  test_anomalies,  test_domains,  test_sections  = extract_embeddings_sharded(test_ds, "test")

        # ── Only rank 0 scores ────────────────────────────────────────────────
        if not self.is_main:
            if self.world_size > 1:
                dist.barrier()
            return DCASEResult()

        logger.info("_eval_dcase: Scoring results...")

        # Group by machine
        train_machine_set = set(map(str, train_machines))
        test_machine_set = set(map(str, test_machines))
        unique_machines = sorted(train_machine_set & test_machine_set)
        
        logger.info("_eval_dcase: Machine detection - train=%s | test=%s | overlap=%s", 
                   sorted(train_machine_set), sorted(test_machine_set), unique_machines)

        if not unique_machines:
            logger.warning(
                "DCASE eval: no overlapping machine types between train/test embeddings. "
                "train=%s | test=%s",
                sorted(train_machine_set),
                sorted(test_machine_set),
            )
        per_machine_results = []

        for machine in unique_machines:
            tr_mask = train_machines == machine
            te_mask = test_machines  == machine

            labels   = test_anomalies[te_mask]
            domains  = test_domains[te_mask]
            sections = test_sections[te_mask]
            
            # Debug: log what labels we have for this machine
            unique_labels = np.unique(labels)
            logger.info("  [%s] test labels: unique=%s, count=%s, dtype=%s", 
                       machine, unique_labels.tolist(), 
                       {int(l): int(np.sum(labels == l)) for l in unique_labels},
                       labels.dtype)

            if len(np.unique(labels)) < 2:
                logger.info("  [%s] skipped — only one class present", machine)
                continue

            logger.info("  [%s] computing score with %d test samples...", machine, len(labels))
            result = dcase_score(
                embeddings_train = train_embs[tr_mask],
                embeddings_test  = test_embs[te_mask],
                labels           = labels,
                domains          = domains,
                sections         = sections,
                machine          = machine,
            )
            per_machine_results.append(result)
            logger.info("  [%s] AUC=%.4f  pAUC=%.4f", machine, result.mean_auc, result.mean_pauc)

        final = dcase_score_multi_machine(per_machine_results) if per_machine_results else DCASEResult()
        
        logger.info("_eval_dcase: Scoring complete - %d machines scored", len(per_machine_results))
        if hasattr(final, 'omega') and final.omega is not None:
            logger.info("_eval_dcase: Final DCASE scores - Ω=%.4f | AUC=%.4f | pAUC=%.4f", 
                       final.omega, final.mean_auc, final.mean_pauc)
        else:
            logger.warning("_eval_dcase: Final result is empty or invalid: %s", final)

        if self.world_size > 1:
            dist.barrier()

        return final

    # ─────────────────────────────────────────────────────────────────────
    # Batch size helper
    # ─────────────────────────────────────────────────────────────────────    
    
    def _find_eval_batch_size(self, dataset, bb, start=64, max_bs=512) -> int:
        import gc

        # Check if batch size is overridden in config
        override_bs = getattr(self.cfg.training, "eval_batch_size_override", None)
        if override_bs is not None:
            logger.info("Eval batch size: override=%d (from config)", override_bs)
            return override_bs

        sample_wav, *_ = dataset[0]
        if isinstance(sample_wav, torch.Tensor):
            # Flatten to (1, T) regardless of input shape
            sample_wav = sample_wav.reshape(1, -1)

        lo, hi, best = 1, max_bs, start
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                dummy = sample_wav.expand(mid, -1).float().to(self.device)
                with torch.no_grad():
                    _ = self._extract_embedding(bb, dummy)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                best = mid
                lo = mid + 1
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                hi = mid - 1
            finally:
                try:
                    del dummy
                except UnboundLocalError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        safe = max(1, int(best * 0.8))
        logger.info("Eval batch size: max=%d → safe=%d", best, safe)
        return safe
    
    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────

    def _apply_checkpoint(self, backbone, loss_step, ckpt):
        m_state = ckpt["model"]
        b_type  = ckpt.get("backbone_type", "").lower()
        if b_type in ["beats", "sslam"]:
            prefix  = f"_{b_type}."
            m_state = {
                (prefix + k): v
                for k, v in m_state.items()
                if not k.startswith(prefix)
            }
        backbone.load_state_dict(m_state, strict=False)
        loss_step.load_head_state(ckpt)

        self.start_epoch              = ckpt.get("epoch", 0) + 1
        self.stopper.best_score       = ckpt.get("stopper_best_score", -float("inf"))
        self.stopper.epochs_no_improve = ckpt.get("stopper_epochs_no_improve", 0)
        self._wandb_id_from_ckpt      = ckpt.get("wandb_run_id")

    def _restore_from_wandb(self) -> Optional[dict]:
        info_path = self.run_dir / "run_info.json"
        if not info_path.exists():
            return None
        try:
            with open(info_path) as f:
                info = json.load(f)
            run_id = info.get("wandb_run_id")
            wcfg   = getattr(self.cfg, "wandb", None)
            if run_id and wcfg and wcfg.enabled:
                import wandb
                logger.info("Local last.pt missing — restoring from WandB run: %s", run_id)
                r_file = wandb.restore(
                    "last.pt",
                    run_path=f"{wcfg.project}/{run_id}",
                    root=str(self.run_dir),
                )
                return torch.load(r_file.name, map_location=self.device)
        except Exception as e:
            logger.warning("WandB restoration failed: %s", e)
        return None

    def _save_checkpoint(self, epoch, train_loss, dcase_result, stopper, is_best=True):
        name = "best.pt" if is_best else "last.pt"
        raw_bb = self.backbone.module if isinstance(self.backbone, DDP) else self.backbone
        ckpt   = raw_bb.checkpoint_state()
        ckpt.update({
            "epoch":                    epoch,
            "train_loss":               train_loss,
            "dcase_omega":              dcase_result.omega    if dcase_result else None,
            "dcase_mean_auc":           dcase_result.mean_auc if dcase_result else None,
            "optimizer_state_dict":     self.optimizer.state_dict(),
            "scheduler_state_dict":     self.scheduler.state_dict(),
            "scaler_state_dict":        self.scaler.state_dict(),
            "stopper_best_score":       stopper.best_score,
            "stopper_epochs_no_improve": stopper.epochs_no_improve,
            "wandb_run_id":             self._wbl.run.id if self._wbl else None,
            **self.loss_step.checkpoint_state(),
        })
        torch.save(ckpt, self.run_dir / name)

        if self._wbl:
            with open(self.run_dir / "run_info.json", "w") as f:
                json.dump({"wandb_run_id": self._wbl.run.id, "fold": self.fold}, f)
            self._wbl.log_artifact(str(self.run_dir / name), f"model-fold{self.fold}")


    # ─────────────────────────────────────────────────────────────────────
    # DataLoader builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_loader(self, buf, epoch, shuffle, drop_last):
        if shuffle and self.min_target_frac > 0.0:
            sampler = DomainBalancedBatchSampler(
                buf._domains, self.batch_size, self.min_target_frac,
                True, self.rank, self.world_size, epoch,
            )
            return DataLoader(
                buf, batch_sampler=sampler,
                num_workers=0, pin_memory=self.pin_memory,
            )

        sampler = (
            DistributedSampler(buf, self.world_size, self.rank, shuffle=True)
            if (self.world_size > 1 and shuffle)
            else None
        )
        if sampler:
            sampler.set_epoch(epoch)
        return DataLoader(
            buf, self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

    # ─────────────────────────────────────────────────────────────────────
    # WandB init
    # ─────────────────────────────────────────────────────────────────────

    def _init_wandb(self, wandb_run, resume_id=None):
        if not self.is_main or wandb_run:
            return wandb_run
        wcfg = getattr(self.cfg, "wandb", None)
        if not (wcfg and getattr(wcfg, "enabled", False)):
            return None
        import wandb
        return wandb.init(
            project   = wcfg.project,
            name      = f"{wcfg.run_name}_fold{self.fold}",
            id        = resume_id,
            resume    = "allow",
            config    = _cfg_to_dict(self.cfg),
            dir       = str(self.run_dir),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────

    def _log_epoch(self, epoch, loss, task_losses, dcase_result, elapsed):
        omega_str = f"{dcase_result.omega:.4f}" if (dcase_result and dcase_result.omega) else "n/a"
        auc_str   = f"{dcase_result.mean_auc:.4f}" if dcase_result else "n/a"
        pauc_str  = f"{dcase_result.mean_pauc:.4f}" if dcase_result else "n/a"

        logger.info(
            "Epoch %3d | loss=%.4f | Ω=%s | AUC=%s | pAUC=%s | %.1fs",
            epoch, loss, omega_str, auc_str, pauc_str, elapsed,
        )

        log_dict: dict = {
            "epoch":       epoch,
            "train/loss":  loss,
            **{f"train/{k}": v for k, v in task_losses.items()},
        }
        if dcase_result:
            log_dict.update(dcase_result.log_dict(prefix="val"))

        self._wbl.log(log_dict)


# ─────────────────────────────────────────────────────────────────────────────
# WandBLogger
# ─────────────────────────────────────────────────────────────────────────────

class WandBLogger:
    """Thin, error-safe wrapper around a WandB run (or *None* for no-op)."""

    def __init__(self, wandb_run=None):
        self.run = wandb_run

    def __bool__(self) -> bool:
        return self.run is not None

    def log(self, data: dict, **kw) -> None:
        if self.run:
            try:
                self.run.log(data, **kw)
            except Exception as exc:
                logger.debug("WandB log failed: %s", exc)

    def summary_set(self, key: str, value) -> None:
        if self.run:
            try:
                self.run.summary[key] = value
            except Exception as exc:
                logger.debug("WandB summary update failed: %s", exc)

    def log_artifact(self, path: str, name: str, metadata: dict | None = None) -> None:
        if self.run:
            try:
                import wandb as _w
                art = _w.Artifact(name=name, type="model", metadata=metadata or {})
                art.add_file(str(path), name=Path(path).name)
                self.run.log_artifact(art)
            except Exception as exc:
                logger.warning("WandB artifact upload failed: %s", exc)

    def finish(self, exit_code: int = 0) -> None:
        if self.run:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg_to_dict(cfg) -> dict:
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    if isinstance(cfg, dict):
        return cfg
    return vars(cfg) if hasattr(cfg, "__dict__") else {}