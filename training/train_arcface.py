#!/usr/bin/env python
"""ArcFace fine-tuning entry point.

Single-GPU::

    python train_arcface.py --config arcface_config.yaml

Multi-GPU (4 GPUs)::

    torchrun --nproc_per_node=4 train_arcface.py --config arcface_config.yaml

Override any config key on the CLI::

    torchrun --nproc_per_node=4 train_arcface.py \\
        --config arcface_config.yaml \\
        --override training.epochs=100 backbone.beats.pool=mean+std
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import datetime
from dataclasses import replace
from pathlib import Path

# ── Make `src` importable regardless of cwd ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.distributed as dist

from src.utils import setup_logging, seed_everything, get_logger
from src.backbone import get_backbone
from src.data.dataset import scan_samples, scan_noise_pool, DataRoot
from src.data.label_encoder import LabelEncoder
from src.data.ram_dataset import RawPool
from src.training.trainer import ArcFaceTrainer
from src.training.loss_step import StandardLossStep

logger = get_logger(__name__)

# Global shutdown flag — set by SIGTERM handler on every DDP rank.
# The training loop checks this after dist.barrier() at the end of each epoch
# so all ranks exit the loop together without deadlocking in a future collective.
_stop_event = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading (OmegaConf with fallback to PyYAML + SimpleNamespace)
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str, overrides: list[str]):
    """Load YAML config; apply CLI overrides; return config object."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        for override in overrides:
            key, _, val = override.partition("=")
            OmegaConf.update(cfg, key.strip(), _coerce(val.strip()), merge=True)
        return cfg
    except ImportError:
        pass

    # Fallback: PyYAML + recursive SimpleNamespace
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    for override in overrides:
        key, _, val = override.partition("=")
        _set_nested(raw, key.strip().split("."), _coerce(val.strip()))

    return _dict_to_ns(raw)


def _coerce(val: str):
    """Try to cast a string CLI value to int / float / bool / None."""
    if val.lower() == "null":
        return None
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _set_nested(d: dict, keys: list, val):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def _dict_to_ns(d):
    import types
    if not isinstance(d, dict):
        return d
    ns = types.SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v))
    return ns


# ─────────────────────────────────────────────────────────────────────────────
# Device and DDP initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_device(local_rank: int, world_size: int) -> str:
    """Detect and set the appropriate device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        if world_size > 1:
            torch.cuda.set_device(local_rank)
        device = "cuda"
        logger.info("✓ CUDA detected and available")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("✓ MPS (Metal Performance Shaders) detected and available")
    else:
        device = "cpu"
        logger.info("⚠ Using CPU (no CUDA/MPS available)")
    return device


def _init_ddp():
    """Initialise process group if launched via torchrun, else return rank=0."""
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        logger.info("DDP initialised: rank=%d / world_size=%d", rank, world_size)

    return rank, local_rank, world_size


# ─────────────────────────────────────────────────────────────────────────────
# Loss-step factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_loss_step(cfg, mode: str, out_dim: int, label_encoder, num_classes: int):
    """Instantiate the correct LossStep for *mode*.

    Parameters
    ----------
    cfg:
        Full training config.
    mode:
        Currently only ``"standard"`` is supported.
    out_dim:
        Backbone embedding dimensionality.
    label_encoder:
        Fitted ``LabelEncoder``.
    num_classes:
        Number of ArcFace classes.

    Returns
    -------
    LossStep
    """
    from src.training.config import ArcFaceConfig
    from src.loss.arcface import ArcFaceHead

    # Standard supervised ArcFace
    head = ArcFaceHead(
        in_dim=out_dim, num_classes=num_classes,
        **ArcFaceConfig.from_cfg(cfg).head_kwargs(),
    )
    return StandardLossStep(head)


# ─────────────────────────────────────────────────────────────────────────────
# main() helpers  (each does one thing; main() orchestrates them)
# ─────────────────────────────────────────────────────────────────────────────

def _init_sweep(args, rank: int, is_main: bool, world_size: int):
    """Contact the WandB sweep controller (rank-0 only) and broadcast to all ranks.

    Updates ``args.config`` and ``args.override`` in-place with sweep-sampled
    hyperparams so every rank trains with the same configuration.
    Returns the initialised WandB run, or ``None`` for non-sweep runs.
    """
    sweep_wandb_run = None
    if args.sweep and is_main:
        sweep_id = os.environ.get("WANDB_SWEEP_ID", "")
        if not sweep_id:
            logger.error("--sweep requires WANDB_SWEEP_ID env var to be set.")
            sys.exit(1)
        try:
            import wandb as _wandb
            sweep_wandb_run = _wandb.init(
                project="aau_p8_arcface",
                entity="chrrhod3-aalborg-universitet",
            )
            sweep_training_mode = getattr(sweep_wandb_run.config, "training_mode", None)
            print(f"[sweep debug] raw config: {dict(sweep_wandb_run.config)}", flush=True)
            if sweep_training_mode is not None:
                _CONFIG_MAP = {
                    "standard":     "sslam_config.yaml",
                    "hierarchical": "sslam_hierarchical_config.yaml",
                }
                args.config = _CONFIG_MAP.get(sweep_training_mode, args.config)

            def _flatten_config(d, prefix=""):
                items = []
                for k, v in d.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.extend(_flatten_config(v, full_key))
                    else:
                        items.append(f"{full_key}={v}")
                return items

            sweep_overrides = [
                entry for entry in _flatten_config(dict(sweep_wandb_run.config))
                if not entry.startswith("training_mode=")
            ]
            args.override = list(args.override or []) + sweep_overrides
            logger.info("Sweep trial | config=%s | overrides=%s", args.config, sweep_overrides)
        except Exception as e:
            logger.error("WandB sweep init failed: %s", e)
            sys.exit(1)

    if args.sweep and world_size > 1:
        obj = [args.config, args.override]
        dist.broadcast_object_list(obj, src=0)
        args.config, args.override = obj[0], obj[1]

    return sweep_wandb_run


def _parse_extra_roots(value):
    """Coerce ``extra_data_roots`` config value to a list of ``DataRoot`` objects."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    else:
        value = list(value)

    def _to_dr(v):
        if isinstance(v, DataRoot):
            return v
        if isinstance(v, dict):
            return DataRoot(**{k: val for k, val in v.items() if k in ("path", "load_csv", "name")})
        return DataRoot(path=str(v), load_csv=False)

    return [_to_dr(e) for e in value]


def _metadata_to_sample(meta: dict):
    """Convert a MachineAudioDataset metadata dict to a Sample object.

    Expected dict keys: path, machine, section, domain, attributes, split.
    The ``attributes`` value ``"none"`` (MachineAudioDataset's sentinel) is
    normalised to ``""`` so LabelEncoder maps it to ``"noAttribute"`` instead
    of creating a spurious ``"none"`` class.
    """
    from src.data.dataset import Sample
    attr_str = meta.get("attributes", "")
    if attr_str == "none":
        attr_str = ""
    anomaly = 1 if meta.get("type", "normal") == "anomaly" else 0 
    return Sample(
        path         = Path(meta["path"]),
        machine_type = meta["machine"],
        section      = int(meta["section"]),
        domain       = meta["domain"],
        split        = meta.get("split", "train"),
        anomaly      = anomaly,
        attr_str     = attr_str,
        attributes   = {},
    )


def _load_machine_audio_dataset(module_path: str):
    """Dynamically import a data_all.py-style module and return train/test Sample lists.

    The module must contain exactly one class whose name ends in ``"Dataset"``
    that exposes a ``.samples`` list of metadata dicts with at minimum the
    keys: ``path``, ``machine``, ``section``, ``domain``, ``attributes``, ``split``.

    The dataset is instantiated with ``preload=False`` because RawPool handles
    all I/O itself.

    Returns
    -------
    (train_samples, test_samples)
    """
    import importlib.util
    p    = Path(module_path)
    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    dataset_cls = next(
        (getattr(mod, name) for name in dir(mod)
         if isinstance(getattr(mod, name), type)
         and name.endswith("Dataset")
         and getattr(mod, name).__module__ == mod.__name__),
        None,
    )
    if dataset_cls is None:
        raise RuntimeError(f"No class ending in 'Dataset' found in {p}")

    dataset     = dataset_cls(preload=False)
    all_samples = [_metadata_to_sample(m) for m in dataset.samples]

    train_samples = [s for s in all_samples if s.split == "train"]
    test_samples  = [s for s in all_samples if s.split == "test"]

    logger.info("_load_machine_audio_dataset | '%s' → %d train / %d test samples",
                p.name, len(train_samples), len(test_samples))
    return train_samples, test_samples


def _scan_data(cfg, mode: str):
    """Scan the dataset directory and return all sample lists and noise paths.

    If ``data.root`` contains a ``data_all.py`` file (MachineAudioDataset-style
    module), that class is used as the primary data source.  Otherwise falls
    back to the original ``scan_samples`` directory scanner.

    Additional datasets of the same style are loaded via the ``data.extra_datasets``
    config key — a list of paths to ``data_all.py``-style module files.

    Returns
    -------
    (root_samples, extra_samples, all_samples, noise_paths, test_samples)
    """
    dcfg = cfg.data

    data_all_py = Path(dcfg.root) / "data_all.py"
    if data_all_py.exists():
        # ── MachineAudioDataset path ──────────────────────────────────────
        root_samples, test_samples = _load_machine_audio_dataset(str(data_all_py))

        # Noise pool: data_2025 has the same supplemental structure as old data/
        noise_root  = Path(dcfg.root) / "data_2025"
        noise_paths = scan_noise_pool(str(noise_root)) if noise_root.exists() else []
        if not noise_paths:
            logger.warning("_scan_data | no noise clips found in '%s'", noise_root)
    else:
        # ── Legacy scan_samples fallback ──────────────────────────────────
        is_hierarchical = mode == "hierarchical"
        machine_types   = getattr(dcfg, "machine_types", None)
        if isinstance(machine_types, (list, tuple)) and len(machine_types) == 0:
            machine_types = None
        splits      = getattr(dcfg, "splits", ["train", "supplemental"])
        if not isinstance(splits, (list, tuple)):
            splits = list(splits)
        extra_roots = _parse_extra_roots(getattr(dcfg, "extra_data_roots", None))

        root_samples = scan_samples(
            data_root         = dcfg.root,
            machine_types     = machine_types,
            splits            = splits,
            target_sr         = getattr(dcfg, "sample_rate", 16_000),
            clip_duration     = getattr(dcfg, "clip_duration", 10.0),
            load_hierarchical = is_hierarchical,
            extra_data_roots  = None,
        )
        noise_paths = scan_noise_pool(
            data_root        = dcfg.root,
            machine_types    = machine_types,
            extra_data_roots = extra_roots or None,
        )
        
        # ── Scan test split separately for validation ─────────────────────
        test_samples = scan_samples(
            data_root         = dcfg.root,
            machine_types     = machine_types,
            splits            = ["test"],
            target_sr         = getattr(dcfg, "sample_rate", 16_000),
            clip_duration     = getattr(dcfg, "clip_duration", 10.0),
            load_hierarchical = is_hierarchical,
            include_anomaly   = True,  # Include both normal AND anomalous samples for validation
            extra_data_roots  = None,
        )
        if test_samples:
            logger.info("_scan_data | loaded %d test samples for validation", len(test_samples))

    # ── Extra datasets (data_all.py-style modules) ────────────────────────
    extra_train:  list = []
    extra_test:   list = []
    extra_dataset_paths = getattr(dcfg, "extra_datasets", None)
    if extra_dataset_paths:
        if isinstance(extra_dataset_paths, str):
            extra_dataset_paths = [extra_dataset_paths]
        for ep in extra_dataset_paths:
            et, ev = _load_machine_audio_dataset(str(ep))
            extra_train.extend(et)
            extra_test.extend(ev)

    root_samples = root_samples + extra_test

    logger.info(
        "_scan_data | root_train=%d  extra_train=%d  test=%d  noise=%d",
        len(root_samples), len(extra_train), len(test_samples), len(noise_paths),
    )
    return root_samples, extra_train, root_samples + extra_train, noise_paths, test_samples


def _fit_encoder(cfg, mode: str, all_samples):
    """Fit and return the label encoder for standard ArcFace.

    Returns
    -------
    (label_encoder, num_classes)
    """
    split_channels = bool(getattr(cfg.data, "separate_channels_as_samples", False))

    samples_for_fit = all_samples
    if split_channels:
        expanded = []
        for s in all_samples:
            expanded.append(replace(s, channel=0))
            expanded.append(replace(s, channel=1))
        samples_for_fit = expanded

    enc = LabelEncoder().fit(samples_for_fit)
    return enc, enc.num_classes


def _run_training(
    root_samples:    list,
    extra_samples:   list,
    test_samples:    list,
    noise_paths:     list,
    mode:            str,
    label_encoder,
    num_classes:     int,
    cfg,
    rank:            int,
    world_size:      int,
    run_dir:         Path,
    sweep_wandb_run,
    is_main:         bool,
    clip_samples:    int,
    device:          str,
    fresh:           bool = False,
):
    """Orchestrates data setup, trainer instantiation, and training.

    Uses all train samples for training and test samples as the validation set.
    """
    train_samples = root_samples + extra_samples

    # ── Skip / Fresh Logic ───────────────────────────────────────────────
    done_flag   = run_dir / "done.flag"
    should_skip = False

    if is_main:
        if fresh:
            logger.info("Fresh start requested — clearing done.flag in %s", run_dir)
            if done_flag.exists():
                done_flag.unlink()
        elif done_flag.exists():
            pass
            logger.info("Run already marked as DONE. Skipping.")
            should_skip = True  # <-- REMOVED TO ALLOW SWEEP TRIALS TO RUN. RE-ADD FOR NON-SWEEP MODE.

    # Synchronise skip decision across all DDP ranks
    if world_size > 1:
        skip_tensor = torch.tensor([1 if should_skip else 0], device=rank)
        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
        if skip_tensor.item() == 1:
            return

    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialization ───────────────────────────────────────────────────
    backbone  = get_backbone(cfg)
    loss_step = _build_loss_step(cfg, mode, backbone.output_dim, label_encoder, num_classes)
    
    # Move backbone to device
    backbone = backbone.to(device)

    dcfg     = cfg.data
    raw_pool = RawPool(
        samples       = train_samples,
        noise_paths   = noise_paths,
        label_encoder = label_encoder,
        sample_rate   = getattr(dcfg, "sample_rate", 16_000),
        clip_samples  = clip_samples,
        aug_cfg       = cfg.augmentation,
        separate_channels_as_samples=bool(getattr(dcfg, "separate_channels_as_samples", False)),
    )

    val_pool = None
    if test_samples:
        val_pool = RawPool(
            samples       = test_samples,
            noise_paths   = [],
            label_encoder = label_encoder,
            sample_rate   = getattr(dcfg, "sample_rate", 16_000),
            clip_samples  = clip_samples,
            aug_cfg       = cfg.augmentation,
            separate_channels_as_samples=False,
        )
        logger.info("Validation pool: %d test samples", len(test_samples))
        # ── Debug: inspect first validation sample ──────────────────────────────
        if test_samples and len(test_samples) > 0:
            s = test_samples[0]

            logger.info("─── DEBUG: First val sample ───")
            logger.info("path: %s", s.path)
            logger.info("machine: %s", getattr(s, "machine_type", "N/A"))
            logger.info("label (anomaly): %s", getattr(s, "anomaly", "N/A"))
            logger.info("domain: %s", getattr(s, "domain", "N/A"))
            logger.info("section: %s", getattr(s, "section", "N/A"))

            # Try accessing waveform from pool (index-based)
            try:
                wav = val_pool.waveforms[0]  # assumes same ordering
                logger.info("waveform shape: %s", tuple(wav.shape))
                logger.info("waveform dtype: %s", wav.dtype)
                logger.info("waveform min/max: %.4f / %.4f", wav.min().item(), wav.max().item())
            except Exception as e:
                logger.info("Failed to access waveform from val_pool: %s", e)

            # Try dataset access (this is what eval uses!)
            try:
                val_ds = val_pool.build_val_buffer()
                x, y, z, _, _= val_ds[0] #(wav, machine_type, anomaly, domain, section)
                logger.info("dataset sample shape: %s", tuple(x.shape))
                logger.info("dataset type: %s", y)
                logger.info("dataset label (anomaly): %s", z)
            except Exception as e:
                logger.info("Failed to access dataset sample: %s", e)

            logger.info("────────────────────────────────")
        else:
            logger.warning("DEBUG: val_samples is empty!")


    #print the first sample in the val_pool to debug
    
    # ── Training ─────────────────────────────────────────────────────────
    trainer = ArcFaceTrainer(
        backbone         = backbone,
        loss_step        = loss_step,
        raw_pool         = raw_pool,
        train_samples      = train_samples,
        val_pool         = val_pool,
        val_samples       = test_samples,
        cfg              = cfg,
        rank             = rank,
        world_size       = world_size,
        run_dir          = str(run_dir),
        fold             = 0,
        wandb_run        = sweep_wandb_run,
        save_checkpoints = sweep_wandb_run is None,
        stop_event       = _stop_event,
    )

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="ArcFace fine-tuning of audio SSL models")
    parser.add_argument("--config",   default="arcface_config.yaml", help="Path to YAML config")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    parser.add_argument("--sweep",    action="store_true", help="WandB sweep mode")
    parser.add_argument("--fresh",    action="store_true", help="Ignore existing done.flag")
    args = parser.parse_args()

    # ── 1. Distributed Environment ───────────────────────────────────────
    rank, local_rank, world_size = _init_ddp()
    is_main = rank == 0
    
    # ── 1b. Device Selection (MPS, CUDA, or CPU) ──────────────────────────
    device = _get_device(local_rank, world_size)

    # Handle SLURM/user termination gracefully
    def _sigterm_handler(signum, frame):
        logger.info("[rank %d] SIGTERM received — shutting down gracefully...", rank)
        _stop_event.set()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── 2. Configuration ─────────────────────────────────────────────────
    sweep_wandb_run = _init_sweep(args, rank, is_main, world_size)

    cfg          = _load_config(args.config, args.override or [])
    mode         = getattr(cfg.training, "mode", "standard")
    clip_samples = int(
        getattr(cfg.data, "sample_rate", 16_000) * getattr(cfg.data, "clip_duration", 10.0)
    )

    # Store device in config for trainer to access
    if not hasattr(cfg, "device"):
        # For OmegaConf
        try:
            from omegaconf import OmegaConf
            cfg.device = device
        except:
            # For SimpleNamespace
            cfg.device = device

    setup_logging(
        level=getattr(cfg.output, "log_level", "INFO"),
        rank=rank, world_size=world_size,
    )
    seed_everything(getattr(cfg, "seed", 42) + rank)

    if is_main:
        logger.info("Run Mode: %s | Config: %s | Device: %s", mode, args.config, device)

    # ── 3. Data & Label Encoding ─────────────────────────────────────────
    root_samples, extra_samples, all_samples, noise_paths, test_samples = _scan_data(cfg, mode)
    if not root_samples:
        logger.error("No training samples found in %s — exiting.", cfg.data.root)
        sys.exit(1)

    label_encoder, num_classes = _fit_encoder(cfg, mode, all_samples)

    # ── 4. Run Directory ─────────────────────────────────────────────────
    job_id = os.environ.get("SLURM_JOB_ID")
    if getattr(getattr(cfg, "wandb", None), "run_name", None):
        run_name = cfg.wandb.run_name
    elif job_id:
        run_name = f"job_{job_id}"
    else:
        run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = Path(getattr(cfg.output, "save_dir", "training/results")) / run_name

    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        label_encoder.save(str(run_dir / "label_encoder.json"))
        logger.info("Logging results to: %s", run_dir)

    # ── 5. Train ─────────────────────────────────────────────────────────
    _run_training(
        root_samples    = root_samples,
        extra_samples   = extra_samples,
        test_samples    = test_samples,
        noise_paths     = noise_paths,
        mode            = mode,
        label_encoder   = label_encoder,
        num_classes     = num_classes,
        cfg             = cfg,
        rank            = rank,
        world_size      = world_size,
        run_dir         = run_dir,
        sweep_wandb_run = sweep_wandb_run,
        is_main         = is_main,
        clip_samples    = clip_samples,
        device          = device,
        fresh           = args.fresh,
    )

    # ── 6. Cleanup ───────────────────────────────────────────────────────
    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()