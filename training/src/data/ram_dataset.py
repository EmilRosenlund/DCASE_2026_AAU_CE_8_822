"""In-RAM dataset with async double-buffer augmentation.

Design
------
1. **RawPool** — loads every raw waveform into RAM at startup once.
   Noise waveforms are also loaded here for fast mixing.

2. **EpochBuffer** — a ``torch.utils.data.Dataset`` backed by a plain
   Python ``list`` of ``(waveform_tensor, label_int)`` tuples that have
   already been augmented.  ``__getitem__`` is a pure list index — zero I/O,
   zero augmentation during training.

3. **AsyncAugmentor** — a ``threading.Thread`` that iterates the ``RawPool``,
   applies ``WaveformAugmentor`` to each clip, and writes the result into the
   *next* ``EpochBuffer`` while the current epoch trains.  A
   ``threading.Event`` signals completion.

Trainer usage::

    pool    = RawPool(samples, noise_paths, label_encoder, cfg)
    current = pool.build_buffer()           # synchronous first fill

    augmentor = AsyncAugmentor(pool, next_buffer_slot)
    augmentor.start()                       # fills next buffer in bg

    for epoch in range(epochs):
        loader = DataLoader(current, ...)
        train_one_epoch(loader)

        augmentor.wait()                    # block until next buffer ready
        current, next_buffer_slot = next_buffer_slot, current
        augmentor = AsyncAugmentor(pool, next_buffer_slot)
        augmentor.start()
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple, Union
import os

import torch
from torch.utils.data import Dataset

from src.data.dataset import Sample, load_wav
from src.data.label_encoder import LabelEncoder
from src.data.augment import WaveformAugmentor
from src.utils import get_logger

logger = get_logger(__name__)

# Standard encoder only
_AnyEncoder = LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# RawPool — all waveforms loaded into CPU RAM
# ─────────────────────────────────────────────────────────────────────────────

class RawPool:
    """Load every training clip and all noise clips into CPU RAM.

    Parameters
    ----------
    samples:
        Training ``Sample`` objects (train + supplemental clean).
    noise_paths:
        Paths to supplemental noise-only WAV files.
    label_encoder:
        Fitted ``LabelEncoder`` used to map each sample → integer label.
    sample_rate:
        Common sample rate (Hz).
    clip_samples:
        Clip length in samples (sr × duration).
    aug_cfg:
        Augmentation config block passed to ``WaveformAugmentor``.
    """

    def __init__(
        self,
        samples: List[Sample],
        noise_paths: List[Path],
        label_encoder,          # LabelEncoder
        sample_rate: int,
        clip_samples: int,
        aug_cfg,
        separate_channels_as_samples: bool = False,
    ) -> None:
        self.sample_rate  = sample_rate
        self.clip_samples = clip_samples
        self.aug_cfg      = aug_cfg
        self.separate_channels_as_samples = separate_channels_as_samples

        logger.info(
            "RawPool — loading %d clips into RAM%s …",
            len(samples),
            " (channel-split mode)" if self.separate_channels_as_samples else "",
        )
        # Parallel I/O: WAV loading is network-filesystem bound; this worker count
        # usually saturates the connection without overwhelming the NFS server.
        self.waveforms:    List[torch.Tensor] = []
        self.labels:       List[int]          = []
        self.domain_flags: List[int]          = []
        self.anomaly_flags: List[int]         = []
        self.machine_types: List[str]         = []
        self.domain_strs:   List[str]         = []  # "source" / "target"
        self.sections:      List[str]         = []

        def _load_one(idx: int, s):
            records = []
            channels = [0, 1] if self.separate_channels_as_samples else [getattr(s, "channel", None)]
            for j, ch in enumerate(channels):
                s_eff = replace(s, channel=ch) if ch is not None else s
                wav = load_wav(s.path, sample_rate, clip_samples, channel=ch).half()
                try:
                    lbl = label_encoder.encode(s_eff)
                except KeyError:
                    lbl = -1
                dom_int = 1 if getattr(s_eff, "domain", "") == "target" else 0
                dom_str = getattr(s_eff, "domain", "source")
                anomaly = getattr(s_eff, "anomaly", 0)
                machine = getattr(s_eff, "machine_type", "unknown")
                section = str(getattr(s_eff, "section", "00"))
                records.append((idx * max(1, len(channels)) + j, wav, lbl, dom_int, dom_str, anomaly, machine, section))
            return records
        
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_one, i, s) for i, s in enumerate(samples)]
            loaded = []
            for fut in as_completed(futures):
                loaded.extend(fut.result())

        loaded.sort(key=lambda x: x[0])
        for _, wav, lbl, dom_int, dom_str, anomaly, machine, section in loaded:
            self.waveforms.append(wav)
            self.labels.append(lbl)
            self.domain_flags.append(dom_int)
            self.domain_strs.append(dom_str)
            self.anomaly_flags.append(anomaly)
            self.machine_types.append(machine)
            self.sections.append(section)

        logger.info("RawPool — training clips loaded (%.1f GB) | target=%d source=%d",
                    sum(w.nbytes for w in self.waveforms) / 1e9,
                    sum(self.domain_flags), len(self.domain_flags) - sum(self.domain_flags))

        logger.info("RawPool — loading %d noise clips into RAM …", len(noise_paths))
        self.noise_waveforms: List[torch.Tensor] = [None] * len(noise_paths)  # type: ignore[list-item]

        def _load_noise(idx: int, p):
            return idx, load_wav(p, sample_rate, clip_samples).half()  # float16 halves RAM

        
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_load_noise, i, p) for i, p in enumerate(noise_paths)]
            for fut in as_completed(futures):
                i, wav = fut.result()
                self.noise_waveforms[i] = wav

        logger.info("RawPool — noise clips loaded (%.1f GB)",
                    sum(w.nbytes for w in self.noise_waveforms) / 1e9)

        self._augmentor = WaveformAugmentor(aug_cfg, self.noise_waveforms)

    # ──────────────────────────────────────────────────────────────────────

    def build_buffer(self) -> "EpochBuffer":
        """Synchronously build a fresh augmented epoch buffer."""
        data: List[Tuple[torch.Tensor, int]] = [None] * len(self.waveforms)  # type: ignore[list-item]

        def _aug(idx: int, wav: torch.Tensor, label: int, augmentor):
            return idx, (augmentor(wav), label)

        augmentor = WaveformAugmentor(self.aug_cfg, self.noise_waveforms)
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_aug, i, wav, lbl, augmentor)
                for i, (wav, lbl) in enumerate(zip(self.waveforms, self.labels))
            ]
            for fut in as_completed(futures):
                i, item = fut.result()
                data[i] = item
        return EpochBuffer(data, list(self.domain_flags))

    def build_val_buffer(self) -> "ValBuffer":
        data = [
            (wav, machine, anomaly, domain, section)
            for wav, machine, anomaly, domain, section in zip(
                self.waveforms,
                self.machine_types,
                self.anomaly_flags,
                self.domain_strs,
                self.sections,
            )
        ]
        return ValBuffer(data)

    def fill_buffer(self, buf: "EpochBuffer") -> None:
        """Fill an existing ``EpochBuffer`` in-place with new augmented copies."""
        augmentor = WaveformAugmentor(self.aug_cfg, self.noise_waveforms)

        def _aug(idx: int, wav: torch.Tensor, label: int):
            return idx, (augmentor(wav), label)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(_aug, i, wav, lbl)
                for i, (wav, lbl) in enumerate(zip(self.waveforms, self.labels))
            ]
            for fut in as_completed(futures):
                i, item = fut.result()
                buf._data[i] = item
        # domain_flags are static — no need to update them

    def build_multiview_buffer(self) -> "MultiViewEpochBuffer":
        """Synchronously build a multi-view epoch buffer for self-labeling.

        Each sample is augmented **twice independently** using different random
        seeds so the two views are genuinely different waveform transformations
        of the same underlying clip.  The backbone is then asked to produce
        consistent prototype assignments despite the acoustic differences.
        """
        data: List[Tuple[torch.Tensor, torch.Tensor, int]] = [None] * len(self.waveforms)  # type: ignore[list-item]

        def _aug_two(idx: int, wav: torch.Tensor, label: int):
            # Create two fresh augmentors per call — each carries its own
            # WaveformAugmentor instance so random state is independent.
            noise = self.noise_waveforms
            aug1 = WaveformAugmentor(self.aug_cfg, noise)
            aug2 = WaveformAugmentor(self.aug_cfg, noise)
            return idx, (aug1(wav), aug2(wav), label)

        #Get the allocated CPUs from slurm and set
        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(_aug_two, i, wav, lbl)
                for i, (wav, lbl) in enumerate(zip(self.waveforms, self.labels))
            ]
            for fut in as_completed(futures):
                i, item = fut.result()
                data[i] = item
        return MultiViewEpochBuffer(data, list(self.domain_flags))

    def fill_multiview_buffer(self, buf: "MultiViewEpochBuffer") -> None:
        """Fill an existing ``MultiViewEpochBuffer`` in-place with new two-view augmented copies."""

        def _aug_two(idx: int, wav: torch.Tensor, label: int):
            aug1 = WaveformAugmentor(self.aug_cfg, self.noise_waveforms)
            aug2 = WaveformAugmentor(self.aug_cfg, self.noise_waveforms)
            return idx, (aug1(wav.clone()), aug2(wav.clone()), label)

        max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_aug_two, i, wav, lbl)
                for i, (wav, lbl) in enumerate(zip(self.waveforms, self.labels))
            ]
            for fut in as_completed(futures):
                i, item = fut.result()
                buf._data[i] = item
        # domain_flags are static — no need to update them


# ─────────────────────────────────────────────────────────────────────────────
# EpochBuffer — pre-augmented, pure-tensor Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EpochBuffer(Dataset):
    """A pre-augmented epoch snapshot stored entirely in CPU RAM.

    ``__getitem__`` is a pure list index — no I/O, no computation.
    """

    def __init__(
        self,
        data: List[Tuple[torch.Tensor, int]],
        domain_flags: Optional[List[int]] = None,
    ) -> None:
        self._data    = data
        # 1 = target domain, 0 = source/other — used by DomainBalancedBatchSampler
        self._domains = domain_flags if domain_flags is not None else [0] * len(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._data[idx]

    @property
    def num_samples(self) -> int:
        return len(self._data)


# ─────────────────────────────────────────────────────────────────────────────
# MultiViewEpochBuffer — two-view pre-augmented Dataset for self-labeling
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewEpochBuffer(Dataset):
    """Pre-augmented two-view epoch snapshot for self-labeling training.

    Each item is a ``(view1, view2, label, domain)`` quad where *view1* and
    *view2* are independently augmented copies of the same raw waveform.
    *domain* is ``0`` for source-domain clips and ``1`` for target-domain clips.

    The domain flag enables the supervised ArcFace branch to train on
    source-only samples while the Sinkhorn OT branch sees all samples —
    this forces target clips to align with source-discovered prototypes
    without corrupting the ArcFace class clusters with unlabelled data.

    ``__getitem__`` is a pure list index — no I/O, no computation.
    """

    def __init__(
        self,
        data: List[Tuple[torch.Tensor, torch.Tensor, int]],
        domain_flags: Optional[List[int]] = None,
    ) -> None:
        self._data    = data
        self._domains = domain_flags if domain_flags is not None else [0] * len(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        view1, view2, label = self._data[idx]
        return view1, view2, label, self._domains[idx]

    @property
    def num_samples(self) -> int:
        return len(self._data)

# ─────────────────────────────────────────────────────────────────────────────
# ValBuffer — non-augmented Dataset for validation
# ─────────────────────────────────────────────────────────────────────────────
class ValBuffer(Dataset):
    """Validation-only dataset returning (waveform, machine_type, anomaly, domain, section)."""

    def __init__(self, data: List[Tuple[torch.Tensor, str, int, str, str]]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, str, str]:
        return self._data[idx]  # (wav, machine_type, anomaly, domain, section)

    @property
    def num_samples(self) -> int:
        return len(self._data)

# ─────────────────────────────────────────────────────────────────────────────
# AsyncAugmentor — background thread that fills the next epoch buffer
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAugmentor:
    """Background thread: fill *target_buffer* with freshly augmented data.

    Usage::

        aug = AsyncAugmentor(pool, target_buffer)
        aug.start()        # non-blocking
        # ... train on current_buffer ...
        aug.wait()         # blocks until target_buffer is ready

    Parameters
    ----------
    pool:
        The ``RawPool`` holding the raw waveforms.
    target_buffer:
        An existing ``EpochBuffer`` whose ``_data`` list will be overwritten
        in-place.  It must already have the same length as ``pool.waveforms``.
    """

    def __init__(self, pool: RawPool, target_buffer: EpochBuffer, on_done=None) -> None:
        self._pool    = pool
        self._buf     = target_buffer
        self._on_done = on_done  # optional zero-arg callable fired when buffer is ready
        self._done    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._done.clear()
        self._thread.start()

    def wait(self) -> None:
        self._done.wait()

    def _run(self) -> None:
        try:
            self._pool.fill_buffer(self._buf)
        except Exception as exc:
            logger.error("AsyncAugmentor error: %s", exc, exc_info=True)
        finally:
            self._done.set()
            logger.debug("AsyncAugmentor — next epoch buffer ready (%d samples)", len(self._buf))
            if self._on_done is not None:
                try:
                    self._on_done()
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# MultiViewAsyncAugmentor — background thread for multi-view buffers
# ─────────────────────────────────────────────────────────────────────────────

class MultiViewAsyncAugmentor:
    """Background thread: fill a ``MultiViewEpochBuffer`` with fresh two-view data.

    Drop-in replacement for ``AsyncAugmentor`` when training with selflabel mode.
    Uses ``RawPool.fill_multiview_buffer`` to augment each clip twice independently.
    """

    def __init__(self, pool: RawPool, target_buffer: MultiViewEpochBuffer, on_done=None) -> None:
        self._pool    = pool
        self._buf     = target_buffer
        self._on_done = on_done
        self._done    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._done.clear()
        self._thread.start()

    def wait(self) -> None:
        self._done.wait()

    def _run(self) -> None:
        try:
            self._pool.fill_multiview_buffer(self._buf)
        except Exception as exc:
            logger.error("MultiViewAsyncAugmentor error: %s", exc, exc_info=True)
        finally:
            self._done.set()


# ─────────────────────────────────────────────────────────────────────────────
# LiveAugDataset / LiveMultiViewDataset — on-the-fly augmentation
# ─────────────────────────────────────────────────────────────────────────────

class LiveAugDataset(Dataset):
    """Augments each clip on-the-fly inside ``__getitem__``.

    Replaces the pre-built ``EpochBuffer`` / ``_DoubleBuffer`` pattern,
    cutting peak RAM from ~3× to ~1× the raw waveform set.

    The augmentation timer is accurate because all DataLoaders in this
    codebase use ``num_workers=0`` (main-process iteration).
    """

    def __init__(self, pool: "RawPool") -> None:
        self._pool     = pool
        self._aug      = WaveformAugmentor(pool.aug_cfg, pool.noise_waveforms)
        self._aug_time = 0.0
        self._lock     = threading.Lock()

    # -- compatibility shims used by _build_loader / DomainBalancedBatchSampler --
    @property
    def _domains(self) -> List[int]:
        return self._pool.domain_flags

    @property
    def num_samples(self) -> int:
        return len(self._pool.waveforms)

    # -- timer helpers --
    def reset_timer(self) -> None:
        with self._lock:
            self._aug_time = 0.0

    @property
    def aug_time(self) -> float:
        """Seconds spent inside ``WaveformAugmentor`` this epoch."""
        return self._aug_time

    # -- Dataset interface --
    def __len__(self) -> int:
        return len(self._pool.waveforms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        wav = self._pool.waveforms[idx]
        t0 = time.perf_counter()
        aug_wav = self._aug(wav)
        with self._lock:
            self._aug_time += time.perf_counter() - t0
        return aug_wav, self._pool.labels[idx]


class LiveMultiViewDataset(Dataset):
    """Two-view on-the-fly augmentation for selflabel mode.

    Each ``__getitem__`` call creates two fresh ``WaveformAugmentor`` instances
    so the two views have fully independent random state.
    """

    def __init__(self, pool: "RawPool") -> None:
        self._pool     = pool
        self._aug_time = 0.0
        self._lock     = threading.Lock()

    @property
    def _domains(self) -> List[int]:
        return self._pool.domain_flags

    @property
    def num_samples(self) -> int:
        return len(self._pool.waveforms)

    def reset_timer(self) -> None:
        with self._lock:
            self._aug_time = 0.0

    @property
    def aug_time(self) -> float:
        return self._aug_time

    def __len__(self) -> int:
        return len(self._pool.waveforms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        wav   = self._pool.waveforms[idx]
        noise = self._pool.noise_waveforms
        aug1  = WaveformAugmentor(self._pool.aug_cfg, noise)
        aug2  = WaveformAugmentor(self._pool.aug_cfg, noise)
        t0 = time.perf_counter()
        v1, v2 = aug1(wav), aug2(wav)
        with self._lock:
            self._aug_time += time.perf_counter() - t0
        return v1, v2, self._pool.labels[idx], self._pool.domain_flags[idx]


# ─────────────────────────────────────────────────────────────────────────────
# DomainBalancedBatchSampler
# ─────────────────────────────────────────────────────────────────────────────

class DomainBalancedBatchSampler(torch.utils.data.Sampler):
    """Yields batches with at least *min_target_frac* of samples from the
    target domain, cycling through the (smaller) target pool as needed.

    Parameters
    ----------
    domain_flags : sequence of int
        Per-sample flag: ``1`` = target domain, ``0`` = source / other.
    batch_size : int
    min_target_frac : float
        Minimum fraction of each batch that must be target-domain samples.
        Default ``0.25`` (25 %).
    shuffle : bool
        Shuffle source and target pools at the start of each iteration.
    rank, world_size :
        DDP rank / world size.  Each rank operates on a consistent shard of
        both the target and source index pools so every sample appears in
        exactly one rank's batches per epoch.
    epoch : int
        Seed offset for deterministic-but-varying shuffles across epochs.
    """

    def __init__(
        self,
        domain_flags,
        batch_size: int,
        min_target_frac: float = 0.25,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
    ) -> None:
        all_tgt = [i for i, d in enumerate(domain_flags) if d == 1]
        all_src = [i for i, d in enumerate(domain_flags) if d != 1]

        # Shard per rank (round-robin, reproducible)
        self._tgt = [i for j, i in enumerate(all_tgt) if j % world_size == rank]
        self._src = [i for j, i in enumerate(all_src) if j % world_size == rank]

        self._bs        = batch_size
        self._tgt_per_b = max(1, int(min_target_frac * batch_size))
        self._src_per_b = batch_size - self._tgt_per_b
        self._shuffle   = shuffle
        self._epoch     = epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return max(0, (len(self._tgt) + len(self._src)) // self._bs)

    def __iter__(self):
        import random
        rng = random.Random(self._epoch)

        tgt = list(self._tgt)
        src = list(self._src)
        if self._shuffle:
            rng.shuffle(tgt)
            rng.shuffle(src)

        tgt_i = src_i = 0

        for _ in range(len(self)):
            batch = []

            # Fill target slots — cycle pool if exhausted.
            # If there are no target samples at all, fall back to source-only.
            needed = self._tgt_per_b
            if not tgt:
                needed = 0  # no target samples available — skip, take all from source
                logger.warning(f"DomainBalancedBatchSampler, rank[{self.rank}] — no target samples available, "
                                "falling back to source-only batches")
            while needed > 0:
                take = min(needed, len(tgt) - tgt_i)
                if take <= 0:
                    if self._shuffle:
                        rng.shuffle(tgt)
                    tgt_i = 0
                    take = min(needed, len(tgt))
                    if take <= 0:
                        break  # safety: should never happen, but avoid infinite loop
                batch.extend(tgt[tgt_i : tgt_i + take])
                tgt_i += take
                needed -= take

            # Fill source slots — cycle pool if exhausted
            needed = self._src_per_b
            while needed > 0:
                take = min(needed, len(src) - src_i)
                if take <= 0:
                    if self._shuffle:
                        rng.shuffle(src)
                    src_i = 0
                    take = min(needed, len(src))
                batch.extend(src[src_i : src_i + take])
                src_i += take
                needed -= take

            rng.shuffle(batch)  # mix target/source positions within batch
            yield batch
