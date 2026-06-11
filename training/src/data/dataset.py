"""DCASE 2025 Task 2 audio dataset scanner.

Standalone — no pipeline3 imports.

Directory layout expected::

    data_root/
      fan/
        attributes_00.csv
        train/*.wav
        test/*.wav
        supplemental/*.wav

Filename conventions::

    section_00_source_train_normal_0326_n_B.wav   → attr_str = "n_B"
    section_00_source_train_normal_0765_noAttribute.wav → attr_str = "noAttribute"
    section_00_noise_supplemental_normal_0000.wav  → domain = "noise"
    section_00_source_supplemental_machine_0000.wav → clean supplemental
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data-root configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataRoot:
    """Configuration for a single dataset root directory.

    Parameters
    ----------
    path:
        Absolute (or relative) path to the root folder that contains
        machine-type sub-directories.
    load_csv:
        Whether to load ``attributes_{section:02d}.csv`` files from this
        root for hierarchical attribute encoding.  Set ``False`` for roots
        that do not ship CSV files (e.g. a denoised mirror).
    name:
        Human-readable label used in log messages.  Defaults to the last
        path component.
    """
    path:     str
    load_csv: bool = True
    name:     Optional[str] = None

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = Path(self.path).name


def _as_data_root(value: Union[str, DataRoot], *, load_csv: bool = False) -> DataRoot:
    """Coerce a plain path string to a :class:`DataRoot` with *load_csv* default."""
    if isinstance(value, DataRoot):
        return value
    return DataRoot(path=value, load_csv=load_csv)


# ─────────────────────────────────────────────────────────────────────────────
# Sample metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    path:         Path
    machine_type: str
    section:      int
    domain:       str   # "source" | "target" | "noise"
    split:        str   # "train" | "supplemental"
    anomaly:      int   # 0 = normal, 1 = anomaly, -1 = unknown
    attr_str:     str   # e.g. "n_B", "noAttribute"
    channel:      Optional[int] = None  # None=legacy mono mix, else explicit channel index
    # CSV-sourced structured attributes — populated when load_hierarchical=True
    attributes:   Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Filename parser
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CSV attribute loader
# ─────────────────────────────────────────────────────────────────────────────

_SPEED_RE = re.compile(r'(\d+(?:[._]\d+)?)V?$', re.IGNORECASE)


def parse_continuous_value(val_str: str) -> Optional[float]:
    """Parse a continuous attribute string to a raw float.

    Examples::

        "31V"  → 31.0
        "7_5V" → 7.5
        "0"    → 0.0
        "5"    → 5.0
        "B1"   → None  (not numeric)
    """
    s = val_str.strip().replace("_", ".")
    m = _SPEED_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def load_attributes_csv(
    data_root: str,
    machine_type: str,
    section: int,
) -> Dict[str, Dict[str, str]]:
    """Load ``attributes_{section:02d}.csv`` → ``{wav_stem: {param: value}}``.

    Handles the three CSV format variants in the dataset:
    - With machine-type path prefix + ``.wav`` suffix (fan, gearbox, valve)
    - No prefix, no suffix (ToyCar, ToyTrain)
    - No attribute columns at all (bearing, slider, ToyTrain)
    """
    csv_path = Path(data_root) / machine_type / f"attributes_{section:02d}.csv"
    if not csv_path.exists():
        return {}

    result: Dict[str, Dict[str, str]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return {}
        for row in reader:
            if not row:
                continue
            # Normalise to bare stem (strip directory prefix and extension)
            stem = Path(row[0].strip()).stem
            attrs: Dict[str, str] = {}
            for j in range(1, len(row) - 1, 2):
                param = row[j].strip()
                val   = row[j + 1].strip() if j + 1 < len(row) else ""
                if param:
                    attrs[param] = val
            result[stem] = attrs
    return result
def _parse_stem(stem: str) -> Tuple[int, str, int, str]:
    """Parse a DCASE 2025 filename stem → (section, domain, anomaly, attr_str)."""
    parts = stem.split("_")
    section = int(parts[1]) if len(parts) > 1 else 0
    domain  = parts[2] if len(parts) > 2 else "unknown"

    if domain == "noise":
        return section, "noise", 0, ""

    # section_XX_domain_split_condition_idx_attr...
    condition_str = parts[4] if len(parts) > 4 else "unknown"
    if condition_str == "anomaly":
        anomaly = 1
    elif condition_str == "normal":
        anomaly = 0
    else:
        anomaly = -1

    attr_str = "_".join(parts[6:]) if len(parts) > 6 else ""
    return section, domain, anomaly, attr_str


# ─────────────────────────────────────────────────────────────────────────────
# WAV loader (scipy — no torchaudio dependency)
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(
    path: Path,
    target_sr: int,
    clip_samples: int,
    channel: Optional[int] = None,
) -> torch.Tensor:
    """Load a WAV file → ``(1, clip_samples)`` float32 tensor."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(str(path))

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2_147_483_648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 2:
        if channel is None:
            # Legacy behaviour: collapse stereo to mono by averaging channels.
            #data = data.mean(axis=1)
            data = data[:, 0]  # Just take ch0 by default, to preserve any channel-specific info.     
            #data = data[:, 0] - data[:, 1]  
        else:
            # Explicit channel mode used when duplicating samples per channel.
            ch = int(channel)
            if 0 <= ch < data.shape[1]:
                data = data[:, ch]
            else:
                logger.warning(
                    "load_wav | requested channel %d out of range for %s (shape=%s) — using ch0",
                    ch, path, tuple(data.shape),
                )
                data = data[:, 0]

    if sr != target_sr:
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    if len(data) < clip_samples:
        data = np.pad(data, (0, clip_samples - len(data)))
    else:
        data = data[:clip_samples]

    return torch.from_numpy(data).unsqueeze(0)   # (1, T)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset scanner
# ─────────────────────────────────────────────────────────────────────────────

def _discover_machine_types(data_root: Path) -> List[str]:
    return sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _scan_root(
    root: Path,
    machine_types: List[str],
    splits: List[str],
    include_anomaly: bool,
    load_hierarchical: bool,
    csv_cache: Dict[Tuple[str, int], Dict[str, Dict[str, str]]],
    *,
    root_name: str = "",
) -> List[Sample]:
    """Internal helper: scan one data root and return Sample objects."""
    samples: List[Sample] = []
    for mt in machine_types:
        mt_path = root / mt
        if not mt_path.exists():
            continue
        for split in splits:
            folder = mt_path / split
            if not folder.exists():
                continue
            for wav in sorted(folder.glob("*.wav")):
                sec, domain, anomaly, attr_str = _parse_stem(wav.stem)

                # Skip noise clips (belong to noise pool, not training set)
                if domain == "noise" or "_noise_" in wav.name:
                    continue

                # Supplemental: only clean machine clips  (_machine_ marker)
                if split == "supplemental" and "_machine_" not in wav.name:
                    continue

                # For supplemental clean clips the anomaly flag may be absent
                if split == "supplemental":
                    anomaly = 0
                    domain  = domain if domain in ("source", "target") else "source"

                # Only normal clips for training (unless caller wants all labels)
                if not include_anomaly and anomaly not in (0,):
                    continue

                # Optionally attach CSV-sourced attributes
                attributes: Dict[str, str] = {}
                if load_hierarchical:
                    key = (mt, sec)
                    if key not in csv_cache:
                        csv_cache[key] = load_attributes_csv(str(root), mt, sec)
                    attributes = csv_cache[key].get(wav.stem, {})

                samples.append(Sample(
                    path=wav,
                    machine_type=mt,
                    section=sec,
                    domain=domain,
                    split=split,
                    anomaly=anomaly,
                    attr_str=attr_str,
                    attributes=attributes,
                ))
    return samples


def scan_samples(
    data_root: str,
    machine_types: Optional[List[str]],
    splits: List[str],
    target_sr: int = 16_000,
    clip_duration: float = 10.0,
    load_hierarchical: bool = False,
    include_anomaly: bool = False,
    extra_data_roots: Optional[List[Union[str, DataRoot]]] = None,
) -> List[Sample]:
    """Scan dataset directories and return a list of Sample metadata objects.

    By default includes only normal (anomaly==0) clips.  Pass
    ``include_anomaly=True`` to also include anomaly (anomaly==1) and unknown
    (anomaly==-1) clips — useful for test-set visualisation.
    Supplemental noise clips (``_noise_`` in filename) are excluded here —
    they are collected separately via :func:`scan_noise_pool`.

    Parameters
    ----------
    data_root:
        Primary root folder containing machine-type subdirectories.
    machine_types:
        ``None`` → auto-discover all subdirectories from *data_root*.
    splits:
        Which folder names to include: ``["train"]``, ``["train", "supplemental"]``, …
    extra_data_roots:
        Optional list of additional roots to scan.  Each entry is either:

        * A plain ``str`` path — treated as :class:`DataRoot` with
          ``load_csv=False`` (safe default for roots without CSV files).
        * A :class:`DataRoot` instance — full per-root control over
          CSV loading and display name.

        Example::

            scan_samples(
                data_root="/data/raw",
                extra_data_roots=[
                    "/data/denoised",                      # str: no CSV
                    DataRoot("/data/other", load_csv=True), # explicit CSV
                ],
            )
    """
    root = Path(data_root)
    if machine_types is None:
        machine_types = _discover_machine_types(root)

    # CSV cache shared across all roots that have load_csv=True.
    # Key: (machine_type, section) → {stem: {param: value}}
    _csv_cache: Dict[Tuple[str, int], Dict[str, Dict[str, str]]] = {}

    samples: List[Sample] = _scan_root(
        root, machine_types, splits, include_anomaly, load_hierarchical, _csv_cache,
        root_name=Path(data_root).name,
    )

    for entry in (extra_data_roots or []):
        dr = _as_data_root(entry, load_csv=False)
        extra_root = Path(dr.path)
        extra_mts = [mt for mt in machine_types if (extra_root / mt).exists()]
        if not extra_mts:
            logger.warning(
                "scan_samples | extra_root '%s' has no matching machine folders — skipping",
                dr.name,
            )
            continue
        extra_samples = _scan_root(
            extra_root, extra_mts, splits, include_anomaly,
            load_hierarchical=load_hierarchical and dr.load_csv,
            csv_cache=_csv_cache if dr.load_csv else {},
            root_name=dr.name,
        )
        logger.info(
            "scan_samples | root='%s' | %d clips added", dr.name, len(extra_samples)
        )
        samples.extend(extra_samples)

    logger.info(
        "scan_samples | %d clips total | machines=%s | splits=%s",
        len(samples), machine_types, splits,
    )
    return samples


def scan_noise_pool(
    data_root: str,
    machine_types: Optional[List[str]] = None,
    extra_data_roots: Optional[List[Union[str, DataRoot]]] = None,
) -> List[Path]:
    """Return paths to all supplemental noise clips across machine types.

    Parameters
    ----------
    extra_data_roots:
        Optional additional roots (plain paths or :class:`DataRoot` objects)
        to also collect noise clips from.
    """
    root = Path(data_root)
    if machine_types is None:
        machine_types = _discover_machine_types(root)

    def _collect_noise(r: Path, mts: List[str]) -> List[Path]:
        found: List[Path] = []
        for mt in mts:
            supp = r / mt / "supplemental"
            if not supp.exists():
                continue
            for wav in sorted(supp.glob("*.wav")):
                if "_noise_" in wav.name:
                    found.append(wav)
        return found

    paths: List[Path] = _collect_noise(root, machine_types)

    for entry in (extra_data_roots or []):
        dr = _as_data_root(entry, load_csv=False)
        extra_root = Path(dr.path)
        extra_mts = [mt for mt in machine_types if (extra_root / mt).exists()]
        extra_paths = _collect_noise(extra_root, extra_mts)
        logger.info(
            "scan_noise_pool | root='%s' | %d noise clips added", dr.name, len(extra_paths)
        )
        paths.extend(extra_paths)

    logger.info("scan_noise_pool | %d noise clips found total", len(paths))
    return paths
