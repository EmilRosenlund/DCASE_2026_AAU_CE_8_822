"""Label encoder: composite identity strings → integer class IDs.

Label key format::

    "{machine_type}_sec{section:02d}_{domain}_{attr_str}"

Examples::

    fan_sec00_source_n_B
    fan_sec00_source_n_A
    bearing_sec00_source_noAttribute

Machines without attributes use the literal ``"noAttribute"`` string from
the filename, so they still get one class per (machine, section, domain) tuple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.dataset import Sample
from src.utils import get_logger

logger = get_logger(__name__)


class LabelEncoder:
    """Map ``Sample`` metadata → integer class IDs.

    Usage::

        enc = LabelEncoder()
        enc.fit(samples)
        label = enc.encode(sample)   # int in [0, num_classes)
    """

    def __init__(self) -> None:
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def label_key(sample: Sample) -> str:
        """Deterministic string representation of a sample's identity."""
        attr = sample.attr_str if sample.attr_str else "noAttribute"
        key = f"{sample.machine_type}_sec{sample.section:02d}_{sample.domain}_{attr}"
        if getattr(sample, "channel", None) is not None:
            key = f"{key}_ch{int(sample.channel)}"
        return key

    def fit(self, samples: List[Sample]) -> "LabelEncoder":
        """Build the vocabulary from a list of samples."""
        keys = sorted({self.label_key(s) for s in samples})
        self._str_to_int = {k: i for i, k in enumerate(keys)}
        self._int_to_str = {i: k for k, i in self._str_to_int.items()}
        logger.info("LabelEncoder — %d classes discovered", len(keys))
        for i, k in enumerate(keys):
            logger.debug("  class %4d: %s", i, k)
        return self

    def encode(self, sample: Sample) -> int:
        key = self.label_key(sample)
        if key not in self._str_to_int:
            raise KeyError(
                f"Label key {key!r} not in encoder vocabulary. "
                "Call fit() before encode()."
            )
        return self._str_to_int[key]

    @property
    def num_classes(self) -> int:
        return len(self._str_to_int)

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._str_to_int, f, indent=2)
        logger.info("LabelEncoder saved → %s (%d classes)", path, self.num_classes)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path) as f:
            str_to_int = json.load(f)
        enc = cls()
        enc._str_to_int = {k: int(v) for k, v in str_to_int.items()}
        enc._int_to_str = {v: k for k, v in enc._str_to_int.items()}
        logger.info("LabelEncoder loaded ← %s (%d classes)", path, enc.num_classes)
        return enc

    def class_name(self, idx: int) -> str:
        return self._int_to_str.get(idx, f"<unknown:{idx}>")
