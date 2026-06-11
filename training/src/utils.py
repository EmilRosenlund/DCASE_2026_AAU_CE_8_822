"""Standalone utility functions for the ArcFace fine-tuning pipeline."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_DATE_FMT = "%H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """Configure root logger.

    - Rank 0 always logs at *level* (default INFO).
    - Other ranks are silenced to WARNING so training progress lines only
      appear once, but errors/warnings from any rank are still visible.
    - The rank is embedded in the format when world_size > 1.
    """
    if world_size > 1 and rank != 0:
        # Non-main ranks: only warnings and above
        effective_level = max(getattr(logging, level.upper(), logging.INFO), logging.WARNING)
        fmt = f"%(asctime)s  %(levelname)-8s  [rank{rank}] %(name)s — %(message)s"
    else:
        effective_level = getattr(logging, level.upper(), logging.INFO)
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file and rank == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    # Force reconfigure: basicConfig is a no-op if root already has handlers
    # (e.g. BEATs imports logging and adds handlers before we get here).
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    logging.basicConfig(level=effective_level, format=fmt, datefmt=_DATE_FMT, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
