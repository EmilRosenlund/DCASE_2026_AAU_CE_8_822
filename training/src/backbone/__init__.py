from .base import BaseBackbone
from .beats_backbone import BEATsBackbone
from .sslam_backbone import SSLAMBackbone


def get_backbone(cfg) -> BaseBackbone:
    """Factory: instantiate a backbone from its config block."""
    btype = cfg.backbone.type.lower()
    tcfg  = cfg.training
    grad_ckpt = bool(getattr(tcfg, "gradient_checkpointing", False))

    if btype == "beats":
        bc = cfg.backbone.beats
        return BEATsBackbone(
            checkpoint=bc.checkpoint,
            pool=bc.pool,
            fine_tune_layers=bc.fine_tune_layers,
            gradient_checkpointing=grad_ckpt,
        )

    if btype == "sslam":
        sc = getattr(cfg.backbone, "sslam", cfg.backbone)  # sslam sub-block or backbone directly
        return SSLAMBackbone(
            pool=getattr(sc, "pool", "mean"),
            fine_tune_layers=getattr(sc, "fine_tune_layers", "all"),
            gradient_checkpointing=grad_ckpt,
        )

    raise ValueError(
        f"Unknown backbone type: {cfg.backbone.type!r}. "
        "Supported: 'beats', 'sslam'."
    )


__all__ = ["BaseBackbone", "BEATsBackbone", "SSLAMBackbone", "get_backbone"]
