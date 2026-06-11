"""ArcFace (Additive Angular Margin) classification head.

Reference:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition", CVPR 2019. https://arxiv.org/abs/1801.07698

Implementation is pure PyTorch — no extra dependencies.

Usage
-----
::

    head = ArcFaceHead(in_dim=768, num_classes=N, scale=30.0, margin=0.5)
    loss = head(embeddings, labels)   # embeddings: (B, D), labels: (B,) long
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """ArcFace margin classifier.

    Parameters
    ----------
    in_dim:
        Dimensionality of the input embedding (backbone output_dim).
    num_classes:
        Total number of identity classes.
    scale:
        Logit scaling factor *s* (default 30.0).  Higher values sharpen
        the softmax distribution.
    margin:
        Additive angular margin *m* in radians (default 0.5 ≈ 28.6°).
    easy_margin:
        When True, only apply the margin when cos(θ) > 0 (avoids gradient
        instability for very wrong predictions early in training).
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Class prototype weight bank — shape (num_classes, in_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)  # cos(π - m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(π - m) * m

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ArcFace loss.

        Parameters
        ----------
        embeddings : ``(B, in_dim)``  — raw (non-normalised) backbone output.
        labels     : ``(B,)``         — integer class indices in [0, num_classes).

        Returns
        -------
        loss : scalar cross-entropy with angular margin applied.
        """
        # L2-normalise embeddings and weight prototypes
        emb_norm = F.normalize(embeddings, p=2, dim=1)           # (B, D)
        w_norm   = F.normalize(self.weight, p=2, dim=1)          # (C, D)

        # cosine similarity between each embedding and each class prototype
        cos_theta = emb_norm @ w_norm.t()                         # (B, C)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(θ) via identity sin²(θ) + cos²(θ) = 1
        sin_theta = (1.0 - cos_theta.pow(2)).clamp(min=1e-7).sqrt()  # (B, C)

        # cos(θ + m) = cos(θ)·cos(m) − sin(θ)·sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m  # (B, C)

        if self.easy_margin:
            # Only apply margin when cos(θ) > 0
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # Smooth fallback when θ > π − m (target cosine too small)
            cos_theta_m = torch.where(
                cos_theta > self.threshold,
                cos_theta_m,
                cos_theta - self.mm,
            )

        # One-hot mask: swap target class cosine for the margined version
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale and cross-entropy
        logits *= self.scale
        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (argmax cosine similarity)."""
        emb_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm   = F.normalize(self.weight, p=2, dim=1)
        cos_theta = emb_norm @ w_norm.t()
        return cos_theta.argmax(dim=1)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, num_classes={self.num_classes}, "
            f"scale={self.scale}, margin={self.margin:.4f}"
        )
