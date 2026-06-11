"""DCASE Task 2 official scoring — pure numpy, no file I/O.

This module is intentionally framework-agnostic so it can be called from
both the ArcFace trainer (in-memory embeddings) and the standalone
AnomalyDetector pipeline (embeddings loaded from disk).

Public API
----------
compute_gwrp_scores(X_train, X_test, r) -> (scores, train_denominators)
dcase_score(embeddings_train, embeddings_test, labels, domains, sections)
    -> DCASEResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DCASEResult:
    """Holds per-machine and aggregate DCASE scores.

    Attributes
    ----------
    machine_aucs:
        Mean AUC per machine (averaged over sections and source/target domains).
    machine_paucs:
        pAUC (max_fpr=0.1) per machine (averaged over sections).
    per_machine:
        Detailed per-machine breakdown: section → domain → auc.
    omega:
        Official DCASE score — harmonic mean of all machine AUCs and pAUCs.
        ``None`` when there are fewer than 2 machines with valid scores.
    mean_auc:
        Macro mean of ``machine_aucs``.
    mean_pauc:
        Macro mean of ``machine_paucs``.
    monitor:
        Single scalar to use for early stopping / best-checkpoint selection.
        Equal to ``omega`` when available, otherwise ``mean_auc``.
        Higher is better.
    """
    machine_aucs:  list[float]               = field(default_factory=list)
    machine_paucs: list[float]               = field(default_factory=list)
    per_machine:   dict[str, dict]           = field(default_factory=dict)
    omega:         float | None              = None
    mean_auc:      float                     = 0.0
    mean_pauc:     float                     = 0.0
    monitor:       float                     = 0.0

    def log_dict(self, prefix: str = "val") -> dict[str, float]:
        """Return a flat dict suitable for WandB / logger."""
        d: dict[str, float] = {
            f"{prefix}/dcase_omega":     self.omega    or 0.0,
            f"{prefix}/dcase_mean_auc":  self.mean_auc,
            f"{prefix}/dcase_mean_pauc": self.mean_pauc,
        }
        for machine, info in self.per_machine.items():
            for section, sinfo in info.items():
                for domain, auc in sinfo.get("domain_aucs", {}).items():
                    d[f"{prefix}/{machine}_s{section}_{domain}_auc"] = auc
                if "pauc" in sinfo:
                    d[f"{prefix}/{machine}_s{section}_pauc"] = sinfo["pauc"]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_gwrp_scores(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    r:       float = 0.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GWRP anomaly scores.

    A_GWRP(x, X_ref | r) := min_{y in X_ref} [ A_cos(x, y) / Denom(y, X_ref | r) ]

    Parameters
    ----------
    X_train:
        (N_train, D) normalised training embeddings.
    X_test:
        (N_test, D) normalised test embeddings.
    r:
        Geometric decay factor for neighbour weights.

    Returns
    -------
    gwrp_scores:
        (N_test,) anomaly score per test sample.
    train_denominators:
        (N_train,) pre-computed scaling factors (useful for threshold estimation).
    """
    num_train = X_train.shape[0]

    # All-neighbour fit to compute denominators
    nn_full = NearestNeighbors(n_neighbors=num_train, metric="cosine")
    nn_full.fit(X_train)
    all_train_dists, _ = nn_full.kneighbors(X_train)

    neighbor_dists     = all_train_dists[:, 1:]  # exclude self
    weights            = np.power(r, np.arange(num_train - 1))
    train_denominators = np.sum(neighbor_dists * weights, axis=1) + 1e-8

    # All test→train cosine distances, scaled by each training point's denominator
    raw_dists    = cosine_distances(X_test, X_train)           # (N_test, N_train)
    scaled_dists = raw_dists / train_denominators              # broadcast over rows
    gwrp_scores  = np.min(scaled_dists, axis=1)                # (N_test,)

    return gwrp_scores, train_denominators


def dcase_score(
    embeddings_train: np.ndarray,
    embeddings_test:  np.ndarray,
    labels:           np.ndarray,
    domains:          np.ndarray,
    sections:         np.ndarray,
    machine:          str  = "unknown",
    r:                float = 0.96,
) -> DCASEResult:
    """Compute the official DCASE score for a single machine type.

    Parameters
    ----------
    embeddings_train:
        (N_train, D) raw (un-scaled) embeddings from training split.
    embeddings_test:
        (N_test, D) raw (un-scaled) embeddings from test split.
    labels:
        (N_test,) binary ground-truth labels (0=normal, 1=anomaly).
    domains:
        (N_test,) string domain labels, e.g. ``"source"`` / ``"target"``.
    sections:
        (N_test,) section IDs (string or int).
    machine:
        Machine type name, used only for the ``per_machine`` breakdown key.
    r:
        GWRP decay factor.

    Returns
    -------
    DCASEResult
    """
    # Scale
    scaler           = MinMaxScaler()
    X_train_scaled   = scaler.fit_transform(embeddings_train)
    X_test_scaled    = scaler.transform(embeddings_test)

    gwrp_scores, _   = compute_gwrp_scores(X_train_scaled, X_test_scaled, r=r)

    # Accept both numeric domain flags (0/1) and string labels (source/target).
    domains_norm = np.asarray(domains)
    if domains_norm.dtype.kind in ("U", "S", "O"):
        mapped = []
        for d in domains_norm:
            s = str(d).strip().lower()
            if s in ("source", "0", "false"):
                mapped.append(0)
            elif s in ("target", "1", "true"):
                mapped.append(1)
            else:
                mapped.append(-1)
        domains_norm = np.asarray(mapped, dtype=int)
    else:
        domains_norm = domains_norm.astype(int, copy=False)

    # Per-section, per-domain AUC
    section_aucs:  list[float] = []
    section_paucs: list[float] = []
    per_section:   dict        = {}

    for sec in np.unique(sections):
        sec_mask    = sections == sec
        domain_aucs: dict[str, float] = {}

        for domain in (0, 1):  # source = 0, target = 1
            dmask = sec_mask & (domains_norm == domain)
            if dmask.sum() == 0 or len(np.unique(labels[dmask])) < 2:
                continue
            auc = roc_auc_score(labels[dmask], gwrp_scores[dmask])
            domain_aucs[domain] = auc

        sec_info: dict = {"domain_aucs": domain_aucs}

        if domain_aucs:
            section_aucs.append(float(np.mean(list(domain_aucs.values()))))

        if len(np.unique(labels[sec_mask])) >= 2:
            pauc = roc_auc_score(
                labels[sec_mask], gwrp_scores[sec_mask], max_fpr=0.1
            )
            section_paucs.append(pauc)
            sec_info["pauc"] = float(pauc)

        per_section[str(sec)] = sec_info

    machine_auc  = float(np.mean(section_aucs))  if section_aucs  else 0.0
    machine_pauc = float(np.mean(section_paucs)) if section_paucs else 0.0

    result = DCASEResult(
        machine_aucs  = [machine_auc],
        machine_paucs = [machine_pauc],
        per_machine   = {machine: per_section},
        mean_auc      = machine_auc,
        mean_pauc     = machine_pauc,
    )

    all_scores = result.machine_aucs + result.machine_paucs
    if len(all_scores) >= 2:
        result.omega   = float(hmean(all_scores))
        result.monitor = result.omega
    else:
        result.monitor = result.mean_auc

    return result


def dcase_score_multi_machine(
    per_machine_results: list[DCASEResult],
) -> DCASEResult:
    """Aggregate per-machine DCASEResults into a single overall result.

    Parameters
    ----------
    per_machine_results:
        One ``DCASEResult`` per machine type (from ``dcase_score``).

    Returns
    -------
    DCASEResult with ``omega``, ``mean_auc``, ``mean_pauc`` filled in.
    """
    all_aucs  = [r.machine_aucs[0]  for r in per_machine_results if r.machine_aucs]
    all_paucs = [r.machine_paucs[0] for r in per_machine_results if r.machine_paucs]
    all_per   = {}
    for r in per_machine_results:
        all_per.update(r.per_machine)

    mean_auc  = float(np.mean(all_aucs))  if all_aucs  else 0.0
    mean_pauc = float(np.mean(all_paucs)) if all_paucs else 0.0

    combined = all_aucs + all_paucs
    omega    = float(hmean(combined)) if len(combined) >= 2 else None

    return DCASEResult(
        machine_aucs  = all_aucs,
        machine_paucs = all_paucs,
        per_machine   = all_per,
        omega         = omega,
        mean_auc      = mean_auc,
        mean_pauc     = mean_pauc,
        monitor       = omega if omega is not None else mean_auc,
    )