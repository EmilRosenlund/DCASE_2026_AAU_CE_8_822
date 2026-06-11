"""
Score-Based Fusion Anomaly Detector
====================================
Optimized rewrite of gwrp_knn_all.py.

Key optimizations over the original:
  1. Parallel scoring — KNN, GMM, GWRP, and Autoencoder run concurrently via
     ThreadPoolExecutor (GIL is released by numpy/sklearn C code).
  2. Memory-efficient GWRP — uses chunked cosine-distance computation and
     limits the neighbor count for denominator calculation instead of N×N.
  3. Autoencoder with early stopping — avoids wasted epochs when loss plateaus.
  4. Proper resource management — no leaked file handles.
  5. Vectorised rank normalisation across all scorers.
  6. Metadata loaded once (from first embedding path) rather than re-read per path.
  7. Pluggable scorer registry — enable / disable individual scorers via config.

Scorer configuration
--------------------
In your ``config.yaml``, add an ``enabled_scorers`` list under ``classification``
to control which scorers are active.  If the key is absent **all** scorers run.

  classification:
    enabled: true
    module: "score_based_fusion"
    enabled_scorers:
      - knn
      - gmm
      - gwrp
      # - ae        # ← commented out = disabled

Available scorer names: ``knn``, ``gmm``, ``gwrp``, ``ae``.
"""

from sklearn.preprocessing._data import MinMaxScaler
from transformers.models.audio_spectrogram_transformer import feature_extraction_audio_spectrogram_transformer
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.stats import hmean, rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.utils as utils
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Autoencoder (module-level so it can be pickled / reused)
# ---------------------------------------------------------------------------
    
class _EmbedAutoencoder(nn.Module):
    """Symmetric autoencoder with GELU activations."""

    def __init__(self, input_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        hidden_dim = max(input_dim // 2, bottleneck_dim * 2)
        self.encoder = nn.Sequential(
            utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.1),
            utils.spectral_norm(nn.Linear(hidden_dim, bottleneck_dim)),
        )
        self.decoder = nn.Sequential(
            utils.spectral_norm(nn.Linear(bottleneck_dim, hidden_dim)),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------
class AnomalyDetector:
    # All available scorers.  Each entry maps a short name to a tuple of:
    #   (method_name, uses_raw_data)
    # * method_name  — name of the ``compute_*`` method on this class.
    # * uses_raw_data — if True the scorer receives the **raw** (unscaled)
    #   embeddings; if False it receives the preprocessed version.
    #
    # To add a new scorer: implement ``compute_foo(self, X_train, X_test)``
    # and add  "foo": ("compute_foo", False)  here.
    AVAILABLE_SCORERS: Dict[str, Tuple[str, bool]] = {
        "knn":  ("compute_knn",         False),   # preprocessed
        "gmm":  ("compute_gmm",         False),   # preprocessed
        "gwrp": ("compute_gwrp",        True),    # raw embeddings
        "ae":   ("compute_autoencoder", False),   # preprocessed
    }

    # Default set of enabled scorers (used when config doesn't specify any)
    DEFAULT_ENABLED: List[str] = ["knn"]

    def __init__(self, config: dict):
        """
        Initialize anomaly detector with a loaded config dictionary.

        Parameters
        ----------
        config : dict
            Loaded YAML configuration.
        """
        if config is None:
            raise ValueError("A config dictionary must be provided.")
        self.config = config

        env = config.get("environment", "local")
        paths_cfg = config.get("paths", {})
        embedding_root = paths_cfg.get(env, {}).get("embeddings", "")
        models = config["embeddings"]["modules"]

        self.embeddings_path: List[str] = [
            os.path.normpath(os.path.join(embedding_root, m)) for m in models
        ]

        for path in self.embeddings_path:
            if not os.path.exists(path):
                print(f"!! Warning: Path does not exist: {path}")

        classification_cfg = config.get("classification", {})
        self.threshold: float = classification_cfg.get("threshold", 0.5)
        machine_types_str: str = classification_cfg.get("machine_types", "")
        self.machine_types: Optional[List[str]] = (
            [m.strip() for m in machine_types_str.split(",") if m.strip()]
            if machine_types_str
            else None
        ) 

        # --- Scorer selection ---
        enabled_cfg = classification_cfg.get("enabled_scorers", None)
        if enabled_cfg is not None:
            # Validate names from config
            self.enabled_scorers: List[str] = [
                s for s in enabled_cfg if s in self.AVAILABLE_SCORERS
            ]
            unknown = set(enabled_cfg) - set(self.AVAILABLE_SCORERS)
            if unknown:
                print(f"!! Warning: Unknown scorers ignored: {unknown}")
        else:
            self.enabled_scorers = list(self.DEFAULT_ENABLED)

        self.results: Dict = {}
        print(f"Initialized AnomalyDetector  env='{env}'  "
              f"machines={self.machine_types or 'auto-discover'}")
        print(f"Enabled scorers: {self.enabled_scorers}")

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_section(path: str) -> str:
        """Extract section ID from a file path."""
        match = re.search(r"section_(\d+)", path)
        return match.group(1) if match else "00"

    @staticmethod
    def _load_lines(filepath: str) -> List[str]:
        """Read lines from *filepath* safely (closes the handle)."""
        with open(filepath, "r") as fh:
            return [line.strip() for line in fh]

    # ------------------------------------------------------------------
    # Individual scorers
    # ------------------------------------------------------------------
    def compute_knn(self, X_train: np.ndarray, X_test: np.ndarray,
                    n_neighbors: int = 6) -> np.ndarray:
        """Cosine KNN distance to the nearest training neighbour."""
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(X_train)
        distances, _ = nn.kneighbors(X_test)
        return distances[:, n_neighbors - 1]

    def compute_gwrp(self, X_train: np.ndarray, X_test: np.ndarray,
                     r: float = 0.98,
                     k_denom: Optional[int] = None,
                     chunk_size: int = 512) -> np.ndarray:
        """
        GWRP-scaled cosine anomaly score.

        Optimizations vs. original
        --------------------------
        * *k_denom* limits the number of neighbours used to compute the
          training denominator (default: all). Using e.g. 50 cuts the
          NearestNeighbors fit from O(n²) memory to O(n·k).
        * Test-to-train cosine distances are computed in *chunk_size*
          blocks to avoid allocating the full (n_test × n_train) matrix.
        """
        n_train = X_train.shape[0]
        k = k_denom if k_denom is not None else n_train

        # 1. Denominator per training sample
        nn_model = NearestNeighbors(n_neighbors=min(k, n_train), metric="cosine")
        nn_model.fit(X_train)
        all_dists, _ = nn_model.kneighbors(X_train)

        # Exclude self-distance (column 0)
        neighbor_dists = all_dists[:, 1:]
        n_neighbors_used = neighbor_dists.shape[1]
        weights = np.power(r, np.arange(n_neighbors_used))
        denominators = np.dot(neighbor_dists, weights) + 1e-8  # (n_train,)

        # 2. Chunked minimum scaled distance
        n_test = X_test.shape[0]
        gwrp_scores = np.empty(n_test, dtype=np.float64)

        for start in range(0, n_test, chunk_size):
            end = min(start + chunk_size, n_test)
            raw = cosine_distances(X_test[start:end], X_train)  # (chunk, n_train)
            raw /= denominators  # in-place broadcast
            gwrp_scores[start:end] = raw.min(axis=1)

        return gwrp_scores

    def compute_gmm(self, X_train: np.ndarray, X_test: np.ndarray,
                    n_components: int = 6) -> np.ndarray:
        """Negative log-likelihood under a diagonal GMM (higher → more anomalous)."""
        gmm = GaussianMixture(
            n_components=n_components, covariance_type="diag", random_state=42
        )
        gmm.fit(X_train)
        return -gmm.score_samples(X_test)

    def compute_autoencoder(self, X_train: np.ndarray, X_test: np.ndarray,
                            epochs: int = 50, batch_size: int = 64,
                            lr: float = 1e-3, bottleneck_dim: int = 32,
                            patience: int = 5, seed: int = 42) -> np.ndarray:
        """
        Autoencoder reconstruction error with **early stopping**.

        Parameters
        ----------
        patience : int
            Stop training if the loss does not improve for *patience* epochs.
        seed : int
            Random seed for reproducible weight init and data shuffling.
        """ 

        # Seed everything for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        device = torch.device("cpu")
        input_dim = X_train.shape[1]

        model = _EmbedAutoencoder(input_dim, bottleneck_dim).to(device)
        criterion = nn.MSELoss()
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        # Seeded generator for deterministic shuffling
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            TensorDataset(train_t, train_t),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        )

        # --- Training with early stopping ---
        best_loss = float("inf")
        stale = 0

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss =  criterion(reconstructed,batch_x)   #mse loss for reconstruction
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
            
            #if epoch % 10 == 0:
            #    print(f"Epoch {epoch}, Loss: {avg_loss}")

        # --- Inference ---
        model.eval()
        with torch.no_grad():
            reconstructed = model(test_t)
            anomaly_scores = F.mse_loss(test_t, reconstructed, reduction="none").mean(dim=1)

        return anomaly_scores.cpu().numpy()

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(X_train: np.ndarray, X_test: np.ndarray):
        """
        QuantileTransformer + hard-threshold + tanh squash.

        Returns copies — does **not** modify the originals.
        """
        qt = QuantileTransformer(output_distribution="uniform")
        Xtr = qt.fit_transform(X_train)
        Xte = qt.transform(X_test)

        # Hard threshold
        Xtr[np.abs(Xtr) < 0.1] = 0.0
        Xte[np.abs(Xte) < 0.1] = 0.0

        # Tanh squash for large values
        mask_tr = np.abs(Xtr) > 0.5
        Xtr[mask_tr] = np.tanh(Xtr[mask_tr])
        mask_te = np.abs(Xte) > 0.5
        Xte[mask_te] = np.tanh(Xte[mask_te])

        return Xtr, Xte

    # ------------------------------------------------------------------
    # Scoring pipeline for one embedding path
    # ------------------------------------------------------------------
    def _score_one_path(self, path: str, machine: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from *path*, compute **enabled** scores in parallel,
        and return a dict of rank-normalised score vectors.

        Returns
        -------
        Dict[str, np.ndarray]
            ``{scorer_name: rank_normalised_scores}``
        """
        X_train_raw = np.load(os.path.join(path, f"{machine}_train_embeddings.npy"))
        X_test_raw = np.load(os.path.join(path, f"{machine}_test_embeddings.npy"))

        # Only preprocess if at least one enabled scorer needs it
        needs_pp = any(
            not self.AVAILABLE_SCORERS[s][1] for s in self.enabled_scorers
        )
        if needs_pp:
            X_train_pp, X_test_pp = self._preprocess(X_train_raw, X_test_raw)
        else:
            X_train_pp, X_test_pp = None, None

        # Build {future: name} for only the enabled scorers
        raw_scores: Dict[str, np.ndarray] = {}
        with ThreadPoolExecutor(max_workers=len(self.enabled_scorers)) as pool:
            futures = {}
            for name in self.enabled_scorers:
                method_name, uses_raw = self.AVAILABLE_SCORERS[name]
                method = getattr(self, method_name)
                if uses_raw:
                    fut = pool.submit(method, X_train_raw, X_test_raw)
                else:
                    fut = pool.submit(method, X_train_pp, X_test_pp)
                futures[fut] = name

            for fut in as_completed(futures):
                scorer_name = futures[fut]
                raw_scores[scorer_name] = fut.result()

        # Rank-normalise each scorer independently
        n = len(next(iter(raw_scores.values())))
        normed: Dict[str, np.ndarray] = {
            name: rankdata(scores) / n for name, scores in raw_scores.items()
        }
        return normed

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Run anomaly detection across all machines and embedding paths."""
        if isinstance(self.embeddings_path, str):
            self.embeddings_path = [self.embeddings_path]

        missing = [p for p in self.embeddings_path if not os.path.exists(p)]
        if not self.embeddings_path:
            print("Error: No embeddings paths provided.")
            return self.results
        if missing:
            print("Error: Missing embedding paths:")
            for p in missing:
                print(f"  - {p}")
            return self.results

        machines = self.machine_types if self.machine_types else self._discover_machines()
        print(f"\nScore-Based Fusion  (threshold={self.threshold})")
        print("=" * 70)

        machine_aucs: List[float] = []
        machine_paucs: List[float] = []

        for machine in machines:
            print(f"\nProcessing machine: {machine}")
            print("-" * 70)

            # --- Load metadata once (from first available path) ---
            y_test, domains, sections = None, None, None
            for path in self.embeddings_path:
                labels_file = os.path.join(path, f"{machine}_test_labels.txt")
                domains_file = os.path.join(path, f"{machine}_test_domains.txt")
                paths_file = os.path.join(path, f"{machine}_test_file_paths.txt")

                if os.path.exists(labels_file):
                    y_test = np.loadtxt(labels_file, dtype=int)
                if os.path.exists(domains_file):
                    domains = np.array(self._load_lines(domains_file))
                if os.path.exists(paths_file):
                    sections = np.array(
                        [self._parse_section(l) for l in self._load_lines(paths_file)]
                    )
                break  # metadata only needed from one path

            # --- Score every embedding path ---
            all_scores: List[Dict[str, np.ndarray]] = []
            for idx, path in enumerate(self.embeddings_path):
                try:
                    print(f"  [{idx + 1}/{len(self.embeddings_path)}] {os.path.basename(path)}")
                    scores = self._score_one_path(path, machine)
                    all_scores.append(scores)
                except Exception as exc:
                    print(f"    ✗ Error: {exc}")

            if not all_scores:
                print("  No valid scores produced — skipping machine.")
                continue

            # --- Fuse across embedding paths (mean of rank-normalised scores) ---
            fused = {}
            for name in self.enabled_scorers:
                fused[name] = np.mean(
                    [s[name] for s in all_scores], axis=0
                )

            stacked = np.stack(list(fused.values()))
            combined = np.min(stacked, axis=0)  # Min-pooling

            # --- DCASE evaluation ---
            if y_test is None:
                print("  No labels available.")
                continue

            print(f"\n  Results for {machine}:")
            section_aucs, section_paucs = [], []

            if sections is not None and domains is not None:
                for section in np.unique(sections):
                    sec_mask = sections == section
                    domain_aucs = []
                    for domain in ("source", "target"):
                        dmask = sec_mask & (domains == domain)
                        if dmask.sum() > 0 and len(np.unique(y_test[dmask])) > 1:
                            auc = roc_auc_score(y_test[dmask], combined[dmask])
                            domain_aucs.append(auc)
                            print(f"    Section {section} / {domain}: AUC={auc:.4f}")
                    if domain_aucs:
                        sec_auc = np.mean(domain_aucs)
                        section_aucs.append(sec_auc)
                        if len(np.unique(y_test[sec_mask])) > 1:
                            pauc = roc_auc_score(
                                y_test[sec_mask], combined[sec_mask], max_fpr=0.1
                            )
                            section_paucs.append(pauc)
                            print(f"    Section {section}: Mean AUC={sec_auc:.4f}  "
                                  f"pAUC={pauc:.4f}\n")

                if section_aucs:
                    machine_aucs.append(np.mean(section_aucs))
                if section_paucs:
                    machine_paucs.append(np.mean(section_paucs))
            else:
                print("    No sections / domains available for scoring.")

            # --- Scorer contribution analysis (Logistic Regression weights) ---
            if y_test is not None and len(np.unique(y_test)) > 1 and len(self.enabled_scorers) > 1:
                try:
                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(stacked.T, y_test)
                    print(f"  Scorer Contributions ({machine}):")
                    for name, weight in zip(self.enabled_scorers, lr.coef_[0]):
                        print(f"    {name.upper():>6s}  weight: {weight:+.4f}")
                    print()
                except Exception as exc:
                    print(f"    (Contribution analysis skipped: {exc})")

        # --- Official DCASE Ω ---
        print("\n" + "=" * 70)
        if machine_aucs and machine_paucs:
            mean_auc = np.mean(machine_aucs)
            mean_pauc = np.mean(machine_paucs)
            omega = hmean(machine_aucs + machine_paucs)
            print(f"DCASE Official Score (Ω): {omega:.4f}")
            print(f"  Mean AUC : {mean_auc:.4f}")
            print(f"  Mean pAUC: {mean_pauc:.4f}")
            self.results["official_score"] = {
                "omega": omega,
                "mean_auc": mean_auc,
                "mean_pauc": mean_pauc,
                "machine_aucs": machine_aucs,
                "machine_paucs": machine_paucs,
            }
        else:
            print("Not enough data to compute DCASE Ω score.")
        print("=" * 70)

        return self.results

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover_machines(self) -> List[str]:
        """Discover machine types from the first embeddings directory."""
        base = (
            self.embeddings_path[0]
            if isinstance(self.embeddings_path, list)
            else self.embeddings_path
        )
        if not os.path.exists(base):
            print(f"Error: Discovery path missing: {base}")
            return []

        suffix = "_train_embeddings.npy"
        machines = sorted(
            {f[:-len(suffix)] for f in os.listdir(base) if f.endswith(suffix)}
        )
        print(f"Discovered {len(machines)} machines: {machines}")
        return machines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(config: Optional[Dict] = None) -> Dict:
    """Main entry point using a loaded YAML config."""
    if config is None:
        raise ValueError("A loaded YAML config dictionary must be provided.")
    detector = AnomalyDetector(config)
    return detector.run()


if __name__ == "__main__":
    with open("pipeline/config.yaml") as f:
        config = yaml.safe_load(f)
    main(config)
