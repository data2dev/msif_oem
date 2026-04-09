"""
ESGD Model
===========
Ensemble Stochastic Gradient Descent for online alpha factor synthesis.

Implements the paper's Section 3.3:
  1. Mini-batch K-means clusters features by market regime
  2. One SGD linear regressor per cluster
  3. Online update: partial_fit regressors + periodic re-clustering
  4. Calinski-Harabasz rollback protection
"""

import logging
import pickle
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import rankdata
import config as cfg

log = logging.getLogger(__name__)


class ESGDModel:
    """Online Ensemble SGD with regime clustering."""

    def __init__(self):
        self.kmeans = None
        self.regressors: list[SGDRegressor] = []
        self.n_clusters = cfg.N_CLUSTERS

        # Online update state
        self.outlier_count = 0
        self.feature_archive = []   # (features, labels) for re-clustering
        self.cluster_labels = []
        self._ch_score = None       # cached CH index
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Initialization ────────────────────────

    def initialize(self, features: np.ndarray, labels: np.ndarray):
        """
        Full initialization from training data.
        
        Args:
            features: [N, D_o] — PTE-TFE output features
            labels: [N] — forward returns
        """
        log.info(f"Initializing ESGD: {features.shape[0]} samples, "
                 f"{features.shape[1]} features, K={self.n_clusters}")

        # Step 1: Cluster features
        # Average features per bar (for minute data, each "bar" may have multiple assets)
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=cfg.CLUSTER_BATCH_SIZE,
            max_iter=cfg.CLUSTER_MAX_ITER,
            random_state=42,
        )
        cluster_ids = self.kmeans.fit_predict(features)

        # Step 2: Train one SGD regressor per cluster
        self.regressors = []
        labels_ranked = self._csrank(labels)

        for k in range(self.n_clusters):
            mask = cluster_ids == k
            n_k = mask.sum()

            reg = SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=cfg.SGD_REGULARIZATION,
                learning_rate="invscaling",
                eta0=cfg.SGD_LR,
                random_state=42,
                warm_start=True,
            )

            if n_k > 0:
                reg.fit(features[mask], labels_ranked[mask])
                log.info(f"  Cluster {k}: {n_k} samples, "
                         f"coef_norm={np.linalg.norm(reg.coef_):.4f}")
            else:
                # Initialize with dummy data to set shapes
                reg.fit(features[:1], labels_ranked[:1])
                log.warning(f"  Cluster {k}: 0 samples (initialized with dummy)")

            self.regressors.append(reg)

        # Cache for re-clustering
        self.feature_archive = list(zip(features, labels))
        self.cluster_labels = cluster_ids.tolist()
        self._ch_score = self._compute_ch(features, cluster_ids)
        self._initialized = True

        log.info(f"ESGD initialized. CH index: {self._ch_score:.2f}")

    # ── Prediction ────────────────────────────

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict alpha factors.
        
        Args:
            features: [N, D_o] or [D_o]
            
        Returns:
            predictions: [N] alpha signals
        """
        if not self._initialized:
            raise RuntimeError("ESGD not initialized. Call initialize() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Find nearest cluster for the batch mean
        mean_feat = features.mean(axis=0, keepdims=True)
        cluster_id = self.kmeans.predict(mean_feat)[0]

        # Use that cluster's regressor
        predictions = self.regressors[cluster_id].predict(features)
        return predictions

    def predict_with_confidence(self, features: np.ndarray) -> tuple:
        """
        Predict with confidence score based on cluster distance.
        
        Returns:
            (predictions, confidence)
            confidence: 1.0 at centroid, decays with distance
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        mean_feat = features.mean(axis=0, keepdims=True)
        distances = self.kmeans.transform(mean_feat)[0]  # [K]
        nearest_k = np.argmin(distances)
        min_dist = distances[nearest_k]

        predictions = self.regressors[nearest_k].predict(features)
        confidence = 1.0 / (1.0 + min_dist)

        return predictions, confidence

    # ── Online Update ─────────────────────────

    def online_update(self, features: np.ndarray, labels: np.ndarray):
        """
        Online update step (Algorithm in paper's Section 3.3.2).
        
        Args:
            features: [N, D_o] new bar features (N = number of assets)
            labels: [N] realized returns
        """
        if not self._initialized:
            log.warning("Cannot online_update before initialization")
            return

        mean_feat = features.mean(axis=0, keepdims=True)

        # Check distance to all centroids
        distances = self.kmeans.transform(mean_feat)[0]
        nearest_k = np.argmin(distances)
        min_dist = distances[nearest_k]

        # Track outliers for re-clustering trigger
        if min_dist > cfg.D_MIN:
            self.outlier_count += 1
            log.debug(f"Outlier detected: dist={min_dist:.3f}, count={self.outlier_count}")

        # Update nearest centroid (SGD-like step on K-means)
        old_center = self.kmeans.cluster_centers_[nearest_k].copy()
        self.kmeans.cluster_centers_[nearest_k] += (
            cfg.CLUSTER_LR * (mean_feat[0] - old_center)
        )

        # Update that cluster's SGD regressor
        labels_ranked = self._csrank(labels)
        self.regressors[nearest_k].partial_fit(features, labels_ranked)

        # Archive
        for f, l in zip(features, labels):
            self.feature_archive.append((f, l))

        # Cap archive size
        max_archive = 50000
        if len(self.feature_archive) > max_archive:
            self.feature_archive = self.feature_archive[-max_archive:]

        # Re-clustering trigger
        if self.outlier_count >= cfg.C_MAX:
            self._recluster()

    def _recluster(self):
        """Full re-clustering with CH-index rollback protection."""
        log.info(f"Re-clustering triggered (outlier_count={self.outlier_count})")

        if len(self.feature_archive) < self.n_clusters * 10:
            log.warning("Not enough archived data for re-clustering")
            self.outlier_count = 0
            return

        features = np.array([f for f, _ in self.feature_archive])
        labels = np.array([l for _, l in self.feature_archive])

        # Cache old state
        old_kmeans = pickle.loads(pickle.dumps(self.kmeans))
        old_regressors = pickle.loads(pickle.dumps(self.regressors))
        old_ch = self._ch_score

        # New clustering
        new_kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=cfg.CLUSTER_BATCH_SIZE,
            max_iter=cfg.CLUSTER_MAX_ITER,
            random_state=None,  # different seed for diversity
        )
        new_ids = new_kmeans.fit_predict(features)
        new_ch = self._compute_ch(features, new_ids)

        log.info(f"Re-cluster CH: old={old_ch:.2f}, new={new_ch:.2f}")

        if new_ch >= old_ch:
            # Accept new clustering, rebuild regressors
            self.kmeans = new_kmeans
            self.regressors = []
            labels_ranked = self._csrank(labels)

            for k in range(self.n_clusters):
                mask = new_ids == k
                reg = SGDRegressor(
                    loss="squared_error", penalty="l2",
                    alpha=cfg.SGD_REGULARIZATION,
                    learning_rate="invscaling", eta0=cfg.SGD_LR,
                    random_state=42, warm_start=True,
                )
                if mask.sum() > 0:
                    reg.fit(features[mask], labels_ranked[mask])
                else:
                    reg.fit(features[:1], labels_ranked[:1])
                self.regressors.append(reg)

            self._ch_score = new_ch
            log.info("Re-clustering accepted")
        else:
            # Rollback
            self.kmeans = old_kmeans
            self.regressors = old_regressors
            log.info("Re-clustering rolled back (CH decreased)")

        self.outlier_count = 0

    # ── Helpers ────────────────────────────────

    @staticmethod
    def _csrank(labels: np.ndarray) -> np.ndarray:
        """Cross-sectional rank normalization to [0, 1]."""
        ranks = rankdata(labels, method="ordinal")
        return (ranks - 1) / max(len(ranks) - 1, 1)

    @staticmethod
    def _compute_ch(features: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz index. Returns 0 if computation fails."""
        try:
            unique = np.unique(labels)
            if len(unique) < 2:
                return 0.0
            return calinski_harabasz_score(features, labels)
        except Exception:
            return 0.0

    # ── Persistence ────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "kmeans": self.kmeans,
            "regressors": self.regressors,
            "outlier_count": self.outlier_count,
            "ch_score": self._ch_score,
            "initialized": self._initialized,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        log.info(f"ESGD model saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.kmeans = state["kmeans"]
        self.regressors = state["regressors"]
        self.outlier_count = state["outlier_count"]
        self._ch_score = state["ch_score"]
        self._initialized = state["initialized"]
        log.info(f"ESGD model loaded from {path}")
