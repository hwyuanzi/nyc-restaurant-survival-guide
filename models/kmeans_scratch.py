import numpy as np


class KMeansScratch:
    """
    K-Means clustering implemented from scratch using NumPy only (no scikit-learn).

    This fulfills the course requirement: 'At least one algorithm from the course must
    be implemented without only being a wrapper around a library.'

    Design choices (explainable at demo):
    - Initialization: K-Means++ — seeds centroids proportional to D(x)², the squared
      distance from x to its nearest already-chosen centroid. Gives an O(log k)
      approximation guarantee and dramatically reduces the chance of dead clusters
      vs. uniform random initialization.
    - Distance metric: Euclidean (L2). Appropriate here because the feature matrix is
      StandardScaler-normalized — all dimensions are on the same scale, so L2 is
      well-defined and comparable across features.
    - E-step: vectorized via ||x - c||² = ||x||² + ||c||² - 2·x·c to avoid an
      explicit (n, k, d) tensor, keeping memory O(n·k).
    - M-step: recompute centroids as the mean of assigned points; reinitialize any
      empty cluster centroid to a random data point to avoid degenerate solutions.
    - Convergence: stop when the maximum centroid shift falls below `tol` OR when
      label assignments are unchanged between iterations.
    - Multi-run: repeat `n_init` times with independent random seeds; keep the run
      with the lowest inertia (sum of squared distances to assigned centroids).

    Public API is a drop-in replacement for sklearn.cluster.KMeans:
        fit_predict(X)  → labels array
        predict(X)      → labels array
        transform(X)    → (n_samples, n_clusters) distance matrix
        .cluster_centers_  .inertia_  .labels_  .n_iter_
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        # Populated after fit_predict()
        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.silhouette_score_: float | None = None  # set externally by run_kmeans

    # ── K-Means++ initialization ───────────────────────────────────────────────
    def _init_centroids_pp(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        K-Means++ seeding:
          1. Pick the first centroid uniformly at random.
          2. For each subsequent centroid, sample a point with probability
             proportional to its squared distance to the nearest existing centroid.
        This biases initial centroids to be spread out, improving convergence.
        """
        n = X.shape[0]
        first_idx = rng.randint(0, n)
        centroids = [X[first_idx].copy()]

        for _ in range(1, self.n_clusters):
            # D²(x) = min squared distance from x to any centroid chosen so far
            current_centroids = np.asarray(centroids, dtype=X.dtype)
            diff = X[:, None, :] - current_centroids[None, :, :]
            D2 = np.sum(diff * diff, axis=2).min(axis=1)
            # Sample next centroid with probability proportional to D²
            total = float(D2.sum())
            probs = D2 / total if total > 0 else np.full(n, 1.0 / n, dtype=np.float64)
            cumprobs = np.cumsum(probs)
            r = rng.random()
            idx = int(np.searchsorted(cumprobs, r))
            centroids.append(X[min(idx, n - 1)].copy())

        return np.array(centroids)  # shape: (k, d)

    # ── Vectorized assignment (E-step) ─────────────────────────────────────────
    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each point to its nearest centroid using the identity:
            ||x - c||² = ||x||² + ||c||² - 2·(x · c)

        Avoids materializing an (n, k, d) tensor; memory is O(n·k).
        Returns label array of shape (n,).
        """
        X_sq = np.einsum("ij,ij->i", X, X)[:, None]         # (n, 1)
        C_sq = np.einsum("ij,ij->i", centroids, centroids)[None, :]  # (1, k)
        cross = X @ centroids.T                              # (n, k)
        dist2 = np.maximum(X_sq + C_sq - 2.0 * cross, 0.0)  # clip numerical noise
        return np.argmin(dist2, axis=1)

    # ── Single K-Means run ─────────────────────────────────────────────────────
    def _run(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        One complete K-Means run from K-Means++ initialization to convergence.
        Returns (labels, centroids, inertia, n_iterations).
        """
        n = X.shape[0]
        centroids = self._init_centroids_pp(X, rng)
        labels = np.full(n, -1, dtype=int)

        for iteration in range(self.max_iter):
            # E-step: assign points to nearest centroid
            new_labels = self._assign(X, centroids)

            # Convergence check: labels unchanged
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

            # M-step: recompute centroids as cluster means
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialize to a random data point
                    new_centroids[k] = X[rng.randint(0, n)]

            # Convergence check: centroid shift
            max_shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            if max_shift < self.tol:
                break

        # Inertia = sum of squared distances from each point to its centroid
        inertia = sum(
            float(np.sum((X[labels == k] - centroids[k]) ** 2))
            for k in range(self.n_clusters)
            if (labels == k).any()
        )
        return labels, centroids, inertia, iteration + 1

    # ── Public API (sklearn-compatible) ───────────────────────────────────────

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means to X and return cluster label for each sample.
        Runs `n_init` times; keeps the solution with the lowest inertia.
        """
        X = np.asarray(X, dtype=np.float64)
        master_rng = np.random.RandomState(self.random_state)

        best_labels, best_centroids, best_inertia, best_iters = None, None, np.inf, 0

        for _ in range(self.n_init):
            seed = int(master_rng.randint(0, 2**31))
            labels, centroids, inertia, n_iter = self._run(
                X, np.random.RandomState(seed)
            )
            if inertia < best_inertia:
                best_labels = labels
                best_centroids = centroids
                best_inertia = inertia
                best_iters = n_iter

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia
        self.n_iter_ = best_iters
        return self.labels_.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to their nearest centroid."""
        return self._assign(np.asarray(X, dtype=np.float64), self.cluster_centers_)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return the (n_samples, n_clusters) matrix of Euclidean distances
        from each point to each centroid. Same contract as sklearn.transform().
        """
        X = np.asarray(X, dtype=np.float64)
        return np.stack(
            [np.linalg.norm(X - c, axis=1) for c in self.cluster_centers_],
            axis=1,
        )
