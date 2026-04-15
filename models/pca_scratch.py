import numpy as np


class PCAScratch:
    """
    Principal Component Analysis implemented from scratch using NumPy only.

    This fulfills the course requirement: 'At least one algorithm from the course must
    be implemented without only being a wrapper around a library.'

    Algorithm (explainable at demo):
    - Center the data: X_c = X - mean(X, axis=0)
    - Compute the Thin SVD: X_c = U · S · Vt
        - U  (n × k): left singular vectors
        - S  (k,):    singular values (non-negative, descending)
        - Vt (k × d): right singular vectors — these ARE the principal components
    - The covariance matrix of X_c is  C = X_c^T X_c / (n-1) = Vt^T · diag(S²/(n-1)) · Vt
      so the rows of Vt are the eigenvectors of C and S²/(n-1) are the eigenvalues.
    - Explained variance of PC_i = S_i² / (n - 1)
    - Projection of X onto the first k components: X_c @ Vt[:k].T

    Why SVD instead of explicit covariance eigendecomposition:
      Computing C = X_c^T X_c materializes a (d × d) matrix — expensive when d is large
      (our feature matrix has 50–80 columns). SVD operates on X_c directly and is
      numerically more stable when columns are nearly collinear (e.g., correlated
      one-hot cuisine features).

    Public API is a drop-in replacement for sklearn.decomposition.PCA:
        fit_transform(X)  → projected array (n, n_components)
        transform(X)      → projected array (n, n_components)
        .components_               (n_components, n_features) — principal axes
        .explained_variance_       (n_components,)
        .explained_variance_ratio_ (n_components,)
        .mean_                     (n_features,)

    Additional attributes (set externally by clustering.py, same as before):
        .axis_labels_          list[str] | None
        .component_summaries_  list[str] | None
    """

    def __init__(self, n_components: int = 3, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state  # unused; kept for API compatibility

        # Populated after fit_transform()
        self.components_: np.ndarray | None = None               # (k, d)
        self.explained_variance_: np.ndarray | None = None       # (k,)
        self.explained_variance_ratio_: np.ndarray | None = None # (k,)
        self.mean_: np.ndarray | None = None                     # (d,)

        # Set externally by clustering.py for axis annotation in the Streamlit UI
        self.axis_labels_: list[str] | None = None
        self.component_summaries_: list[str] | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA to X and return the projected coordinates.

        Steps:
          1. Center X by subtracting the column means.
          2. Compute Thin SVD of the centered matrix.
          3. Extract principal components (rows of Vt) and explained variance.
          4. Project: X_c @ Vt[:k].T

        Returns array of shape (n_samples, n_components).
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        # Step 1: center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Step 2: thin SVD — Vt rows are the principal directions
        # full_matrices=False: U is (n, min(n,d)), S is (min(n,d),), Vt is (min(n,d), d)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Step 3: eigenvalues of the covariance matrix = S² / (n - 1)
        explained_variance_full = (S ** 2) / max(n - 1, 1)
        total_variance = explained_variance_full.sum()

        k = min(self.n_components, len(S))
        self.components_ = Vt[:k]                                        # (k, d)
        self.explained_variance_ = explained_variance_full[:k]           # (k,)
        self.explained_variance_ratio_ = (                               # (k,)
            self.explained_variance_ / total_variance if total_variance > 0
            else np.zeros(k)
        )

        # Step 4: project onto the top-k principal components
        return Xc @ self.components_.T  # (n, k)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data onto the already-fitted principal components."""
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T
