import os
import time
import json
import hashlib
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "data/cluster_cache.parquet"
MODEL_PATH = "data/kmeans_model.joblib"
# Per-algorithm cache paths (KMeans uses the legacy paths above to avoid
# breaking any on-disk cache already written by older code).
ALGO_CACHE_PATHS = {
    "kmeans": ("data/cluster_cache.parquet", "data/kmeans_model.joblib"),
    "gmm": ("data/cluster_cache_gmm.parquet", "data/cluster_model_gmm.joblib"),
    "agglomerative": ("data/cluster_cache_agglo.parquet", "data/cluster_model_agglo.joblib"),
}
CACHE_TTL  = 86400  # 24 hours
CLUSTER_SCHEMA_VERSION = 22  # Added multi-algorithm clustering + cache signature updates
TSNE_MAX_ROWS = 3500

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

REQUIRED_COLUMNS = ["restaurant_id", "name", "lat", "lng", "cuisine_type", "price_tier", "avg_rating", "review_count"]

# Top cuisines to one-hot encode (rest → "Other"); chosen for interpretability
TOP_CUISINES = [
    "American", "Chinese", "Italian", "Mexican", "Japanese",
    "Pizza", "Coffee/Tea", "Donuts", "Hamburgers", "Chicken",
]

# Human-readable labels for every clustering feature
FEATURE_LABELS = {
    "price_norm": "Price level",
    "rating_norm": "Google rating",
    "review_norm": "Review volume",
    "health_norm": "Health grade",
    "lat_norm": "North–south location",
    "lng_norm": "East–west location",
    "user_affinity": "User affinity",
}
# Dynamically add cuisine and borough labels
for _c in TOP_CUISINES:
    FEATURE_LABELS[f"cuisine_{_c}"] = f"Cuisine: {_c}"
FEATURE_LABELS["cuisine_Other"] = "Cuisine: Other"
for _b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]:
    FEATURE_LABELS[f"boro_{_b}"] = f"Borough: {_b}"


def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"restaurants dataset is missing required columns: {missing}")


def _price_descriptor(price_value):
    if pd.isna(price_value):
        return "Mixed Price"
    if price_value <= 1.5:
        return "Budget"
    if price_value <= 2.5:
        return "Mid-Range"
    return "Upscale"


def _rating_descriptor(rating_value):
    if pd.isna(rating_value):
        return "Mixed Quality"
    if rating_value >= 4.5:
        return "Top Rated"
    if rating_value >= 4.1:
        return "Well Rated"
    if rating_value >= 3.7:
        return "Reliable"
    return "Emerging"


def _label_looks_internal(label):
    text = str(label or "").strip()
    return not text or bool(re.match(r"^(cluster\s*\d+|\d+\b)", text, flags=re.IGNORECASE))


def _assign_cluster_labels(df: pd.DataFrame):
    """Generate unique, human-readable labels directly from interpretable features."""
    label_map = {}
    used_labels = set()

    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]

        # Dominant cuisine
        cuisine_counts = cluster_df["cuisine_type"].fillna("").astype(str).value_counts()
        top_cuisine = cuisine_counts.index[0] if not cuisine_counts.empty and cuisine_counts.index[0] else "Mixed"

        # Price descriptor
        avg_price = pd.to_numeric(cluster_df["price_tier"], errors="coerce").mean()
        price_desc = _price_descriptor(avg_price)

        # Rating descriptor
        avg_rating = pd.to_numeric(cluster_df["avg_rating"], errors="coerce").mean()
        rating_desc = _rating_descriptor(avg_rating)

        # Dominant borough
        boro_counts = cluster_df["boro"].fillna("").astype(str).value_counts()
        top_boro = boro_counts.index[0] if not boro_counts.empty and boro_counts.index[0] not in ("", "0") else ""

        # Build candidate labels in priority order
        candidates = []
        if top_cuisine != "Mixed":
            candidates.append(f"{price_desc} {top_cuisine} · {rating_desc}")
            if top_boro:
                candidates.append(f"{top_cuisine} in {top_boro} · {rating_desc}")
            candidates.append(f"{top_cuisine} · {price_desc}")
        if top_boro:
            candidates.append(f"{price_desc} {rating_desc} · {top_boro}")
        candidates.append(f"{price_desc} {rating_desc}")
        candidates.append(f"Cluster {cluster_id}")

        chosen = None
        for c in candidates:
            if c not in used_labels and not _label_looks_internal(c):
                chosen = c
                break
        if chosen is None:
            chosen = f"Group {cluster_id}"
        label_map[cluster_id] = chosen
        used_labels.add(chosen)

    return df["cluster_id"].map(label_map)


def _feature_category(feature_name):
    if feature_name.startswith("cuisine_"):
        return "Cuisine"
    if feature_name.startswith("boro_"):
        return "Borough"
    if feature_name in {"price_norm"}:
        return "Price"
    if feature_name in {"rating_norm", "review_norm", "health_norm"}:
        return "Quality"
    if feature_name in {"lat_norm", "lng_norm"}:
        return "Location"
    if feature_name == "user_affinity":
        return "Affinity"
    return "Other"


def _humanize_feature(feature_name):
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("cuisine_"):
        return feature_name.replace("cuisine_", "").replace("_", " ").title()
    if feature_name.startswith("boro_"):
        return feature_name.replace("boro_", "")
    return feature_name.replace("_", " ").title()


def _component_axis_label(component, feature_columns):
    """Generate a human-readable PCA axis label from the top-2 feature loadings."""
    weights = pd.Series(np.abs(component), index=feature_columns)
    category_weights = weights.groupby(
        [_feature_category(name) for name in feature_columns]
    ).sum().sort_values(ascending=False)
    primary = category_weights.index[0] if not category_weights.empty else "Pattern"
    secondary = category_weights.index[1] if len(category_weights) > 1 else None

    category_display = {
        "Cuisine": "Cuisine Type",
        "Borough": "Geographic Area",
        "Price": "Price Level",
        "Quality": "Quality & Popularity",
        "Location": "Geographic Location",
        "Affinity": "User Match",
        "Other": "Mixed Factors",
    }
    p_label = category_display.get(primary, primary)
    s_label = category_display.get(secondary, "") if secondary else ""

    if secondary and secondary != primary:
        return f"{p_label} vs {s_label}"
    return p_label


def _component_summary(component, feature_columns):
    loading_series = pd.Series(component, index=feature_columns)
    top_positive = loading_series.sort_values(ascending=False).head(2)
    top_negative = loading_series.sort_values(ascending=True).head(2)

    positive_text = ", ".join(_humanize_feature(name) for name in top_positive.index if top_positive[name] > 0)
    negative_text = ", ".join(_humanize_feature(name) for name in top_negative.index if top_negative[name] < 0)

    if positive_text and negative_text:
        return f"Leans toward {positive_text} over {negative_text}"
    if positive_text:
        return f"Mostly driven by {positive_text}"
    if negative_text:
        return f"Mostly separates against {negative_text}"
    return "Mixed restaurant attributes"


def build_feature_matrix(df: pd.DataFrame):
    """Build a fully interpretable feature matrix for clustering.

    Every feature is human-readable: price, rating, review volume, health score,
    cuisine one-hot, borough one-hot, and normalized lat/lng.
    """
    validate_dataframe(df)
    df = df.copy()

    # --- Numerical features (all normalized to 0-1) ---

    # 1. Price tier (1-4 → 0-1)
    price_norm = ((df["price_tier"].fillna(2) - 1) / 3.0).values.reshape(-1, 1)

    # 2. Rating (1-5 → 0-1)
    rating_norm = ((df["avg_rating"].fillna(3.0) - 1.0) / 4.0).values.reshape(-1, 1)

    # 3. Review count (log-scaled then min-max → 0-1)
    log_reviews = np.log1p(df["review_count"].fillna(0).values).reshape(-1, 1)
    log_min, log_max = log_reviews.min(), log_reviews.max()
    review_norm = (log_reviews - log_min) / (log_max - log_min) if log_max > log_min else np.zeros_like(log_reviews)

    # 4. Health score (0-42, lower is better → inverted to 0-1 where 1 = best)
    health_series = df["score"] if "score" in df.columns else pd.Series([21] * len(df), index=df.index)
    health_source = pd.to_numeric(health_series, errors="coerce").fillna(21).clip(0, 42).values.reshape(-1, 1)
    health_norm = 1 - (health_source / 42.0)

    # 5. Latitude / longitude (min-max normalized)
    lat_vals = pd.to_numeric(df["lat"], errors="coerce").fillna(df["lat"].median()).values.reshape(-1, 1)
    lng_vals = pd.to_numeric(df["lng"], errors="coerce").fillna(df["lng"].median()).values.reshape(-1, 1)
    lat_min, lat_max = lat_vals.min(), lat_vals.max()
    lng_min, lng_max = lng_vals.min(), lng_vals.max()
    lat_norm = (lat_vals - lat_min) / (lat_max - lat_min) if lat_max > lat_min else np.zeros_like(lat_vals)
    lng_norm = (lng_vals - lng_min) / (lng_max - lng_min) if lng_max > lng_min else np.zeros_like(lng_vals)

    # --- Categorical features (one-hot) ---

    # 6. Cuisine one-hot (top 10 + Other)
    cuisine_series = df["cuisine_type"].fillna("Other").astype(str)
    cuisine_group = cuisine_series.where(cuisine_series.isin(TOP_CUISINES), other="Other")
    cuisine_dummies = pd.get_dummies(cuisine_group, prefix="cuisine").reindex(
        columns=[f"cuisine_{c}" for c in TOP_CUISINES] + ["cuisine_Other"],
        fill_value=0,
    ).values.astype(np.float32)

    # 7. Borough one-hot
    boro_series = df["boro"].fillna("Unknown").astype(str)
    boro_series = boro_series.where(boro_series.isin(["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]), other="Unknown")
    boro_dummies = pd.get_dummies(boro_series, prefix="boro").reindex(
        columns=[f"boro_{b}" for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]],
        fill_value=0,
    ).values.astype(np.float32)

    # --- Assemble with feature weights ---
    # Rebalanced so cuisine one-hots no longer dominate: price/rating/location
    # carry more signal, cuisine is softened to avoid a catch-all "Other" mega-cluster.
    X = np.hstack([
        price_norm * 1.5,
        rating_norm * 1.3,
        review_norm * 0.7,
        health_norm * 0.5,
        lat_norm * 0.8,
        lng_norm * 0.8,
        cuisine_dummies * 0.8,
        boro_dummies * 0.8,
    ]).astype(np.float32)

    feature_columns = (
        ["price_norm", "rating_norm", "review_norm", "health_norm", "lat_norm", "lng_norm"]
        + [f"cuisine_{c}" for c in TOP_CUISINES] + ["cuisine_Other"]
        + [f"boro_{b}" for b in ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]]
    )

    return X, feature_columns, df


def apply_user_weights(X: np.ndarray, df: pd.DataFrame, user_history: dict):
    visited_ids = user_history.get("visited_ids", [])
    rated       = user_history.get("rated", {})

    if not visited_ids:
        affinity = np.zeros(len(df))
        return np.hstack([X, affinity.reshape(-1, 1)])

    visited_mask = df["restaurant_id"].isin(visited_ids)
    visited_X    = X[visited_mask.values]

    if len(visited_X) == 0:
        affinity = np.zeros(len(df))
        return np.hstack([X, affinity.reshape(-1, 1)])

    # Weighted mean of visited restaurants by rating
    weights = []
    for rid in df.loc[visited_mask, "restaurant_id"]:
        weights.append(rated.get(rid, 3.0) / 5.0)
    weights = np.array(weights).reshape(-1, 1)
    user_vec = (visited_X * weights).sum(axis=0, keepdims=True) / (weights.sum() + 1e-8)

    affinity = cosine_similarity(X, user_vec).flatten()
    return np.hstack([X, affinity.reshape(-1, 1)])


def prepare_clustering_space(X_aug: np.ndarray, scaler: StandardScaler | None = None, fit: bool = True):
    if fit or scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_aug)
    else:
        X_scaled = scaler.transform(X_aug)

    reducer = getattr(scaler, "cluster_reducer_", None)
    if fit:
        reducer = None
        max_reducer_components = min(
            24,
            X_scaled.shape[1] - 1,
            X_scaled.shape[0] - 1,
        )
        if max_reducer_components >= 3:
            probe = PCA(n_components=max_reducer_components, random_state=42)
            probe.fit(X_scaled)
            explained = np.cumsum(probe.explained_variance_ratio_)
            target_components = int(np.searchsorted(explained, 0.92) + 1)
            n_components = int(np.clip(target_components, 3, max_reducer_components))
            reducer = PCA(n_components=n_components, random_state=42)
            reducer.fit(X_scaled)
        scaler.cluster_reducer_ = reducer

    X_cluster = reducer.transform(X_scaled) if reducer is not None else X_scaled
    return X_scaled, X_cluster, scaler


def _cluster_signature(df: pd.DataFrame, user_history: dict, k: int,
                       algorithm: str = "kmeans"):
    history_payload = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "algorithm": algorithm,
        "visited_ids": sorted(str(value) for value in user_history.get("visited_ids", [])),
        "rated": {str(key): float(value) for key, value in sorted(user_history.get("rated", {}).items())},
        "cuisine_preferences": sorted(str(value) for value in user_history.get("cuisine_preferences", [])),
        "price_preference": int(user_history.get("price_preference", 2)),
        "neighborhood_preference": sorted(str(value) for value in user_history.get("neighborhood_preference", [])),
        "k": int(k),
        "row_count": int(len(df)),
        "restaurant_sample": sorted(df["restaurant_id"].astype(str).head(25).tolist()),
    }
    payload = json.dumps(history_payload, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_optimal_k(X_scaled: np.ndarray, k_range=range(4, 16)) -> int:
    best_k, best_score = 8, -1
    for k in k_range:
        if k >= len(X_scaled):
            break
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=100, random_state=42)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)), random_state=42)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _prepare_cluster_inputs(df: pd.DataFrame, user_history: dict):
    X, feature_columns, df = build_feature_matrix(df)
    X_aug = apply_user_weights(X, df, user_history)
    projection_feature_columns = feature_columns + ["user_affinity"]
    X_scaled, X_cluster, scaler = prepare_clustering_space(X_aug, fit=True)
    return df, X_aug, X_scaled, X_cluster, scaler, projection_feature_columns


def _compute_cluster_centroids(X_cluster: np.ndarray, labels: np.ndarray):
    cluster_ids = sorted(int(cid) for cid in np.unique(labels))
    centroids = np.vstack([
        X_cluster[labels == cid].mean(axis=0)
        for cid in cluster_ids
    ]).astype(np.float32)
    return cluster_ids, centroids


def _merge_small_clusters(labels: np.ndarray, X_cluster: np.ndarray, min_cluster_size: int):
    labels = labels.astype(int).copy()
    while True:
        cluster_sizes = pd.Series(labels).value_counts()
        small_cluster_ids = cluster_sizes[cluster_sizes < min_cluster_size].index.tolist()
        large_cluster_ids = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        if not small_cluster_ids or not large_cluster_ids:
            break

        smallest_cid = int(cluster_sizes.idxmin())
        cluster_ids, centroids = _compute_cluster_centroids(X_cluster, labels)
        id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
        smallest_vec = centroids[id_to_idx[smallest_cid]]

        best_target = None
        best_dist = np.inf
        for cid in large_cluster_ids:
            cid = int(cid)
            if cid == smallest_cid:
                continue
            dist = np.linalg.norm(centroids[id_to_idx[cid]] - smallest_vec)
            if dist < best_dist:
                best_dist = dist
                best_target = cid
        if best_target is None:
            break
        labels[labels == smallest_cid] = int(best_target)

    unique_ids = sorted(np.unique(labels))
    id_remap = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    return np.array([id_remap[int(cid)] for cid in labels], dtype=int)


def _to_three_dims(matrix: np.ndarray):
    if matrix.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    n_components = min(3, matrix.shape[0], matrix.shape[1])
    if n_components <= 0:
        return np.zeros((matrix.shape[0], 3), dtype=np.float32), None
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(matrix)
    if n_components < 3:
        padded = np.zeros((matrix.shape[0], 3), dtype=np.float32)
        padded[:, :n_components] = projected
        projected = padded
    return projected, pca


def _finalize_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    X_cluster: np.ndarray,
    X_scaled: np.ndarray,
    X_aug: np.ndarray,
    projection_feature_columns: list[str],
):
    df = df.copy()
    labels = labels.astype(int)

    min_cluster_size = max(1, int(round(len(df) * 0.03)))
    labels = _merge_small_clusters(labels, X_cluster, min_cluster_size=min_cluster_size)
    df["cluster_id"] = labels

    cluster_ids, centroids = _compute_cluster_centroids(X_cluster, labels)

    # PCA coordinates for global structure.
    X_pca, pca = _to_three_dims(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
    df["pca_z"] = X_pca[:, 2]

    axis_labels = ["Principal Component 1", "Principal Component 2", "Principal Component 3"]
    component_summaries = ["", "", ""]
    if pca is not None and hasattr(pca, "components_"):
        dynamic_labels = [
            _component_axis_label(component, projection_feature_columns)
            for component in pca.components_
        ]
        dynamic_summaries = [
            _component_summary(component, projection_feature_columns)
            for component in pca.components_
        ]
        axis_labels[:len(dynamic_labels)] = dynamic_labels
        component_summaries[:len(dynamic_summaries)] = dynamic_summaries
    if pca is not None:
        pca.axis_labels_ = axis_labels
        pca.component_summaries_ = component_summaries

    # Cluster-oriented view from distances-to-centroids.
    centroid_distances = np.linalg.norm(
        X_cluster[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    if centroid_distances.shape[1] > 1:
        centroid_distances = StandardScaler().fit_transform(centroid_distances)
    X_cluster_view, _ = _to_three_dims(centroid_distances)
    df["cluster_view_x"] = X_cluster_view[:, 0]
    df["cluster_view_y"] = X_cluster_view[:, 1]
    df["cluster_view_z"] = X_cluster_view[:, 2]

    # t-SNE coordinates for local neighborhoods.
    if len(df) <= TSNE_MAX_ROWS and len(df) > 5:
        tsne_perplexity = int(np.clip(max(5, len(df) // 70), 5, min(45, len(df) - 1)))
        X_tsne = TSNE(
            n_components=3,
            perplexity=tsne_perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=42,
        ).fit_transform(X_cluster)
        df["tsne_x"] = X_tsne[:, 0]
        df["tsne_y"] = X_tsne[:, 1]
        df["tsne_z"] = X_tsne[:, 2]
    else:
        df["tsne_x"] = np.nan
        df["tsne_y"] = np.nan
        df["tsne_z"] = np.nan

    df["cluster_label"] = _assign_cluster_labels(df)
    df["user_affinity_score"] = X_aug[:, -1]

    if UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            X_umap = reducer.fit_transform(X_scaled)
            df["umap_x"] = X_umap[:, 0]
            df["umap_y"] = X_umap[:, 1]
            df["umap_z"] = X_umap[:, 2]
        except Exception:
            pass

    silhouette = -1.0
    if len(cluster_ids) > 1:
        try:
            silhouette = float(
                silhouette_score(
                    X_cluster,
                    labels,
                    sample_size=min(1500, len(X_cluster)),
                    random_state=42,
                )
            )
        except Exception:
            silhouette = -1.0

    return df, pca, centroids, silhouette


def run_kmeans(df: pd.DataFrame, user_history: dict, k: int = 10):
    df, X_aug, X_scaled, X_cluster, scaler, projection_feature_columns = _prepare_cluster_inputs(df, user_history)

    # Use MiniBatchKMeans for large datasets
    KMeansClass = MiniBatchKMeans if len(df) > 10000 else KMeans
    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    df = df.copy()
    best_model = None
    best_labels = None
    best_score = -1
    candidate_seeds = [42, 52, 62, 72, 82]
    for seed in candidate_seeds:
        candidate_model = KMeansClass(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=seed,
        )
        candidate_labels = candidate_model.fit_predict(X_cluster)
        if len(set(candidate_labels)) < 2:
            continue
        try:
            candidate_score = silhouette_score(
                X_cluster,
                candidate_labels,
                sample_size=min(1500, len(X_cluster)),
                random_state=seed,
            )
        except Exception:
            candidate_score = -1
        if candidate_score > best_score:
            best_score = candidate_score
            best_model = candidate_model
            best_labels = candidate_labels

    model = best_model or KMeansClass(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    ).fit(X_cluster)
    labels = best_labels if best_labels is not None else model.labels_
    result_df, pca, centroids, sil = _finalize_clusters(
        df, labels, X_cluster, X_scaled, X_aug, projection_feature_columns
    )
    model.cluster_centroids_ = centroids
    model.silhouette_score_ = sil
    model.algorithm_ = "kmeans"
    return result_df, model, scaler, pca


def run_gmm(df: pd.DataFrame, user_history: dict, k: int = 10):
    df, X_aug, X_scaled, X_cluster, scaler, projection_feature_columns = _prepare_cluster_inputs(df, user_history)

    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="tied",
        random_state=42,
        n_init=5,
    )
    gmm.fit(X_cluster)
    labels = gmm.predict(X_cluster)

    result_df, pca, centroids, sil = _finalize_clusters(
        df, labels, X_cluster, X_scaled, X_aug, projection_feature_columns
    )
    gmm.cluster_centroids_ = centroids
    gmm.silhouette_score_ = sil
    gmm.algorithm_ = "gmm"
    return result_df, gmm, scaler, pca


def run_agglomerative(df: pd.DataFrame, user_history: dict, k: int = 10):
    df, X_aug, X_scaled, X_cluster, scaler, projection_feature_columns = _prepare_cluster_inputs(df, user_history)

# ---------------------------------------------------------------------------
# Alternate clustering algorithms (GMM and Hierarchical / Ward)
# Both reuse ``build_feature_matrix`` / ``apply_user_weights`` so they operate
# in the same 22-dim interpretable space as K-Means.
# ---------------------------------------------------------------------------


def _merge_small_clusters(df: pd.DataFrame, centroids: np.ndarray,
                           min_fraction: float = 0.03):
    """Merge clusters smaller than ``min_fraction`` of the total into their
    nearest large cluster by centroid Euclidean distance.  Mutates ``df``.
    """
    min_cluster_size = max(1, int(round(len(df) * min_fraction)))
    while True:
        cluster_sizes = df["cluster_id"].value_counts()
        small_cluster_ids = cluster_sizes[cluster_sizes < min_cluster_size].index.tolist()
        large_cluster_ids = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        if not small_cluster_ids or not large_cluster_ids:
            break
        smallest_cid = cluster_sizes.idxmin()
        dists = np.linalg.norm(centroids - centroids[smallest_cid], axis=1)
        dists[smallest_cid] = np.inf
        for cid in range(len(centroids)):
            if cid not in large_cluster_ids:
                dists[cid] = np.inf
        target_cid = int(np.argmin(dists))
        df.loc[df["cluster_id"] == smallest_cid, "cluster_id"] = target_cid


def _finalize_clusters(df: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray,
                        X_scaled: np.ndarray, X_cluster: np.ndarray,
                        X_aug: np.ndarray, projection_feature_columns: list):
    """Attach cluster_id + PCA / t-SNE / UMAP / cluster-view projections.

    Shared post-processing for non-KMeans algorithms that don't have a
    native ``transform()``.  The centroid-to-point distance matrix is
    computed via broadcasting.
    """
    df = df.copy()
    df["cluster_id"] = labels

    _merge_small_clusters(df, centroids)

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
    df["pca_z"] = X_pca[:, 2]
    pca.axis_labels_ = [
        _component_axis_label(component, projection_feature_columns)
        for component in pca.components_[:3]
    ]
    pca.component_summaries_ = [
        _component_summary(component, projection_feature_columns)
        for component in pca.components_[:3]
    ]

    # Distance from each point to each centroid (analogue of kmeans.transform).
    centroid_distances = np.linalg.norm(
        X_cluster[:, None, :] - centroids[None, :, :], axis=2
    )
    cluster_view_pca = PCA(n_components=3, random_state=42)
    X_cluster_view = cluster_view_pca.fit_transform(
        StandardScaler().fit_transform(centroid_distances)
    )
    df["cluster_view_x"] = X_cluster_view[:, 0]
    df["cluster_view_y"] = X_cluster_view[:, 1]
    df["cluster_view_z"] = X_cluster_view[:, 2]

    if len(df) <= TSNE_MAX_ROWS:
        tsne_perplexity = int(np.clip(max(5, len(df) // 70), 5, 45))
        X_tsne = TSNE(
            n_components=3,
            perplexity=tsne_perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=1000,
            random_state=42,
        ).fit_transform(X_cluster)
        df["tsne_x"] = X_tsne[:, 0]
        df["tsne_y"] = X_tsne[:, 1]
        df["tsne_z"] = X_tsne[:, 2]
    else:
        df["tsne_x"] = np.nan
        df["tsne_y"] = np.nan
        df["tsne_z"] = np.nan

    df["cluster_label"] = _assign_cluster_labels(df)
    df["user_affinity_score"] = X_aug[:, -1]

    if UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                                metric="cosine", random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            df["umap_x"] = X_umap[:, 0]
            df["umap_y"] = X_umap[:, 1]
            df["umap_z"] = X_umap[:, 2]
        except Exception:
            pass

    return df, pca


class _CentroidClusteringModel:
    """Lightweight wrapper so GMM / Agglomerative results expose the same
    ``cluster_centers_`` / ``transform`` / ``predict`` surface that downstream
    code expects from a K-Means model."""

    def __init__(self, centroids: np.ndarray, labels: np.ndarray, algorithm: str,
                 base_model=None):
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.n_clusters = int(centroids.shape[0])
        self.algorithm = algorithm
        self.base_model_ = base_model  # the underlying sklearn object

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X).argmin(axis=1)


def run_gmm(df: pd.DataFrame, user_history: dict, k: int = 10):
    """Gaussian Mixture clustering with tied covariance.

    Tied covariance shares one covariance matrix across components, which
    is a strong regulariser when feature dims (22) outnumber typical cluster
    sizes and prevents the catch-all ``k=1`` collapse that full covariance
    sometimes produces on sparse one-hot features.
    """
    X, feature_columns, df = build_feature_matrix(df)
    X_aug = apply_user_weights(X, df, user_history)
    projection_feature_columns = feature_columns + ["user_affinity"]

    X_scaled, X_cluster, scaler = prepare_clustering_space(X_aug, fit=True)

    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    gmm = GaussianMixture(
        n_components=k,
        covariance_type="tied",
        random_state=42,
        n_init=5,
        max_iter=200,
        reg_covar=1e-4,
    )
    labels = gmm.fit_predict(X_cluster)
    centroids = gmm.means_.astype(np.float32)

    wrapper = _CentroidClusteringModel(centroids, labels, "gmm", base_model=gmm)
    df, pca = _finalize_clusters(df, labels, centroids, X_scaled, X_cluster,
                                  X_aug, projection_feature_columns)
    return df, wrapper, scaler, pca


def run_agglomerative(df: pd.DataFrame, user_history: dict, k: int = 10):
    """Agglomerative (Ward-linkage) hierarchical clustering.

    Ward minimises within-cluster variance at each merge, producing
    compact roughly-spherical clusters without the k-means init-sensitivity.
    Centroids are computed post-hoc as per-cluster means.
    """
    X, feature_columns, df = build_feature_matrix(df)
    X_aug = apply_user_weights(X, df, user_history)
    projection_feature_columns = feature_columns + ["user_affinity"]

    X_scaled, X_cluster, scaler = prepare_clustering_space(X_aug, fit=True)

    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X_cluster)

    unique_labels = np.unique(labels)
    centroids = np.vstack([
        X_cluster[labels == lbl].mean(axis=0) for lbl in unique_labels
    ]).astype(np.float32)
    # Remap labels to 0..K-1 in case unique_labels isn't already dense.
    remap = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)

    wrapper = _CentroidClusteringModel(centroids, labels, "agglomerative", base_model=agg)
    df, pca = _finalize_clusters(df, labels, centroids, X_scaled, X_cluster,
                                  X_aug, projection_feature_columns)
    return df, wrapper, scaler, pca


def run_clustering(df: pd.DataFrame, user_history: dict, k: int = 10,
                   algorithm: str = "kmeans"):
    """Dispatcher that runs the selected clustering algorithm.

    All algorithms return a tuple with the same shape —
    ``(df_with_labels_and_projections, model, scaler, pca)`` — so downstream
    UI code does not need to special-case the algorithm.
    """
    if algorithm == "kmeans":
        return run_kmeans(df, user_history, k)
    if algorithm == "gmm":
        return run_gmm(df, user_history, k)
    if algorithm == "agglomerative":
        return run_agglomerative(df, user_history, k)
    raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")


def compute_silhouette(df: pd.DataFrame, user_history: dict, algorithm: str,
                       k: int = 10) -> dict:
    """Run one algorithm and return a small summary dict for comparison UIs.

    Runs on a sample (max 1500 rows) for silhouette computation to keep the
    comparison fast regardless of dataset size.
    """
    result_df, model, _scaler, _pca = run_clustering(df, user_history, k, algorithm)
    X, _cols, _df_unused = build_feature_matrix(df)
    X_aug = apply_user_weights(X, df, user_history)
    X_scaled, X_cluster, _s = prepare_clustering_space(X_aug, fit=True)
    labels = result_df["cluster_id"].values
    try:
        score = silhouette_score(
            X_cluster, labels,
            sample_size=min(1500, len(X_cluster)),
            random_state=42,
        )
    except Exception:
        score = float("nan")
    top_labels = (
        result_df.groupby("cluster_id")["cluster_label"].first()
        .value_counts().head(3).index.tolist()
    )
    return {
        "algorithm": algorithm,
        "silhouette": float(score),
        "n_clusters": int(result_df["cluster_id"].nunique()),
        "top_labels": top_labels,
    }


# ---------------------------------------------------------------------------
# Cache helpers (per algorithm)
# ---------------------------------------------------------------------------


def _cache_paths_for(algorithm: str):
    return ALGO_CACHE_PATHS.get(algorithm, ALGO_CACHE_PATHS["kmeans"])


def cache_is_fresh(algorithm: str = "kmeans"):
    cache_path, _ = _cache_paths_for(algorithm)
    if not os.path.exists(cache_path):
        return False
    return (time.time() - os.path.getmtime(cache_path)) < CACHE_TTL

    ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = ward.fit_predict(X_cluster)

def load_cache(algorithm: str = "kmeans"):
    cache_path, model_path = _cache_paths_for(algorithm)
    df = pd.read_parquet(cache_path)
    artifacts = joblib.load(model_path)
    model = artifacts.get("model") or artifacts.get("kmeans")
    return df, model, artifacts["scaler"], artifacts["pca"], artifacts.get("signature")


def save_cache(df, model, scaler, pca, signature, algorithm: str = "kmeans"):
    cache_path, model_path = _cache_paths_for(algorithm)
    os.makedirs("data", exist_ok=True)
    df.to_parquet(cache_path, index=False)
    joblib.dump(
        {
            "model": model,
            "kmeans": model,  # legacy key for backward-compat with old loaders
            "scaler": scaler,
            "pca": pca,
            "signature": signature,
        },
        model_path,
    )


def get_clustered_data(df: pd.DataFrame, user_history: dict, k: int = 10,
                       force: bool = False, algorithm: str = "kmeans"):
    signature = _cluster_signature(df, user_history, k, algorithm)
    if not force and cache_is_fresh(algorithm):
        try:
            cached_df, cached_model, cached_scaler, cached_pca, cached_signature = load_cache(algorithm)
            cached_labels = cached_df.groupby("cluster_id")["cluster_label"].first()
            has_duplicate_labels = cached_labels.duplicated().any()
            has_internal_labels = cached_labels.map(_label_looks_internal).any()
            has_cluster_view = {"cluster_view_x", "cluster_view_y", "cluster_view_z"}.issubset(cached_df.columns)
            has_tsne_view = {"tsne_x", "tsne_y", "tsne_z"}.issubset(cached_df.columns)
            if cached_signature == signature and not has_duplicate_labels and not has_internal_labels and has_cluster_view and has_tsne_view:
                return cached_df, cached_model, cached_scaler, cached_pca
        except Exception:
            # Corrupt or schema-mismatched cache — fall through to recompute.
            pass
    result_df, model, scaler, pca = run_clustering(df, user_history, k, algorithm)
    save_cache(result_df, model, scaler, pca, signature, algorithm)
    return result_df, model, scaler, pca


# ---------------------------------------------------------------------------
# K-NN Recommendation Engine
# ---------------------------------------------------------------------------

BUDGET_TO_PRICE = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
BOROUGH_LIST = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def build_user_feature_vector(profile: dict, restaurant_df: pd.DataFrame) -> np.ndarray:
    """Construct a user feature vector in the same space as restaurant features.

    Uses the user's profile preferences (cuisine, borough, budget) blended
    with statistics from their liked restaurants to place them in the
    interpretable feature space used by clustering.

    Returns a 1-D numpy array with 22 elements matching the feature columns
    produced by ``build_feature_matrix()``.
    """
    liked_restaurants = profile.get("likes", [])
    liked_ids = {str(l.get("restaurant_id", "")) for l in liked_restaurants if l.get("restaurant_id")}

    # --- From profile preferences ---
    budget_str = profile.get("budget", "$$")
    price_pref = BUDGET_TO_PRICE.get(budget_str, 2)
    price_norm = (price_pref - 1) / 3.0

    fav_cuisines = profile.get("favorite_cuisines", [])
    pref_boroughs = profile.get("preferred_boroughs", [])

    # --- From liked restaurant history (if available) ---
    liked_mask = restaurant_df["restaurant_id"].astype(str).isin(liked_ids)
    liked_df = restaurant_df[liked_mask]

    if len(liked_df) > 0:
        avg_rating_norm = ((liked_df["avg_rating"].mean() - 1.0) / 4.0)
        log_reviews = np.log1p(liked_df["review_count"].values)
        log_min, log_max = np.log1p(restaurant_df["review_count"].fillna(0).values).min(), np.log1p(restaurant_df["review_count"].fillna(0).values).max()
        avg_review_norm = ((log_reviews.mean() - log_min) / (log_max - log_min)) if log_max > log_min else 0.5
        health_series = pd.to_numeric(liked_df["score"], errors="coerce").fillna(21).clip(0, 42)
        avg_health_norm = 1 - (health_series.mean() / 42.0)
        avg_lat = liked_df["lat"].mean()
        avg_lng = liked_df["lng"].mean()
    else:
        avg_rating_norm = 0.5
        avg_review_norm = 0.5
        avg_health_norm = 0.7
        avg_lat = restaurant_df["lat"].median()
        avg_lng = restaurant_df["lng"].median()

    # Normalize lat/lng using the same range as build_feature_matrix
    lat_all = pd.to_numeric(restaurant_df["lat"], errors="coerce").fillna(restaurant_df["lat"].median())
    lng_all = pd.to_numeric(restaurant_df["lng"], errors="coerce").fillna(restaurant_df["lng"].median())
    lat_min, lat_max = lat_all.min(), lat_all.max()
    lng_min, lng_max = lng_all.min(), lng_all.max()
    lat_norm = (avg_lat - lat_min) / (lat_max - lat_min) if lat_max > lat_min else 0.5
    lng_norm = (avg_lng - lng_min) / (lng_max - lng_min) if lng_max > lng_min else 0.5

    # --- Cuisine one-hot (from preferences + liked history) ---
    cuisine_vec = np.zeros(len(TOP_CUISINES) + 1, dtype=np.float32)  # +1 for Other
    # From explicit preferences
    for c in fav_cuisines:
        if c in TOP_CUISINES:
            cuisine_vec[TOP_CUISINES.index(c)] += 1.0
        else:
            cuisine_vec[-1] += 0.5  # Other
    # From liked restaurant cuisines
    if len(liked_df) > 0:
        for cuisine in liked_df["cuisine_type"].fillna("Other"):
            if cuisine in TOP_CUISINES:
                cuisine_vec[TOP_CUISINES.index(cuisine)] += 0.5
            else:
                cuisine_vec[-1] += 0.25
    # Normalize to sum to 1 if non-zero
    if cuisine_vec.sum() > 0:
        cuisine_vec = cuisine_vec / cuisine_vec.sum()

    # --- Borough one-hot (from preferences + liked history) ---
    boro_vec = np.zeros(len(BOROUGH_LIST), dtype=np.float32)
    for b in pref_boroughs:
        if b in BOROUGH_LIST:
            boro_vec[BOROUGH_LIST.index(b)] += 1.0
    if len(liked_df) > 0:
        for boro in liked_df["boro"].fillna(""):
            if boro in BOROUGH_LIST:
                boro_vec[BOROUGH_LIST.index(boro)] += 0.5
    if boro_vec.sum() > 0:
        boro_vec = boro_vec / boro_vec.sum()

    # --- Assemble (must match build_feature_matrix order & weights) ---
    user_vec = np.concatenate([
        [price_norm * 1.5],
        [avg_rating_norm * 1.3],
        [avg_review_norm * 0.7],
        [avg_health_norm * 0.5],
        [lat_norm * 0.8],
        [lng_norm * 0.8],
        cuisine_vec * 0.8,
        boro_vec * 0.8,
    ]).astype(np.float32)

    return user_vec


def recommend_knn(user_vector: np.ndarray,
                  restaurant_matrix: np.ndarray,
                  restaurant_df: pd.DataFrame,
                  visited_ids: set,
                  k: int = 15,
                  scaler: StandardScaler = None) -> pd.DataFrame:
    """Find the K nearest restaurants to the user vector using cosine similarity.

    Parameters
    ----------
    user_vector : 1-D array matching the feature columns of build_feature_matrix (22 dims)
    restaurant_matrix : 2-D array (N, 22) — raw output of build_feature_matrix
    restaurant_df : DataFrame with restaurant metadata
    visited_ids : set of restaurant_id strings to exclude
    k : number of recommendations to return
    scaler : optional StandardScaler for normalisation before similarity.
             Note: the scaler was fit on 23-dim vectors (22 features + user_affinity),
             so we append a placeholder affinity column before transforming.

    Returns
    -------
    DataFrame of top-K recommended restaurants with a ``knn_similarity`` column.
    """
    X_scaled = _scale_for_recommendations(restaurant_matrix, scaler=scaler)
    u_scaled = _scale_for_recommendations(np.asarray(user_vector).reshape(1, -1), scaler=scaler)

    similarities = cosine_similarity(u_scaled, X_scaled).flatten()

    # Mask visited restaurants
    visited_mask = restaurant_df["restaurant_id"].astype(str).isin(visited_ids)
    similarities[visited_mask.values] = -np.inf

    ranked = np.argsort(similarities)[::-1]
    top_indices = [idx for idx in ranked if np.isfinite(similarities[idx])][:k]
    if not top_indices:
        return restaurant_df.iloc[[]].copy().assign(knn_similarity=pd.Series(dtype=float))
    result = restaurant_df.iloc[top_indices].copy()
    result["knn_similarity"] = similarities[top_indices]
    return result.reset_index(drop=True)


def _scaled_space(vectors: np.ndarray, scaler: StandardScaler | None):
    """Apply the StandardScaler fit during clustering (which expects the
    23-dim [features + user_affinity] layout) to a 22-dim vector matrix.

    The recommender works on the 22-dim ``build_feature_matrix`` output but
    the scaler was fit on that plus a user-affinity column, so we append a
    neutral placeholder column before transforming.
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    n = vectors.shape[0]
    aug = np.hstack([vectors, np.full((n, 1), 0.5, dtype=np.float32)])
    if scaler is None:
        return aug
    return scaler.transform(aug)


def recommend_per_liked_knn(
    liked_vectors: np.ndarray,
    profile_vector: np.ndarray,
    restaurant_matrix: np.ndarray,
    restaurant_df: pd.DataFrame,
    visited_ids: set,
    liked_metadata: list | None = None,
    k_per_liked: int = 30,
    k_final: int = 50,
    scaler: StandardScaler | None = None,
    rrf_constant: int = 60,
) -> pd.DataFrame:
    """Recommend via per-liked-restaurant KNN fused by reciprocal rank.

    For each liked restaurant, rank all restaurants by cosine similarity.
    Combine the per-liked rankings via Reciprocal Rank Fusion:

        score[r] = sum over l of 1 / (rank_l(r) + rrf_constant)

    plus a small profile-similarity term so the results remain anchored
    to the user's explicit preferences even if they liked nothing.

    Parameters
    ----------
    liked_vectors : (L, 22) array — one row per liked restaurant's feature
        vector from ``build_feature_matrix`` (same order as ``liked_metadata``
        if provided).  May be empty.
    profile_vector : (22,) array — the profile-only user vector from
        ``build_user_feature_vector``.  Used as a fallback and as a small
        bias term during fusion.
    restaurant_matrix : (N, 22) array of candidate restaurant features.
    restaurant_df : DataFrame with restaurant metadata (``restaurant_id``
        required).  Rows align with ``restaurant_matrix``.
    visited_ids : set of restaurant_id strings to exclude from output.
    liked_metadata : optional list of L dicts with at least
        ``restaurant_id`` and ``name`` — used to attribute the
        ``primary_influencer`` column.  If ``None``, influencer labels
        fall back to ``"liked #i"``.
    k_per_liked : how many top neighbours to keep per liked restaurant.
    k_final : total number of candidates to return for downstream re-ranking.
    scaler : StandardScaler fit by clustering (see ``_scaled_space``).
    rrf_constant : RRF damping constant (60 is the standard value from
        Cormack et al. 2009).

    Returns
    -------
    DataFrame of top ``k_final`` candidates with columns
    ``knn_similarity`` (cosine-sim to profile), ``rrf_score`` (fusion
    score), and ``primary_influencer`` (the liked restaurant that gave
    the candidate its best rank).
    """
    visited_set = {str(v) for v in visited_ids}
    restaurant_ids = restaurant_df["restaurant_id"].astype(str).values
    visited_mask = np.array([rid in visited_set for rid in restaurant_ids])

    X_scaled = _scaled_space(restaurant_matrix, scaler)
    profile_scaled = _scaled_space(profile_vector, scaler)
    profile_sim = cosine_similarity(profile_scaled, X_scaled).flatten()

    # Profile-only fallback when no likes are available.
    if liked_vectors is None or len(liked_vectors) == 0:
        sims = profile_sim.copy()
        sims[visited_mask] = -np.inf
        top = np.argsort(sims)[::-1][:k_final]
        result = restaurant_df.iloc[top].copy()
        result["knn_similarity"] = sims[top]
        result["rrf_score"] = sims[top]
        result["primary_influencer"] = "Profile preferences"
        return result.reset_index(drop=True)

    liked_scaled = _scaled_space(np.asarray(liked_vectors, dtype=np.float32), scaler)
    per_liked_sims = cosine_similarity(liked_scaled, X_scaled)  # (L, N)

    # Rank each row; rank_l[r] is r's 0-based rank among unvisited restaurants
    # in liked-query l (lower rank = more similar).
    rrf_scores = np.zeros(len(restaurant_df), dtype=np.float64)
    # Track which liked restaurant gave each candidate its best (smallest) rank.
    best_rank = np.full(len(restaurant_df), np.iinfo(np.int64).max, dtype=np.int64)
    best_source = np.full(len(restaurant_df), -1, dtype=np.int64)

    for l, sims_l in enumerate(per_liked_sims):
        masked = sims_l.copy()
        masked[visited_mask] = -np.inf
        top_idx = np.argsort(masked)[::-1][:k_per_liked]
        for rank, candidate_idx in enumerate(top_idx):
            if masked[candidate_idx] == -np.inf:
                break
            rrf_scores[candidate_idx] += 1.0 / (rank + rrf_constant)
            if rank < best_rank[candidate_idx]:
                best_rank[candidate_idx] = rank
                best_source[candidate_idx] = l

    # Small profile bias keeps explicit preferences relevant when likes are
    # sparse or divergent from stated preferences.
    rrf_scores += 0.1 * profile_sim
    rrf_scores[visited_mask] = -np.inf

    top_indices = np.argsort(rrf_scores)[::-1][:k_final]
    top_indices = [i for i in top_indices if rrf_scores[i] > -np.inf]

    result = restaurant_df.iloc[top_indices].copy()
    result["knn_similarity"] = profile_sim[top_indices]
    result["rrf_score"] = rrf_scores[top_indices]

    def _influencer_label(idx):
        src = int(best_source[idx])
        if src < 0:
            return "Profile preferences"
        if liked_metadata and src < len(liked_metadata):
            meta = liked_metadata[src]
            return str(meta.get("name") or meta.get("dba") or f"liked #{src + 1}")
        return f"liked #{src + 1}"

    result["primary_influencer"] = [_influencer_label(i) for i in top_indices]
    return result.reset_index(drop=True)


def apply_mmr(
    candidates_df: pd.DataFrame,
    candidate_matrix: np.ndarray,
    user_vector: np.ndarray,
    k: int = 15,
    lambda_: float = 0.7,
    scaler: StandardScaler | None = None,
    relevance_column: str = "rrf_score",
) -> pd.DataFrame:
    """Greedy Maximal Marginal Relevance re-ranking.

    At each step, pick the candidate that maximises

        lambda * relevance(r) - (1 - lambda) * max_{j in selected} sim(r, j)

    where ``relevance(r)`` comes from ``candidates_df[relevance_column]``
    (fallback: cosine similarity to ``user_vector``) and pairwise
    similarities are cosine distances in the scaled 22-dim feature space.

    Returns the selected candidates in MMR order.
    """
    if len(candidates_df) == 0:
        return candidates_df.copy()

    n = len(candidates_df)
    k = min(k, n)
    lambda_ = float(np.clip(lambda_, 0.0, 1.0))

    X_scaled = _scaled_space(candidate_matrix, scaler)
    # Normalise scaled vectors for an efficient dot-product cosine-sim.
    norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_unit = X_scaled / norms
    pairwise_sim = X_unit @ X_unit.T  # (n, n)

    if relevance_column in candidates_df.columns:
        relevance = candidates_df[relevance_column].values.astype(np.float64)
    else:
        u_scaled = _scaled_space(user_vector, scaler)
        u_unit = u_scaled / max(np.linalg.norm(u_scaled), 1e-8)
        relevance = (X_unit @ u_unit.flatten()).astype(np.float64)

    # Normalise relevance to [0, 1] so it is on the same scale as cosine sim.
    rmin, rmax = float(relevance.min()), float(relevance.max())
    if rmax > rmin:
        relevance_norm = (relevance - rmin) / (rmax - rmin)
    else:
        relevance_norm = np.zeros_like(relevance)

    selected: list[int] = []
    remaining = set(range(n))
    max_pairwise = np.zeros(n)

    # First pick: pure relevance.
    first = int(np.argmax(relevance_norm))
    selected.append(first)
    remaining.discard(first)
    max_pairwise = np.maximum(max_pairwise, pairwise_sim[first])

    while len(selected) < k and remaining:
        idx_list = np.array(sorted(remaining))
        mmr_scores = lambda_ * relevance_norm[idx_list] - (1 - lambda_) * max_pairwise[idx_list]
        choice = int(idx_list[int(np.argmax(mmr_scores))])
        selected.append(choice)
        remaining.discard(choice)
        max_pairwise = np.maximum(max_pairwise, pairwise_sim[choice])

    result = candidates_df.iloc[selected].copy()
    result["mmr_rank"] = range(1, len(selected) + 1)
    return result.reset_index(drop=True)


def collect_liked_vectors(
    profile: dict,
    restaurant_matrix: np.ndarray,
    restaurant_df: pd.DataFrame,
) -> tuple[np.ndarray, list]:
    """Extract liked restaurants' feature rows from ``restaurant_matrix``.

    Returns (liked_vectors, liked_metadata) where ``liked_metadata`` is a
    list of dicts aligned to the rows of ``liked_vectors``.  Likes whose
    ``restaurant_id`` cannot be found in ``restaurant_df`` are dropped.
    """
    likes = profile.get("likes", []) or []
    if not likes:
        return np.zeros((0, restaurant_matrix.shape[1]), dtype=np.float32), []

    id_to_idx = {
        rid: idx
        for idx, rid in enumerate(restaurant_df["restaurant_id"].astype(str).values)
    }

    rows, metadata = [], []
    for like in likes:
        rid = str(like.get("restaurant_id", "")).strip()
        if not rid or rid not in id_to_idx:
            continue
        idx = id_to_idx[rid]
        rows.append(restaurant_matrix[idx])
        metadata.append({
            "restaurant_id": rid,
            "name": like.get("dba") or restaurant_df.iloc[idx].get("name", ""),
            "rating": float(like.get("rating", 3.0)),
            "cuisine": like.get("cuisine", restaurant_df.iloc[idx].get("cuisine_type", "")),
        })
    if not rows:
        return np.zeros((0, restaurant_matrix.shape[1]), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), metadata

