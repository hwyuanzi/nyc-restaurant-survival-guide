import os
import time
import json
import hashlib
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "data/cluster_cache.parquet"
MODEL_PATH = "data/kmeans_model.joblib"
CACHE_TTL  = 86400  # 24 hours
CLUSTER_SCHEMA_VERSION = 20  # Bumped to force rebuild with interpretable features
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
    # Weights control relative importance: cuisine and price are primary clustering drivers
    X = np.hstack([
        price_norm * 1.2,
        rating_norm * 1.0,
        review_norm * 0.7,
        health_norm * 0.5,
        lat_norm * 0.6,
        lng_norm * 0.6,
        cuisine_dummies * 1.0,
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


def _cluster_signature(df: pd.DataFrame, user_history: dict, k: int):
    history_payload = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
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


def run_kmeans(df: pd.DataFrame, user_history: dict, k: int = 8):
    X, feature_columns, df = build_feature_matrix(df)
    X_aug = apply_user_weights(X, df, user_history)
    projection_feature_columns = feature_columns + ["user_affinity"]

    X_scaled, X_cluster, scaler = prepare_clustering_space(X_aug, fit=True)

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

    kmeans = best_model or KMeansClass(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    ).fit(X_cluster)
    df["cluster_id"] = best_labels if best_labels is not None else kmeans.labels_

    # Merge degenerate single-restaurant clusters
    cluster_sizes = df["cluster_id"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
    centroids = kmeans.cluster_centers_
    for cid in small_clusters:
        idx = df[df["cluster_id"] == cid].index[0]
        vec = X_cluster[df.index.get_loc(idx)].reshape(1, -1)
        dists = np.linalg.norm(centroids - vec, axis=1)
        dists[cid] = np.inf
        df.loc[idx, "cluster_id"] = int(np.argmin(dists))

    # PCA 3D coordinates
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

    # Cluster-oriented display coordinates based on centroid distances.
    centroid_distances = kmeans.transform(X_cluster)
    cluster_view_pca = PCA(n_components=3, random_state=42)
    X_cluster_view = cluster_view_pca.fit_transform(StandardScaler().fit_transform(centroid_distances))
    df["cluster_view_x"] = X_cluster_view[:, 0]
    df["cluster_view_y"] = X_cluster_view[:, 1]
    df["cluster_view_z"] = X_cluster_view[:, 2]

    # t-SNE 3D coordinates for visualization only.
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

    # Auto-label clusters with unique descriptive names
    df["cluster_label"] = _assign_cluster_labels(df)

    # User affinity score (last column of X_aug)
    df["user_affinity_score"] = X_aug[:, -1]

    # UMAP (optional)
    if UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            df["umap_x"] = X_umap[:, 0]
            df["umap_y"] = X_umap[:, 1]
            df["umap_z"] = X_umap[:, 2]
        except Exception:
            pass

    return df, kmeans, scaler, pca


def cache_is_fresh():
    if not os.path.exists(CACHE_PATH):
        return False
    return (time.time() - os.path.getmtime(CACHE_PATH)) < CACHE_TTL


def load_cache():
    df = pd.read_parquet(CACHE_PATH)
    artifacts = joblib.load(MODEL_PATH)
    return df, artifacts["kmeans"], artifacts["scaler"], artifacts["pca"], artifacts.get("signature")


def save_cache(df, kmeans, scaler, pca, signature):
    os.makedirs("data", exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    joblib.dump({"kmeans": kmeans, "scaler": scaler, "pca": pca, "signature": signature}, MODEL_PATH)


def get_clustered_data(df: pd.DataFrame, user_history: dict, k: int = 8, force: bool = False):
    signature = _cluster_signature(df, user_history, k)
    if not force and cache_is_fresh():
        cached_df, cached_kmeans, cached_scaler, cached_pca, cached_signature = load_cache()
        cached_labels = cached_df.groupby("cluster_id")["cluster_label"].first()
        has_duplicate_labels = cached_labels.duplicated().any()
        has_internal_labels = cached_labels.map(_label_looks_internal).any()
        has_cluster_view = {"cluster_view_x", "cluster_view_y", "cluster_view_z"}.issubset(cached_df.columns)
        has_tsne_view = {"tsne_x", "tsne_y", "tsne_z"}.issubset(cached_df.columns)
        if cached_signature == signature and not has_duplicate_labels and not has_internal_labels and has_cluster_view and has_tsne_view:
            return cached_df, cached_kmeans, cached_scaler, cached_pca
    result_df, kmeans, scaler, pca = run_kmeans(df, user_history, k)
    save_cache(result_df, kmeans, scaler, pca, signature)
    return result_df, kmeans, scaler, pca


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
        [price_norm * 1.2],
        [avg_rating_norm * 1.0],
        [avg_review_norm * 0.7],
        [avg_health_norm * 0.5],
        [lat_norm * 0.6],
        [lng_norm * 0.6],
        cuisine_vec * 1.0,
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
    # The scaler expects 23 features (build_feature_matrix output + user_affinity).
    # Append a placeholder user_affinity column to match.
    n_rows = restaurant_matrix.shape[0]
    X_aug = np.hstack([restaurant_matrix, np.zeros((n_rows, 1), dtype=np.float32)])
    u_aug = np.append(user_vector, [0.5]).reshape(1, -1)  # neutral affinity

    if scaler is not None:
        X_scaled = scaler.transform(X_aug)
        u_scaled = scaler.transform(u_aug)
    else:
        X_scaled = X_aug
        u_scaled = u_aug

    similarities = cosine_similarity(u_scaled, X_scaled).flatten()

    # Mask visited restaurants
    visited_mask = restaurant_df["restaurant_id"].astype(str).isin(visited_ids)
    similarities[visited_mask.values] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:k]
    result = restaurant_df.iloc[top_indices].copy()
    result["knn_similarity"] = similarities[top_indices]
    return result.reset_index(drop=True)

