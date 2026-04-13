import os
import time
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

CACHE_PATH = "data/cluster_cache.parquet"
MODEL_PATH = "data/kmeans_model.joblib"
CACHE_TTL  = 86400  # 24 hours
CLUSTER_SCHEMA_VERSION = 3

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

REQUIRED_COLUMNS = ["restaurant_id", "name", "lat", "lng", "cuisine_type", "price_tier", "avg_rating", "review_count"]
CUISINE_FAMILY_KEYWORDS = {
    "Asian": {"chinese", "japanese", "korean", "thai", "vietnamese", "indian", "sushi", "ramen", "dim sum"},
    "European": {"italian", "french", "spanish", "greek", "mediterranean", "pizza", "pasta"},
    "Latin American": {"mexican", "latin", "caribbean", "peruvian", "cuban", "dominican"},
    "American": {"american", "burger", "chicken", "sandwich", "steak", "barbecue", "bbq"},
    "Cafe & Bakery": {"cafe", "coffee", "bakery", "dessert", "donut", "juice"},
    "Middle Eastern & African": {"middle eastern", "ethiopian", "halal", "shawarma", "falafel"},
}
FEATURE_LABELS = {
    "price_tier_norm": "Price level",
    "rating_norm": "Google rating",
    "review_norm": "Review volume",
    "lat_norm": "North-south location",
    "lng_norm": "East-west location",
    "user_affinity": "User affinity",
}


def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"restaurants dataset is missing required columns: {missing}")


def _cuisine_family(cuisine_name):
    name = str(cuisine_name or "").lower()
    for family, keywords in CUISINE_FAMILY_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            return family
    return "Other"


def _cluster_label(cluster_cuisines: pd.Series):
    clean_values = cluster_cuisines.fillna("").astype(str)
    if clean_values.empty:
        return "Mixed Cluster"

    family_counts = clean_values.map(_cuisine_family).value_counts()
    top_family = family_counts.index[0]
    top_family_share = family_counts.iloc[0] / max(len(clean_values), 1)
    top_cuisines = clean_values.value_counts().head(2).index.tolist()

    if top_family != "Other" and top_family_share >= 0.6:
        return f"{top_family} Favorites"
    if len(top_cuisines) >= 2:
        return f"{top_cuisines[0]} + {top_cuisines[1]}"
    return f"{top_cuisines[0]} & Similar"


def _price_descriptor(price_value):
    if pd.isna(price_value):
        return "Mixed Price"
    if price_value <= 1.5:
        return "Budget"
    if price_value <= 2.5:
        return "Mid-Range"
    return "Upscale"


def _cluster_label_candidates(cluster_df: pd.DataFrame):
    cuisines = cluster_df["cuisine_type"].fillna("").astype(str)
    top_cuisines = cuisines.value_counts().head(3).index.tolist()
    borough_counts = cluster_df["boro"].fillna("").astype(str).value_counts()
    top_borough = borough_counts.index[0] if not borough_counts.empty else ""
    borough_share = borough_counts.iloc[0] / max(len(cluster_df), 1) if not borough_counts.empty else 0.0
    price_desc = _price_descriptor(pd.to_numeric(cluster_df["price_tier"], errors="coerce").mean())
    family_counts = cuisines.map(_cuisine_family).value_counts()
    top_family = family_counts.index[0] if not family_counts.empty else "Mixed"

    candidates = []
    if len(top_cuisines) >= 2:
        candidates.append(f"{top_cuisines[0]} + {top_cuisines[1]}")
    if top_cuisines:
        candidates.append(f"{top_cuisines[0]} {price_desc}")
    if top_cuisines and top_borough and borough_share >= 0.33:
        candidates.append(f"{top_borough} {top_cuisines[0]}")
    if top_family != "Other" and top_borough and borough_share >= 0.33:
        candidates.append(f"{top_borough} {top_family}")
    if top_family != "Other":
        candidates.append(f"{top_family} {price_desc}")
    if top_cuisines:
        candidates.append(f"{top_cuisines[0]} & Similar")
    candidates.append(f"Cluster {int(cluster_df['cluster_id'].iloc[0]) + 1}")
    return candidates


def _assign_cluster_labels(df: pd.DataFrame):
    label_map = {}
    used_labels = set()
    cluster_summaries = []

    for cluster_id, cluster_df in df.groupby("cluster_id"):
        cluster_summaries.append(
            (
                cluster_id,
                cluster_df["cuisine_type"].nunique(),
                len(cluster_df),
                cluster_df["cuisine_type"].value_counts().iloc[0] if not cluster_df.empty else 0,
                cluster_df.copy(),
            )
        )

    cluster_summaries.sort(key=lambda item: (item[1], item[3], item[2], -int(item[0])), reverse=True)

    for cluster_id, _, _, _, cluster_df in cluster_summaries:
        chosen_label = None
        for candidate in _cluster_label_candidates(cluster_df):
            if candidate not in used_labels:
                chosen_label = candidate
                break
        if chosen_label is None:
            chosen_label = f"Cluster {int(cluster_id) + 1}"
        label_map[cluster_id] = chosen_label
        used_labels.add(chosen_label)

    return df["cluster_id"].map(label_map)


def _feature_category(feature_name):
    if feature_name.startswith("cuisine_") or feature_name.startswith("family_"):
        return "Cuisine"
    if feature_name == "price_tier_norm":
        return "Price"
    if feature_name in {"rating_norm", "review_norm"}:
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
    if feature_name.startswith("family_"):
        return feature_name.replace("family_", "").replace("_", " ").title()
    if feature_name.startswith("tag_"):
        return feature_name.replace("tag_", "").replace("_", " ").title()
    return feature_name.replace("_", " ").title()


def _component_axis_label(component, feature_columns):
    weights = pd.Series(np.abs(component), index=feature_columns)
    category_weights = weights.groupby([_feature_category(name) for name in feature_columns]).sum().sort_values(ascending=False)
    primary_category = category_weights.index[0] if not category_weights.empty else "Pattern"
    secondary_category = category_weights.index[1] if len(category_weights) > 1 else None

    category_labels = {
        "Cuisine": "Taste Profile",
        "Price": "Price Level",
        "Quality": "Quality & Popularity",
        "Location": "NYC Geography",
        "Affinity": "User Match",
        "Other": "Mixed Factors",
    }
    primary_label = category_labels.get(primary_category, "Mixed Factors")
    secondary_label = category_labels.get(secondary_category, "") if secondary_category else ""

    if primary_category == "Cuisine" and secondary_category in {"Price", "Quality", "Location", "Affinity"}:
        return f"{primary_label} vs {secondary_label}"
    if primary_category in {"Price", "Quality", "Location", "Affinity"} and secondary_category == "Cuisine":
        return f"{primary_label} vs Taste"
    return primary_label


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
    validate_dataframe(df)

    # 1. Cuisine one-hot — group low-frequency (<1%) into 'other'
    cuisine_counts = df["cuisine_type"].value_counts(normalize=True)
    df = df.copy()
    df["cuisine_type_grouped"] = df["cuisine_type"].apply(
        lambda x: x if cuisine_counts.get(x, 0) >= 0.01 else "other"
    )
    df["cuisine_family"] = df["cuisine_type"].apply(_cuisine_family)
    cuisine_dummies = pd.get_dummies(df["cuisine_type_grouped"], prefix="cuisine")
    cuisine_family_dummies = pd.get_dummies(df["cuisine_family"], prefix="family")

    # 2. Price tier normalized
    price_norm = ((df["price_tier"].fillna(2) - 1) / 3.0).values.reshape(-1, 1)

    # 3. Rating normalized
    rating_norm = ((df["avg_rating"].fillna(3.0) - 1.0) / 4.0).values.reshape(-1, 1)

    # 4. Review count log-scaled then min-max
    log_reviews = np.log1p(df["review_count"].fillna(0).values).reshape(-1, 1)
    log_min, log_max = log_reviews.min(), log_reviews.max()
    if log_max > log_min:
        review_norm = (log_reviews - log_min) / (log_max - log_min)
    else:
        review_norm = np.zeros_like(log_reviews)

    # 5. Geo coordinates — NYC bounding box, weighted 0.5x
    lat_norm = ((df["lat"].fillna(40.7128) - 40.4774) / (40.9176 - 40.4774)).values.reshape(-1, 1) * 0.5
    lng_norm = ((df["lng"].fillna(-74.006) - (-74.2591)) / (-73.7004 - (-74.2591))).values.reshape(-1, 1) * 0.5

    # 6. Tag binary features (top 30 tags)
    tag_features = np.zeros((len(df), 0))
    if "tags" in df.columns:
        all_tags = []
        for tags in df["tags"].dropna():
            all_tags.extend([t.strip() for t in str(tags).split(",")])
        top_tags = pd.Series(all_tags).value_counts().head(30).index.tolist()
        tag_matrix = np.zeros((len(df), len(top_tags)))
        for i, tags in enumerate(df["tags"].fillna("")):
            tag_set = set(t.strip() for t in str(tags).split(","))
            for j, tag in enumerate(top_tags):
                tag_matrix[i, j] = 1.0 if tag in tag_set else 0.0
        tag_features = tag_matrix

    X = np.hstack([
        cuisine_dummies.values * 2.8,
        cuisine_family_dummies.values * 1.6,
        price_norm * 0.9,
        rating_norm * 0.7,
        review_norm * 0.45,
        lat_norm * 0.2,
        lng_norm * 0.2,
        tag_features,
    ]).astype(np.float32)

    feature_columns = (
        list(cuisine_dummies.columns) +
        list(cuisine_family_dummies.columns) +
        ["price_tier_norm", "rating_norm", "review_norm", "lat_norm", "lng_norm"] +
        ([f"tag_{t}" for t in top_tags] if "tags" in df.columns else [])
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

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    # Use MiniBatchKMeans for large datasets
    KMeansClass = MiniBatchKMeans if len(df) > 10000 else KMeans
    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    kmeans = KMeansClass(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )
    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Merge degenerate single-restaurant clusters
    cluster_sizes = df["cluster_id"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
    centroids = kmeans.cluster_centers_
    for cid in small_clusters:
        idx = df[df["cluster_id"] == cid].index[0]
        vec = X_scaled[df.index.get_loc(idx)].reshape(1, -1)
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
    centroid_distances = kmeans.transform(X_scaled)
    cluster_view_pca = PCA(n_components=3, random_state=42)
    X_cluster_view = cluster_view_pca.fit_transform(StandardScaler().fit_transform(centroid_distances))
    df["cluster_view_x"] = X_cluster_view[:, 0]
    df["cluster_view_y"] = X_cluster_view[:, 1]
    df["cluster_view_z"] = X_cluster_view[:, 2]

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
        has_cluster_view = {"cluster_view_x", "cluster_view_y", "cluster_view_z"}.issubset(cached_df.columns)
        if cached_signature == signature and not has_duplicate_labels and has_cluster_view:
            return cached_df, cached_kmeans, cached_scaler, cached_pca
    result_df, kmeans, scaler, pca = run_kmeans(df, user_history, k)
    save_cache(result_df, kmeans, scaler, pca, signature)
    return result_df, kmeans, scaler, pca
