"""
utils/clustering.py — K-Means clustering pipeline for restaurant features.

Builds a feature matrix from restaurant data (cuisine one-hot, price, rating,
reviews, geo coordinates, tags), applies user affinity weighting, and runs
K-Means with PCA dimensionality reduction.

Original author: Rahul Adusumalli
Integrated by: Ryan Han (PapTR)
"""

import os
import time
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

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

REQUIRED_COLUMNS = ["restaurant_id", "name", "lat", "lng", "cuisine_type", "price_tier", "avg_rating", "review_count"]


def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"restaurants dataset is missing required columns: {missing}")


def build_feature_matrix(df: pd.DataFrame):
    validate_dataframe(df)

    cuisine_counts = df["cuisine_type"].value_counts(normalize=True)
    df = df.copy()
    df["cuisine_type_grouped"] = df["cuisine_type"].apply(
        lambda x: x if cuisine_counts.get(x, 0) >= 0.01 else "other"
    )
    cuisine_dummies = pd.get_dummies(df["cuisine_type_grouped"], prefix="cuisine")

    price_norm = ((df["price_tier"].fillna(2) - 1) / 3.0).values.reshape(-1, 1)
    rating_norm = ((df["avg_rating"].fillna(3.0) - 1.0) / 4.0).values.reshape(-1, 1)

    log_reviews = np.log1p(df["review_count"].fillna(0).values).reshape(-1, 1)
    log_min, log_max = log_reviews.min(), log_reviews.max()
    if log_max > log_min:
        review_norm = (log_reviews - log_min) / (log_max - log_min)
    else:
        review_norm = np.zeros_like(log_reviews)

    lat_norm = ((df["lat"].fillna(40.7128) - 40.4774) / (40.9176 - 40.4774)).values.reshape(-1, 1) * 0.5
    lng_norm = ((df["lng"].fillna(-74.006) - (-74.2591)) / (-73.7004 - (-74.2591))).values.reshape(-1, 1) * 0.5

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
        cuisine_dummies.values,
        price_norm,
        rating_norm,
        review_norm,
        lat_norm,
        lng_norm,
        tag_features,
    ]).astype(np.float32)

    feature_columns = (
        list(cuisine_dummies.columns) +
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

    weights = []
    for rid in df.loc[visited_mask, "restaurant_id"]:
        weights.append(rated.get(rid, 3.0) / 5.0)
    weights = np.array(weights).reshape(-1, 1)
    user_vec = (visited_X * weights).sum(axis=0, keepdims=True) / (weights.sum() + 1e-8)

    affinity = cosine_similarity(X, user_vec).flatten()
    return np.hstack([X, affinity.reshape(-1, 1)])


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

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    KMeansClass = MiniBatchKMeans if len(df) > 10000 else KMeans
    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    kmeans = KMeansClass(
        n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42,
    )
    df = df.copy()
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    cluster_sizes = df["cluster_id"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
    centroids = kmeans.cluster_centers_
    for cid in small_clusters:
        idx = df[df["cluster_id"] == cid].index[0]
        vec = X_scaled[df.index.get_loc(idx)].reshape(1, -1)
        dists = np.linalg.norm(centroids - vec, axis=1)
        dists[cid] = np.inf
        df.loc[idx, "cluster_id"] = int(np.argmin(dists))

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]
    df["pca_z"] = X_pca[:, 2]

    df["cluster_label"] = df.groupby("cluster_id")["cuisine_type"].transform(
        lambda x: x.value_counts().index[0] + " & Similar"
    )

    df["user_affinity_score"] = X_aug[:, -1]

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
    return df, artifacts["kmeans"], artifacts["scaler"], artifacts["pca"]


def save_cache(df, kmeans, scaler, pca):
    os.makedirs("data", exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    joblib.dump({"kmeans": kmeans, "scaler": scaler, "pca": pca}, MODEL_PATH)


def get_clustered_data(df: pd.DataFrame, user_history: dict, k: int = 8, force: bool = False):
    if not force and cache_is_fresh():
        return load_cache()
    result_df, kmeans, scaler, pca = run_kmeans(df, user_history, k)
    save_cache(result_df, kmeans, scaler, pca)
    return result_df, kmeans, scaler, pca
