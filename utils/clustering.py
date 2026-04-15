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
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

CACHE_PATH = "data/cluster_cache.parquet"
MODEL_PATH = "data/kmeans_model.joblib"
CACHE_TTL  = 86400  # 24 hours
CLUSTER_SCHEMA_VERSION = 10

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

REQUIRED_COLUMNS = ["restaurant_id", "name", "lat", "lng", "cuisine_type", "price_tier", "avg_rating", "review_count"]
LABEL_BANNED_TERMS = {
    "restaurant", "restaurants", "new", "york", "city", "nyc", "food", "place", "spot", "spots",
    "google", "rating", "reviews", "health", "inspection", "grade", "score", "address", "great",
    "good", "best", "serves", "serving", "dining", "delicious", "brooklyn", "manhattan", "queens",
    "bronx", "staten", "island", "casual", "dinner", "lunch", "breakfast", "cozy", "lively", "late",
    "night", "fresh", "hearty", "counter",
}
FEATURE_LABELS = {
    "semantic_latent_1": "Semantic style",
    "semantic_latent_2": "Semantic style",
    "semantic_latent_3": "Semantic style",
    "semantic_latent_4": "Semantic style",
    "semantic_latent_5": "Semantic style",
    "semantic_latent_6": "Semantic style",
    "semantic_latent_7": "Semantic style",
    "semantic_latent_8": "Semantic style",
    "health_norm": "Health grade",
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


def _build_cluster_text(df: pd.DataFrame):
    tags = df.get("tags", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    summary = df.get("g_summary", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    description = df.get("description", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    cleaned_description = description.str.replace(r"Address:.*$", "", regex=True).str.replace(
        r"Health inspection grade:.*$", "", regex=True
    )
    primary_text = tags.str.replace(",", " ", regex=False) + ". " + summary
    return primary_text.where(primary_text.str.strip().str.len() > 8, cleaned_description).str.strip()


def _cluster_theme_label(cluster_df: pd.DataFrame):
    cluster_text = _build_cluster_text(cluster_df)
    if cluster_text.empty or not cluster_text.str.strip().any():
        return None
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=120)
        matrix = vectorizer.fit_transform(cluster_text)
    except ValueError:
        matrix = None

    if matrix is not None and matrix.shape[1] > 0:
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        ranked_terms = []
        for idx in np.argsort(mean_scores)[::-1]:
            term = str(terms[idx]).strip()
            if not term or term in LABEL_BANNED_TERMS:
                continue
            if any(part in LABEL_BANNED_TERMS for part in term.split()):
                continue
            ranked_terms.append(term)
            if len(ranked_terms) == 3:
                break
        if ranked_terms:
            if len(ranked_terms) >= 2 and len(ranked_terms[0].split()) == 1 and len(ranked_terms[1].split()) == 1:
                return f"{ranked_terms[0].title()} & {ranked_terms[1].title()}"
            return ranked_terms[0].title()

    cuisine_counts = cluster_df["cuisine_type"].fillna("").astype(str).value_counts()
    top_cuisines = [value for value in cuisine_counts.index.tolist() if value][:2]
    if len(top_cuisines) >= 2:
        return f"{top_cuisines[0]} + {top_cuisines[1]}"
    if top_cuisines:
        return top_cuisines[0]
    return None


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


def _review_descriptor(review_value):
    if pd.isna(review_value):
        return "Steady"
    if review_value >= 400:
        return "Crowd Favorites"
    if review_value >= 150:
        return "Local Staples"
    return "Hidden Gems"


def _health_descriptor(score_value, grade_value):
    grade = str(grade_value or "").strip().upper()
    if grade == "A":
        return "Strong Health Scores"
    if grade == "B":
        return "Solid Health Scores"
    if grade == "C":
        return "Mixed Health Scores"
    numeric_score = pd.to_numeric(score_value, errors="coerce")
    if pd.notna(numeric_score) and numeric_score <= 10:
        return "Strong Health Scores"
    return "Mixed Health Scores"


def _fallback_cluster_label(cluster_df: pd.DataFrame):
    theme_desc = _cluster_theme_label(cluster_df)
    price_desc = _price_descriptor(pd.to_numeric(cluster_df["price_tier"], errors="coerce").mean())
    rating_desc = _rating_descriptor(pd.to_numeric(cluster_df["avg_rating"], errors="coerce").mean())

    if theme_desc:
        return f"{theme_desc} {rating_desc}"
    return f"{price_desc} {rating_desc}"


def _label_looks_internal(label):
    text = str(label or "").strip()
    return not text or bool(re.match(r"^(cluster\s*\d+|\d+\b)", text, flags=re.IGNORECASE))


def _cluster_label_candidates(cluster_df: pd.DataFrame):
    theme_desc = _cluster_theme_label(cluster_df)
    price_desc = _price_descriptor(pd.to_numeric(cluster_df["price_tier"], errors="coerce").mean())
    rating_desc = _rating_descriptor(pd.to_numeric(cluster_df["avg_rating"], errors="coerce").mean())
    review_desc = _review_descriptor(pd.to_numeric(cluster_df["review_count"], errors="coerce").mean())
    dominant_grade = ""
    if "grade" in cluster_df.columns:
        grade_mode = cluster_df["grade"].dropna().astype(str)
        if not grade_mode.empty:
            dominant_grade = grade_mode.mode().iloc[0]
    score_series = cluster_df["score"] if "score" in cluster_df.columns else pd.Series([np.nan] * len(cluster_df), index=cluster_df.index)
    health_desc = _health_descriptor(
        pd.to_numeric(score_series, errors="coerce").mean(),
        dominant_grade,
    )

    candidates = []
    if theme_desc:
        candidates.append(theme_desc)
        candidates.append(f"{theme_desc} {rating_desc}")
        candidates.append(f"{theme_desc} {price_desc}")
    candidates.append(f"{price_desc} {rating_desc}")
    candidates.append(f"{rating_desc} {review_desc}")
    candidates.append(f"{health_desc} {rating_desc}")
    candidates.append(_fallback_cluster_label(cluster_df))
    return candidates


def _assign_cluster_labels(df: pd.DataFrame):
    label_map = {}
    used_labels = set()
    cluster_summaries = []

    for cluster_id, cluster_df in df.groupby("cluster_id"):
        cluster_summaries.append(
            (
                cluster_id,
                cluster_df["cuisine_type"].fillna("").astype(str).nunique(),
                len(cluster_df),
                pd.to_numeric(cluster_df["review_count"], errors="coerce").fillna(0).mean(),
                cluster_df.copy(),
            )
        )

    cluster_summaries.sort(key=lambda item: (item[1], item[3], item[2], -int(item[0])), reverse=True)

    for cluster_id, _, _, _, cluster_df in cluster_summaries:
        chosen_label = None
        for candidate in _cluster_label_candidates(cluster_df):
            if candidate not in used_labels and not _label_looks_internal(candidate):
                chosen_label = candidate
                break
        if chosen_label is None:
            chosen_label = _fallback_cluster_label(cluster_df)
        label_map[cluster_id] = chosen_label
        used_labels.add(chosen_label)

    return df["cluster_id"].map(label_map)


def _feature_category(feature_name):
    if feature_name.startswith("semantic_latent_"):
        return "Semantics"
    if feature_name == "price_tier_norm":
        return "Price"
    if feature_name in {"rating_norm", "review_norm", "health_norm"}:
        return "Quality"
    if feature_name == "user_affinity":
        return "Affinity"
    return "Other"


def _humanize_feature(feature_name):
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    if feature_name.startswith("semantic_latent_"):
        return "Semantic style"
    if feature_name.startswith("tag_"):
        return feature_name.replace("tag_", "").replace("_", " ").title()
    return feature_name.replace("_", " ").title()


def _component_axis_label(component, feature_columns):
    weights = pd.Series(np.abs(component), index=feature_columns)
    category_weights = weights.groupby([_feature_category(name) for name in feature_columns]).sum().sort_values(ascending=False)
    primary_category = category_weights.index[0] if not category_weights.empty else "Pattern"
    secondary_category = category_weights.index[1] if len(category_weights) > 1 else None

    category_labels = {
        "Semantics": "Restaurant Style",
        "Price": "Price Level",
        "Quality": "Quality & Popularity",
        "Affinity": "User Match",
        "Other": "Mixed Factors",
    }
    primary_label = category_labels.get(primary_category, "Mixed Factors")
    secondary_label = category_labels.get(secondary_category, "") if secondary_category else ""

    if primary_category == "Semantics" and secondary_category in {"Price", "Quality", "Affinity"}:
        return f"{primary_label} vs {secondary_label}"
    if primary_category in {"Price", "Quality", "Affinity"} and secondary_category == "Semantics":
        return f"{primary_label} vs Style"
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

    df = df.copy()

    text_corpus = _build_cluster_text(df)
    tfidf_max_features = min(1500, max(len(df) * 3, 250))
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if len(df) >= 50 else 1,
        max_df=0.9,
        max_features=tfidf_max_features,
    )
    try:
        text_matrix = vectorizer.fit_transform(text_corpus)
    except ValueError:
        text_matrix = None

    semantic_features = np.zeros((len(df), 0), dtype=np.float32)
    semantic_feature_columns = []
    if text_matrix is not None and text_matrix.shape[1] > 0:
        semantic_dims = min(8, text_matrix.shape[0] - 1, text_matrix.shape[1] - 1)
        if semantic_dims >= 2:
            svd = TruncatedSVD(n_components=semantic_dims, random_state=42)
            semantic_features = svd.fit_transform(text_matrix).astype(np.float32)
            semantic_feature_columns = [f"semantic_latent_{idx + 1}" for idx in range(semantic_features.shape[1])]

    # 1. Price tier normalized
    price_norm = ((df["price_tier"].fillna(2) - 1) / 3.0).values.reshape(-1, 1)

    # 2. Rating normalized
    rating_norm = ((df["avg_rating"].fillna(3.0) - 1.0) / 4.0).values.reshape(-1, 1)

    # 3. Review count log-scaled then min-max
    log_reviews = np.log1p(df["review_count"].fillna(0).values).reshape(-1, 1)
    log_min, log_max = log_reviews.min(), log_reviews.max()
    if log_max > log_min:
        review_norm = (log_reviews - log_min) / (log_max - log_min)
    else:
        review_norm = np.zeros_like(log_reviews)

    # 4. Health inspection score, where lower is better
    health_series = df["score"] if "score" in df.columns else pd.Series([21] * len(df), index=df.index)
    health_source = pd.to_numeric(health_series, errors="coerce").fillna(21).clip(0, 42).values.reshape(-1, 1)
    health_norm = 1 - (health_source / 42.0)

    X = np.hstack([
        semantic_features * 1.7,
        price_norm * 1.0,
        rating_norm * 1.0,
        review_norm * 0.8,
        health_norm * 0.55,
    ]).astype(np.float32)

    feature_columns = (
        semantic_feature_columns +
        ["price_tier_norm", "rating_norm", "review_norm", "health_norm"]
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
        if cached_signature == signature and not has_duplicate_labels and not has_internal_labels and has_cluster_view:
            return cached_df, cached_kmeans, cached_scaler, cached_pca
    result_df, kmeans, scaler, pca = run_kmeans(df, user_history, k)
    save_cache(result_df, kmeans, scaler, pca, signature)
    return result_df, kmeans, scaler, pca
