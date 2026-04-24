"""
The problem we are fixing:
  If a user likes 4 Chinese restaurants OR sets "Chinese" as a favorite cuisine,
  the top-15 recommendations should be dominated by Chinese restaurants — not
  Thai, Japanese, etc. that happen to match on price/rating/location.

Root causes identified:
  1. build_user_feature_vector normalizes the cuisine one-hot to sum=1, diluting
     the Chinese signal when there are multiple likes.
  2. Non-top-10 cuisines (Thai, Indian, Vietnamese, Korean, ...) all collapse
     into the single "cuisine_Other" bucket.  If a Thai restaurant and a
     Vietnamese restaurant both have cuisine_Other=1, they look identical on
     the cuisine axis even though they are very different cuisines.
  3. Price (× 1.5) and Rating (× 1.3) outweigh any individual cuisine dimension
     (× 0.8), so a Thai restaurant with matching price/rating can beat a
     Chinese restaurant with slightly mismatched price/rating.

Fixes:
  A. Strengthen the user's cuisine signal in build_user_feature_vector so a
     single cuisine preference produces a full 1.0 in that dimension.
  B. Add a cuisine_boost() post-filter: after K-NN returns top-N candidates,
     boost restaurants whose cuisine matches either the user's favorite_cuisines
     or the cuisines of their likes, and penalize mismatches.  This lives
     *outside* the cosine similarity so we can tune it without retraining.
  C. Keep the existing interpretable-feature cosine search for ranking within
     a cuisine family (since that's still where price/rating/location matter).

Paste these into utils/clustering.py, replacing:
  - build_user_feature_vector  (lines ~1020-1112)
  - recommend_per_liked_knn    (lines ~1177-1291)
And add a new helper cuisine_score_boost() right before recommend_per_liked_knn.
"""

import os
import time
import json
import hashlib
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from models.kmeans_scratch import KMeansScratch

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
CLUSTER_SCHEMA_VERSION = 22  # Scratch K-Means + stable global clusters + profile labels
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


def _driver_short_label(feature_name: str, delta: float):
    if feature_name.startswith("cuisine_") and delta > 0:
        return feature_name.replace("cuisine_", "").replace("_", " ")
    if feature_name.startswith("boro_") and delta > 0:
        return feature_name.replace("boro_", "")

    positive = {
        "price_norm": "Premium",
        "rating_norm": "Highly Rated",
        "review_norm": "Popular",
        "health_norm": "Strong Health",
    }
    negative = {
        "price_norm": "Budget-Friendly",
    }
    if delta >= 0:
        return positive.get(feature_name)
    return negative.get(feature_name)


def _driver_phrase(feature_name: str, delta: float):
    if feature_name.startswith("cuisine_") and delta > 0:
        cuisine = feature_name.replace("cuisine_", "").replace("_", " ")
        return f"over-indexes on {cuisine}"
    if feature_name.startswith("boro_") and delta > 0:
        borough = feature_name.replace("boro_", "")
        return f"is concentrated in {borough}"

    positive = {
        "price_norm": "higher prices",
        "rating_norm": "higher ratings",
        "review_norm": "stronger review volume",
        "health_norm": "better health inspection scores",
        "lat_norm": "more northern locations",
        "lng_norm": "more eastern locations",
    }
    negative = {
        "price_norm": "more budget-friendly prices",
        "rating_norm": "lower ratings",
        "review_norm": "lighter review volume",
        "health_norm": "weaker health inspection scores",
        "lat_norm": "more southern locations",
        "lng_norm": "more western locations",
    }
    if delta >= 0:
        return positive.get(feature_name)
    return negative.get(feature_name)


def _select_cluster_drivers(feature_means: pd.Series, global_means: pd.Series):
    deltas = (feature_means - global_means).sort_values(
        key=lambda series: series.abs(), ascending=False
    )
    drivers = []
    seen_categories = set()

    for feature_name, delta in deltas.items():
        category = _feature_category(feature_name)
        abs_delta = abs(float(delta))
        threshold = 0.06 if category in {"Cuisine", "Borough"} else 0.10

        if abs_delta < threshold:
            continue
        if category in {"Cuisine", "Borough"} and delta <= 0:
            continue
        if category in seen_categories and category not in {"Cuisine", "Borough"}:
            continue

        short_label = _driver_short_label(feature_name, float(delta))
        phrase = _driver_phrase(feature_name, float(delta))
        if not short_label or not phrase:
            continue

        drivers.append(
            {
                "feature": feature_name,
                "category": category,
                "delta": float(delta),
                "short_label": short_label,
                "phrase": phrase,
            }
        )
        seen_categories.add(category)
        if len(drivers) >= 4:
            break

    return drivers

# --------------------------------------------------------------------------
# Persona naming — combine signals into a short 2-3 word label
# --------------------------------------------------------------------------

def _price_persona(price):
    """Price tier → 'Budget' / 'Mid-Range' / 'Upscale' / 'Luxury'."""
    if pd.isna(price):
        return None
    if price <= 1.4:
        return "Budget"
    if price <= 2.0:
        return "Mid-Range"
    if price <= 2.8:
        return "Upscale"
    return "Luxury"


def _rating_persona(rating, review_count_median):
    """Rating + review signal → 'Hidden Gem' / 'Tourist Favorite' / ..."""
    if pd.isna(rating):
        return None
    # High rating + low reviews = hidden gem
    # High rating + high reviews = tourist favorite
    # Low rating + high reviews = overexposed
    # Low rating + low reviews = under-the-radar
    if rating >= 4.4 and review_count_median < 150:
        return "Hidden Gem"
    if rating >= 4.4 and review_count_median >= 400:
        return "Tourist Favorite"
    if rating >= 4.2:
        return "Highly Rated"
    if rating >= 3.9:
        return "Reliable"
    if rating < 3.5 and review_count_median >= 400:
        return "Overhyped"
    return "Under-the-Radar"


def _geographic_persona(boro_share, top_boro):
    """Is this cluster geographically concentrated?"""
    if boro_share >= 0.55 and top_boro and top_boro not in ("", "Unknown", "0"):
        return top_boro
    return None


def _cuisine_persona(cuisine_counts, top_n=3):
    """
    Return (dominant_label, top_cuisines, is_mixed).
    - If a single cuisine is >= 40%, dominant_label is that cuisine.
    - If two cuisines dominate (>= 25% each), label is "A-B fusion crowd".
    - Otherwise it's "Mixed Cuisine".
    """
    if cuisine_counts.empty:
        return "Mixed Cuisine", [], True

    top = cuisine_counts.head(top_n)
    top_list = [(name, float(share)) for name, share in top.items()
                if name and str(name).strip() not in ("", "0", "Other")]

    if not top_list:
        return "Mixed Cuisine", [], True

    first_name, first_share = top_list[0]

    if first_share >= 0.40:
        # Single-cuisine dominant cluster
        return first_name, top_list, False

    if len(top_list) >= 2:
        second_name, second_share = top_list[1]
        if first_share >= 0.25 and second_share >= 0.20:
            # Genuine mix of two cuisines (e.g. American 30% + Chinese 25%)
            return f"{first_name} & {second_name}", top_list, True

    # Broadly mixed — no single cuisine carries the cluster
    return "Mixed Cuisine", top_list, True

def _build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Generate persona labels + narrative stories per cluster.
 
    Output columns (unchanged so the rest of the app still works):
        cluster_id, cluster_label, cluster_key_drivers, cluster_story
    Extra columns added for richer UI (safe to ignore in older callers):
        cluster_persona, cluster_cuisine_mix, cluster_boro_mix
    """
    # Import here to avoid a circular import with build_feature_matrix.
    from utils.clustering import build_feature_matrix  # type: ignore
 
    if df.empty:
        return pd.DataFrame(columns=[
            "cluster_id", "cluster_label", "cluster_key_drivers", "cluster_story",
            "cluster_persona", "cluster_cuisine_mix", "cluster_boro_mix",
        ])
 
    feature_matrix, feature_columns, aligned_df = build_feature_matrix(df)
    feature_df = pd.DataFrame(feature_matrix, columns=feature_columns,
                              index=aligned_df.index)
 
    # Global baselines — a cluster is "distinctive" relative to these.
    global_rating = pd.to_numeric(df["avg_rating"], errors="coerce").mean()
    global_price = pd.to_numeric(df["price_tier"], errors="coerce").mean()
    global_reviews_med = pd.to_numeric(df["review_count"], errors="coerce").median()
    global_health = pd.to_numeric(df.get("score", pd.Series(np.nan)),
                                  errors="coerce").mean()
 
    profile_rows = []
    used_labels: set[str] = set()
 
    for cluster_id in sorted(df["cluster_id"].unique()):
        cluster_df = df[df["cluster_id"] == cluster_id]
        n = len(cluster_df)
        if n == 0:
            continue
 
        # ---- Raw statistics ----
        avg_rating = pd.to_numeric(cluster_df["avg_rating"],
                                   errors="coerce").mean()
        avg_price = pd.to_numeric(cluster_df["price_tier"],
                                  errors="coerce").mean()
        review_med = pd.to_numeric(cluster_df["review_count"],
                                   errors="coerce").median()
        avg_health = pd.to_numeric(cluster_df.get("score", pd.Series(np.nan,
                                   index=cluster_df.index)), errors="coerce").mean()
 
        # ---- Cuisine mix ----
        cuisine_counts = (
            cluster_df["cuisine_type"].fillna("")
            .astype(str).str.strip()
            .replace({"0": ""})
            .value_counts(normalize=True)
        )
        cuisine_counts = cuisine_counts[cuisine_counts.index != ""]
        cuisine_persona, top_cuisines, cuisine_is_mixed = _cuisine_persona(
            cuisine_counts, top_n=3,
        )
 
        # ---- Borough concentration ----
        boro_counts = (
            cluster_df["boro"].fillna("Unknown")
            .astype(str).str.strip()
            .replace({"0": "Unknown"})
            .value_counts(normalize=True)
        )
        top_boro = boro_counts.index[0] if not boro_counts.empty else "Unknown"
        top_boro_share = float(boro_counts.iloc[0]) if not boro_counts.empty else 0.0
        geographic = _geographic_persona(top_boro_share, top_boro)
 
        # ---- Persona qualifiers ----
        price_label = _price_persona(avg_price)
        rating_label = _rating_persona(avg_rating, review_med)
 
        # ---- Assemble the label ----
        # We want the label to answer: who is this cluster for?
        # Format: "[Cuisine persona] · [Borough if concentrated] · [Price/Quality]"
        label_parts: list[str] = []
        if cuisine_persona:
            label_parts.append(cuisine_persona)
        if geographic:
            label_parts.append(geographic)
        # Pick the more distinctive of {price, quality} for the third slot.
        # Distinctiveness = how far from the global mean.
        distinctive: str | None = None
        price_delta = abs((avg_price or global_price) - global_price)
        rating_delta = abs((avg_rating or global_rating) - global_rating)
        if price_label and price_delta >= 0.25:
            distinctive = price_label
        if rating_label and rating_delta >= 0.2:
            # Rating is more evocative — override if it's distinctive too
            distinctive = rating_label
        if distinctive:
            label_parts.append(distinctive)
 
        if not label_parts:
            label_parts = [f"Cluster {cluster_id}"]
 
        label = " · ".join(label_parts[:3])
 
        # De-dupe clashing labels across clusters (happens when two clusters
        # look similar on our low-dimensional persona summary).
        if label in used_labels:
            label = f"{label} · C{cluster_id}"
        used_labels.add(label)
 
        # ---- Build a cuisine mix blurb ----
        if top_cuisines:
            cuisine_blurb = ", ".join(
                f"{name} {share * 100:.0f}%" for name, share in top_cuisines
            )
        else:
            cuisine_blurb = "mixed"
 
        # ---- Build a borough mix blurb ----
        top_boros = boro_counts.head(2)
        boro_blurb_parts = [
            f"{name} {share * 100:.0f}%" for name, share in top_boros.items()
            if name not in ("Unknown", "")
        ]
        boro_blurb = ", ".join(boro_blurb_parts) if boro_blurb_parts else "spread across NYC"
 
        # ---- Narrative story ----
        # Lead with the persona, then justify with numbers.
        story_parts: list[str] = []
 
        if cuisine_is_mixed and len(top_cuisines) >= 2:
            story_parts.append(
                f"A **mixed-cuisine** cluster: the three most common cuisines are "
                f"{cuisine_blurb} — no single cuisine carries more than "
                f"{top_cuisines[0][1] * 100:.0f}% of the cluster, so these restaurants "
                f"are grouped by shared **price, rating, and location** signals rather "
                f"than cuisine."
            )
        else:
            if top_cuisines:
                main_cuisine, main_share = top_cuisines[0]
                story_parts.append(
                    f"A **{main_cuisine}-led** cluster — {main_share * 100:.0f}% of "
                    f"restaurants here serve {main_cuisine}."
                )
 
        # Geography
        if geographic:
            story_parts.append(
                f"Geographically concentrated in **{geographic}** "
                f"({top_boro_share * 100:.0f}% of the cluster)."
            )
        elif boro_blurb_parts:
            story_parts.append(f"Borough mix: {boro_blurb}.")
 
        # Price + rating signal with direction
        price_dir = ""
        if not pd.isna(avg_price):
            if avg_price > global_price + 0.2:
                price_dir = "above-average prices"
            elif avg_price < global_price - 0.2:
                price_dir = "below-average prices"
        rating_dir = ""
        if not pd.isna(avg_rating):
            if avg_rating > global_rating + 0.15:
                rating_dir = "higher-than-average ratings"
            elif avg_rating < global_rating - 0.15:
                rating_dir = "lower-than-average ratings"
 
        signal_bits = [bit for bit in (price_dir, rating_dir) if bit]
        if signal_bits:
            story_parts.append(
                f"These restaurants show {' and '.join(signal_bits)} "
                f"(avg rating {avg_rating:.2f}, avg price tier {avg_price:.2f} vs "
                f"city averages {global_rating:.2f} / {global_price:.2f})."
            )
        else:
            story_parts.append(
                f"Operating near NYC averages on price and rating "
                f"(avg rating {avg_rating:.2f}, avg price tier {avg_price:.2f})."
            )
 
        # Health context (only if meaningfully different or of interest)
        if not pd.isna(avg_health) and not pd.isna(global_health):
            if avg_health > global_health + 3:
                story_parts.append(
                    f"Slightly **higher DOHMH inspection scores** (avg "
                    f"{avg_health:.1f}, worse than the citywide "
                    f"{global_health:.1f}) — worth checking individual grades."
                )
            elif avg_health < global_health - 3:
                story_parts.append(
                    f"**Cleaner on health inspections** (avg score "
                    f"{avg_health:.1f} vs city {global_health:.1f}, lower = better)."
                )
 
        story = " ".join(story_parts)
 
        # ---- Key drivers — short enough for a chip, derived from persona pieces ----
        drivers = [p for p in label_parts if not re.match(r"^Cluster\s*\d+$", p)]
        if not drivers:
            drivers = ["Balanced mix"]
        key_drivers = " | ".join(drivers[:3])
 
        profile_rows.append({
            "cluster_id": cluster_id,
            "cluster_label": label,
            "cluster_key_drivers": key_drivers,
            "cluster_story": story,
            # Extras for UI use:
            "cluster_persona": cuisine_persona,
            "cluster_cuisine_mix": cuisine_blurb,
            "cluster_boro_mix": boro_blurb,
        })
 
    return pd.DataFrame(profile_rows)
 

def _attach_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profiles = _build_cluster_profiles(df)
    if profiles.empty:
        df = df.copy()
        df["cluster_label"] = "Cluster"
        df["cluster_key_drivers"] = ""
        df["cluster_story"] = ""
        return df
    return df.merge(profiles, on="cluster_id", how="left")


def _assign_cluster_labels(df: pd.DataFrame):
    """Backward-compatible wrapper around centroid-based profile labels."""
    profiles = _build_cluster_profiles(df)
    if profiles.empty:
        return pd.Series(["Cluster"] * len(df), index=df.index)
    return df["cluster_id"].map(profiles.set_index("cluster_id")["cluster_label"])


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
    affinity = compute_user_affinity(X, df, user_history)
    return np.hstack([X, affinity.reshape(-1, 1)])


def compute_user_affinity(X: np.ndarray, df: pd.DataFrame, user_history: dict):
    visited_ids = user_history.get("visited_ids", [])
    rated       = user_history.get("rated", {})

    if not visited_ids:
        return np.zeros(len(df), dtype=np.float32)

    visited_mask = df["restaurant_id"].isin(visited_ids)
    visited_X    = X[visited_mask.values]

    if len(visited_X) == 0:
        return np.zeros(len(df), dtype=np.float32)

    # Weighted mean of visited restaurants by rating
    weights = []
    for rid in df.loc[visited_mask, "restaurant_id"]:
        weights.append(rated.get(rid, 3.0) / 5.0)
    weights = np.array(weights).reshape(-1, 1)
    user_vec = (visited_X * weights).sum(axis=0, keepdims=True) / (weights.sum() + 1e-8)

    affinity = cosine_similarity(X, user_vec).flatten().astype(np.float32)
    return affinity


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
    sample = []
    if "restaurant_id" in df.columns and len(df):
        sample = sorted(df["restaurant_id"].astype(str).head(25).tolist())
    history_payload = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "algorithm": algorithm,
        "k": int(k),
        "row_count": int(len(df)),
        "restaurant_sample": sample,
    }
    payload = json.dumps(history_payload, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_optimal_k(X_scaled: np.ndarray, k_range=range(4, 16),
                   algorithm: str = "kmeans") -> int:
    best_k, best_score = 8, -1
    for k in k_range:
        if k >= len(X_scaled):
            break
        if algorithm == "kmeans":
            model = KMeansScratch(n_clusters=k, n_init=6, max_iter=200, random_state=42)
            labels = model.fit_predict(X_scaled)
        elif algorithm == "gmm":
            model = GaussianMixture(
                n_components=k,
                covariance_type="tied",
                random_state=42,
                n_init=3,
                max_iter=150,
                reg_covar=1e-4,
            )
            labels = model.fit_predict(X_scaled)
        elif algorithm == "agglomerative":
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown clustering algorithm: {algorithm!r}")
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X_scaled, labels, sample_size=min(1000, len(X_scaled)), random_state=42)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _reindex_labels_and_centroids(labels: np.ndarray, X_cluster: np.ndarray):
    unique_labels = sorted(np.unique(labels).tolist())
    remap = {old: new for new, old in enumerate(unique_labels)}
    dense_labels = np.array([remap[label] for label in labels], dtype=np.int64)
    centroids = np.vstack([
        X_cluster[dense_labels == cid].mean(axis=0)
        for cid in range(len(unique_labels))
    ]).astype(np.float32)
    return dense_labels, centroids


def run_kmeans(df: pd.DataFrame, user_history: dict, k: int = 10):
    X, feature_columns, df = build_feature_matrix(df)
    user_affinity = compute_user_affinity(X, df, user_history)
    projection_feature_columns = feature_columns

    X_scaled, X_cluster, scaler = prepare_clustering_space(X, fit=True)

    max_clusters = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_clusters))

    df = df.copy()
    best_model = None
    best_labels = None
    best_score = -1
    candidate_seeds = [42, 52, 62, 72, 82]
    for seed in candidate_seeds:
        candidate_model = KMeansScratch(
            n_clusters=k,
            n_init=8,
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

    kmeans = best_model or KMeansScratch(
        n_clusters=k,
        n_init=8,
        max_iter=300,
        random_state=42,
    )
    if best_model is None:
        best_labels = kmeans.fit_predict(X_cluster)
    else:
        best_labels = best_labels if best_labels is not None else best_model.labels_
        kmeans = best_model
    df["cluster_id"] = best_labels

    # Merge undersized clusters (< 3% of total) into the nearest larger cluster
    # by centroid distance. Prevents tiny degenerate groups and "catch-all" patterns
    # where one cluster absorbs everything that didn't fit elsewhere.
    centroids = kmeans.cluster_centers_
    min_cluster_size = max(1, int(round(len(df) * 0.03)))
    while True:
        cluster_sizes = df["cluster_id"].value_counts()
        small_cluster_ids = cluster_sizes[cluster_sizes < min_cluster_size].index.tolist()
        large_cluster_ids = cluster_sizes[cluster_sizes >= min_cluster_size].index.tolist()
        if not small_cluster_ids or not large_cluster_ids:
            break
        # Merge the smallest cluster into the nearest large cluster by centroid distance
        smallest_cid = cluster_sizes.idxmin()
        dists = np.linalg.norm(centroids - centroids[smallest_cid], axis=1)
        dists[smallest_cid] = np.inf
        for cid in range(len(centroids)):
            if cid not in large_cluster_ids:
                dists[cid] = np.inf
        target_cid = int(np.argmin(dists))
        df.loc[df["cluster_id"] == smallest_cid, "cluster_id"] = target_cid

    final_labels, final_centroids = _reindex_labels_and_centroids(
        df["cluster_id"].to_numpy(dtype=np.int64), X_cluster
    )
    df["cluster_id"] = final_labels
    kmeans.labels_ = final_labels
    kmeans.cluster_centers_ = final_centroids
    kmeans.n_clusters = int(len(final_centroids))
    kmeans.inertia_ = float(np.sum((X_cluster - final_centroids[final_labels]) ** 2))
    kmeans.silhouette_score_ = best_score if best_score >= 0 else None

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
    pca.feature_columns_ = projection_feature_columns

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

    df = _attach_cluster_profiles(df)
    df["user_affinity_score"] = user_affinity

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
                        user_affinity: np.ndarray, projection_feature_columns: list):
    """Attach cluster_id + PCA / t-SNE / UMAP / cluster-view projections.

    Shared post-processing for non-KMeans algorithms that don't have a
    native ``transform()``.  The centroid-to-point distance matrix is
    computed via broadcasting.
    """
    df = df.copy()
    df["cluster_id"] = labels

    _merge_small_clusters(df, centroids)
    final_labels, final_centroids = _reindex_labels_and_centroids(
        df["cluster_id"].to_numpy(dtype=np.int64), X_cluster
    )
    df["cluster_id"] = final_labels

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
    pca.feature_columns_ = projection_feature_columns

    # Distance from each point to each centroid (analogue of kmeans.transform).
    centroid_distances = np.linalg.norm(
        X_cluster[:, None, :] - final_centroids[None, :, :], axis=2
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

    df = _attach_cluster_profiles(df)
    df["user_affinity_score"] = user_affinity

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

    return df, pca, final_centroids


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
    user_affinity = compute_user_affinity(X, df, user_history)
    projection_feature_columns = feature_columns

    X_scaled, X_cluster, scaler = prepare_clustering_space(X, fit=True)

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

    df, pca, final_centroids = _finalize_clusters(
        df, labels, centroids, X_scaled, X_cluster, user_affinity, projection_feature_columns
    )
    wrapper = _CentroidClusteringModel(
        final_centroids,
        df["cluster_id"].to_numpy(dtype=np.int64),
        "gmm",
        base_model=gmm,
    )
    return df, wrapper, scaler, pca


def run_agglomerative(df: pd.DataFrame, user_history: dict, k: int = 10):
    """Agglomerative (Ward-linkage) hierarchical clustering.

    Ward minimises within-cluster variance at each merge, producing
    compact roughly-spherical clusters without the k-means init-sensitivity.
    Centroids are computed post-hoc as per-cluster means.
    """
    X, feature_columns, df = build_feature_matrix(df)
    user_affinity = compute_user_affinity(X, df, user_history)
    projection_feature_columns = feature_columns

    X_scaled, X_cluster, scaler = prepare_clustering_space(X, fit=True)

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

    df, pca, final_centroids = _finalize_clusters(
        df, labels, centroids, X_scaled, X_cluster, user_affinity, projection_feature_columns
    )
    wrapper = _CentroidClusteringModel(
        final_centroids,
        df["cluster_id"].to_numpy(dtype=np.int64),
        "agglomerative",
        base_model=agg,
    )
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
    X_scaled, X_cluster, _s = prepare_clustering_space(X, fit=True)
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


def _attach_user_context(df: pd.DataFrame, user_history: dict):
    """Keep clustering global/stable while computing user affinity per session."""
    if df.empty:
        return df.copy()

    df = df.copy()
    feature_matrix, _feature_columns, aligned_df = build_feature_matrix(df)
    affinity = compute_user_affinity(feature_matrix, aligned_df, user_history)
    df["user_affinity_score"] = affinity
    return df


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
                return _attach_user_context(cached_df, user_history), cached_model, cached_scaler, cached_pca
        except Exception:
            # Corrupt or schema-mismatched cache — fall through to recompute.
            pass
    result_df, model, scaler, pca = run_clustering(df, user_history, k, algorithm)
    save_cache(result_df, model, scaler, pca, signature, algorithm)
    return _attach_user_context(result_df, user_history), model, scaler, pca


# ---------------------------------------------------------------------------
# K-NN Recommendation Engine
# ---------------------------------------------------------------------------

BUDGET_TO_PRICE = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
BOROUGH_LIST = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]


def build_user_feature_vector(profile: dict, restaurant_df: pd.DataFrame) -> np.ndarray:
    """Construct a user feature vector in the same space as restaurant features.

    Differences from the previous version:
    - Cuisine preferences are expressed as *presence indicators* (max-scaled to
      1.0), not a normalized probability.  If the user likes Chinese and only
      Chinese, the Chinese dimension is 1.0 — matching a Chinese restaurant row
      exactly instead of being divided by the number of likes.
    - Explicit favorite_cuisines override liked-cuisine averages so the user's
      stated preferences are never diluted.
    """
    from utils.clustering import TOP_CUISINES, BOROUGH_LIST, BUDGET_TO_PRICE  # type: ignore

    liked_restaurants = profile.get("likes", [])
    liked_ids = {str(l.get("restaurant_id", "")) for l in liked_restaurants
                 if l.get("restaurant_id")}

    # --- Price from budget preference ---
    budget_str = profile.get("budget", "$$")
    price_pref = BUDGET_TO_PRICE.get(budget_str, 2)
    price_norm = (price_pref - 1) / 3.0

    fav_cuisines = profile.get("favorite_cuisines", [])
    pref_boroughs = profile.get("preferred_boroughs", [])

    # --- Stats from liked restaurants (if any) ---
    liked_mask = restaurant_df["restaurant_id"].astype(str).isin(liked_ids)
    liked_df = restaurant_df[liked_mask]

    if len(liked_df) > 0:
        avg_rating_norm = ((liked_df["avg_rating"].mean() - 1.0) / 4.0)
        log_reviews = np.log1p(liked_df["review_count"].values)
        log_all = np.log1p(restaurant_df["review_count"].fillna(0).values)
        log_min, log_max = log_all.min(), log_all.max()
        avg_review_norm = ((log_reviews.mean() - log_min) /
                           (log_max - log_min)) if log_max > log_min else 0.5
        health_series = pd.to_numeric(liked_df["score"],
                                      errors="coerce").fillna(21).clip(0, 42)
        avg_health_norm = 1 - (health_series.mean() / 42.0)
        avg_lat = liked_df["lat"].mean()
        avg_lng = liked_df["lng"].mean()
    else:
        avg_rating_norm = 0.5
        avg_review_norm = 0.5
        avg_health_norm = 0.7
        avg_lat = restaurant_df["lat"].median()
        avg_lng = restaurant_df["lng"].median()

    # Lat/lng normalization in the same range as build_feature_matrix
    lat_all = pd.to_numeric(restaurant_df["lat"], errors="coerce").fillna(
        restaurant_df["lat"].median())
    lng_all = pd.to_numeric(restaurant_df["lng"], errors="coerce").fillna(
        restaurant_df["lng"].median())
    lat_min, lat_max = lat_all.min(), lat_all.max()
    lng_min, lng_max = lng_all.min(), lng_all.max()
    lat_norm = ((avg_lat - lat_min) / (lat_max - lat_min)
                if lat_max > lat_min else 0.5)
    lng_norm = ((avg_lng - lng_min) / (lng_max - lng_min)
                if lng_max > lng_min else 0.5)

    # --- Cuisine one-hot ---
    # We encode *which* cuisines the user prefers as ones (0/1), not a
    # normalized probability.  This matches the one-hot shape of a
    # restaurant row, which is the key to making cosine similarity do the
    # right thing.  If the user likes multiple cuisines, multiple dimensions
    # are 1 — their taste is genuinely multi-modal and the recommender
    # surfaces both.
    cuisine_vec = np.zeros(len(TOP_CUISINES) + 1, dtype=np.float32)

    # Explicit preferences take priority (they're what the user stated).
    for c in fav_cuisines:
        if c in TOP_CUISINES:
            cuisine_vec[TOP_CUISINES.index(c)] = 1.0
        else:
            cuisine_vec[-1] = 1.0  # Other

    # Likes add to the signal but only flip dimensions to 1 (never normalize).
    if len(liked_df) > 0:
        liked_cuisine_counts = liked_df["cuisine_type"].fillna("Other").value_counts()
        # Require at least 2 likes (or ≥ 25% of all likes) before a cuisine
        # from liked history gets an auto-preference.  This prevents a single
        # outlier like from poisoning the user's taste vector.
        min_count = max(2, int(len(liked_df) * 0.25))
        for cuisine, count in liked_cuisine_counts.items():
            if count >= min_count:
                if cuisine in TOP_CUISINES:
                    cuisine_vec[TOP_CUISINES.index(cuisine)] = 1.0
                else:
                    cuisine_vec[-1] = 1.0  # Other

    # --- Borough one-hot (same 0/1 treatment) ---
    boro_vec = np.zeros(len(BOROUGH_LIST), dtype=np.float32)
    for b in pref_boroughs:
        if b in BOROUGH_LIST:
            boro_vec[BOROUGH_LIST.index(b)] = 1.0
    if len(liked_df) > 0:
        liked_boro_counts = liked_df["boro"].fillna("").value_counts()
        min_count = max(2, int(len(liked_df) * 0.3))
        for boro, count in liked_boro_counts.items():
            if count >= min_count and boro in BOROUGH_LIST:
                boro_vec[BOROUGH_LIST.index(boro)] = 1.0

    # --- Assemble (must match build_feature_matrix's order AND weights) ---
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
                  scaler: StandardScaler | None = None,
                  profile: dict | None = None) -> pd.DataFrame:
    """Find the K nearest restaurants to the user vector using cosine similarity.

    With an optional `profile`, a cuisine-alignment boost is applied so users
    with stated or implicit cuisine preferences see matching restaurants
    ranked higher.
    """
    X_scaled = _scaled_space(restaurant_matrix, scaler)
    u_scaled = _scaled_space(user_vector, scaler)
    similarities = cosine_similarity(u_scaled, X_scaled).flatten()

    if profile is not None:
        liked_ids_for_cuisine = {
            str(l.get("restaurant_id", ""))
            for l in profile.get("likes", [])
            if l.get("restaurant_id")
        }
        liked_for_cuisine = restaurant_df[
            restaurant_df["restaurant_id"].astype(str).isin(liked_ids_for_cuisine)
        ]
        boost = cuisine_alignment_score(
            profile, restaurant_df["cuisine_type"], liked_for_cuisine,
        )
        similarities = similarities * boost

    visited_mask = restaurant_df["restaurant_id"].astype(str).isin(visited_ids)
    similarities[visited_mask.values] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:k]
    result = restaurant_df.iloc[top_indices].copy()
    result["knn_similarity"] = similarities[top_indices]
    return result.reset_index(drop=True)


def _scaled_space(vectors: np.ndarray, scaler: StandardScaler | None):
    """Apply the cluster scaler to 22-dim feature vectors.

    Supports both:
    - current 22-dim clustering space
    - legacy 23-dim [features + user_affinity] scaler artifacts
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if scaler is None:
        return vectors

    expected_dim = getattr(scaler, "n_features_in_", vectors.shape[1])
    if expected_dim == vectors.shape[1]:
        return scaler.transform(vectors)
    if expected_dim == vectors.shape[1] + 1:
        n = vectors.shape[0]
        aug = np.hstack([vectors, np.full((n, 1), 0.5, dtype=np.float32)])
        return scaler.transform(aug)
    raise ValueError(
        f"Scaler expects {expected_dim} features but received {vectors.shape[1]}"
    )


# --------------------------------------------------------------------------
# New helper: cuisine alignment score (0 → 1+)
# --------------------------------------------------------------------------

def cuisine_alignment_score(profile: dict, cuisine_series: pd.Series,
                            liked_df: pd.DataFrame | None = None) -> np.ndarray:
    """
    Return a per-restaurant score in [0, 1] describing how well its cuisine
    matches the user's taste.

    Rules (in order of priority):
      - If the user has any explicit favorite_cuisines → match is 1.0, miss is 0.15
        (we keep a small floor so we can still see other cuisines if not enough
         matches exist).
      - Else if the user's liked restaurants have a dominant cuisine family
        (≥ 50% of likes in one cuisine) → match is 1.0, miss is 0.3.
      - Else (no strong cuisine signal) → everything gets 1.0 (no boost).
    """
    cuisines = cuisine_series.fillna("Other").astype(str).str.strip()
    n = len(cuisines)

    fav_cuisines = [c for c in profile.get("favorite_cuisines", []) if c]
    if fav_cuisines:
        # Exact match against the user's stated cuisines
        match = cuisines.isin(fav_cuisines).values.astype(np.float32)
        return np.where(match, 1.0, 0.15).astype(np.float32)

    if liked_df is not None and len(liked_df) >= 2:
        liked_cuisines = liked_df["cuisine_type"].fillna("Other").value_counts(normalize=True)
        if not liked_cuisines.empty and float(liked_cuisines.iloc[0]) >= 0.5:
            dominant = liked_cuisines.index[0]
            match = (cuisines == dominant).values.astype(np.float32)
            return np.where(match, 1.0, 0.30).astype(np.float32)

    return np.ones(n, dtype=np.float32)


# --------------------------------------------------------------------------
# Replacement for recommend_per_liked_knn
# --------------------------------------------------------------------------

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
    profile: dict | None = None,
) -> pd.DataFrame:
    """K-NN recommendation with per-liked fusion + cuisine alignment boost.

    New behaviour:
      - A multiplicative cuisine-alignment term is applied to the fused RRF
        score, so restaurants whose cuisine matches the user's stated or
        implicit preferences are ranked higher than equally-similar misses.
      - When the user has explicit favorite_cuisines, non-matching cuisines
        are heavily deprioritised (score × 0.15) so Chinese-preferring users
        see Chinese restaurants first, not Thai restaurants that happen to
        share a similar price tier.
    """
    from utils.clustering import _scaled_space  # type: ignore

    visited_set = {str(v) for v in visited_ids}
    restaurant_ids = restaurant_df["restaurant_id"].astype(str).values
    visited_mask = np.array([rid in visited_set for rid in restaurant_ids])

    X_scaled = _scaled_space(restaurant_matrix, scaler)
    profile_scaled = _scaled_space(profile_vector, scaler)
    profile_sim = cosine_similarity(profile_scaled, X_scaled).flatten()

    # --- Cuisine alignment based on profile ---
    if profile is not None:
        liked_ids_for_cuisine = {str(l.get("restaurant_id", ""))
                                 for l in profile.get("likes", [])
                                 if l.get("restaurant_id")}
        liked_for_cuisine = restaurant_df[
            restaurant_df["restaurant_id"].astype(str).isin(liked_ids_for_cuisine)
        ]
        cuisine_boost = cuisine_alignment_score(
            profile, restaurant_df["cuisine_type"], liked_for_cuisine,
        )
    else:
        cuisine_boost = np.ones(len(restaurant_df), dtype=np.float32)

    # --- Fallback: no likes, rank by profile similarity alone (+ cuisine boost) ---
    if liked_vectors is None or len(liked_vectors) == 0:
        sims = profile_sim.copy() * cuisine_boost
        sims[visited_mask] = -np.inf
        top = np.argsort(sims)[::-1][:k_final]
        result = restaurant_df.iloc[top].copy()
        result["knn_similarity"] = profile_sim[top]
        result["rrf_score"] = sims[top]
        result["cuisine_boost"] = cuisine_boost[top]
        result["primary_influencer"] = "Profile preferences"
        return result.reset_index(drop=True)

    liked_scaled = _scaled_space(np.asarray(liked_vectors, dtype=np.float32), scaler)
    per_liked_sims = cosine_similarity(liked_scaled, X_scaled)  # (L, N)

    rrf_scores = np.zeros(len(restaurant_df), dtype=np.float64)
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

    rrf_scores += 0.1 * profile_sim
    # Apply multiplicative cuisine boost — this is the key fix.
    rrf_scores = rrf_scores * cuisine_boost
    rrf_scores[visited_mask] = -np.inf

    top_indices = np.argsort(rrf_scores)[::-1][:k_final]
    top_indices = [i for i in top_indices if rrf_scores[i] > -np.inf]

    result = restaurant_df.iloc[top_indices].copy()
    result["knn_similarity"] = profile_sim[top_indices]
    result["rrf_score"] = rrf_scores[top_indices]
    result["cuisine_boost"] = cuisine_boost[top_indices]

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
