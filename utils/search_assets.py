from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data import load_nyc_base_safe
from utils.google_places import get_enriched_restaurants, get_google_api_key
from utils.search import build_description, get_embeddings, neighborhood_from_zipcode

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_SEARCH_SAMPLE_SIZE = 750
BASE_DATASET_LIMIT = 8000
PREPARED_DATASET_VERSION = 3


def _prepared_df_path(sample_size):
    return CACHE_DIR / f"prepared_search_v{PREPARED_DATASET_VERSION}_{sample_size}.pkl"


def _embedding_cache_key(sample_size, row_count):
    return f"prepared_v{PREPARED_DATASET_VERSION}_{sample_size}_{row_count}"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_prepared_search_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=False):
    _ensure_cache_dir()
    prepared_df_path = _prepared_df_path(sample_size)
    base_df = load_nyc_base_safe(limit=BASE_DATASET_LIMIT)

    if base_df.empty:
        return pd.DataFrame(), None, {"prepared": False, "sample_size": sample_size}

    if prepared_df_path.exists() and not force_refresh:
        try:
            prepared_df = pd.read_pickle(prepared_df_path)
            if not prepared_df.empty:
                embeddings = get_embeddings(prepared_df, _embedding_cache_key(sample_size, len(prepared_df)))
                enriched_rows = prepared_df["g_place_id"].notna().sum() if "g_place_id" in prepared_df.columns else 0
                return prepared_df, embeddings, {
                    "prepared": True,
                    "sample_size": sample_size,
                    "rows": len(prepared_df),
                    "base_rows": len(prepared_df),
                    "enriched_rows": int(enriched_rows),
                }
        except Exception:
            pass

    api_key = get_google_api_key()
    enriched_df = get_enriched_restaurants(base_df, sample_size, api_key, force_refresh=force_refresh)
    prepared_df = base_df.copy()

    if not enriched_df.empty:
        enrichment_columns = [
            "camis",
            "g_rating",
            "g_reviews",
            "g_price",
            "g_summary",
            "g_photo_ref",
            "g_maps_url",
            "g_place_id",
        ]
        available_columns = [column for column in enrichment_columns if column in enriched_df.columns]
        enriched_subset = (
            enriched_df[available_columns]
            .drop_duplicates(subset=["camis"], keep="first")
        )
        prepared_df = prepared_df.merge(enriched_subset, on="camis", how="left")

    required_google_columns = ["g_rating", "g_reviews", "g_price", "g_place_id"]
    available_required_columns = [column for column in required_google_columns if column in prepared_df.columns]
    if available_required_columns:
        prepared_df = prepared_df.dropna(subset=available_required_columns).reset_index(drop=True)

    prepared_df["neighborhood"] = prepared_df.get("zipcode", pd.Series([""] * len(prepared_df), index=prepared_df.index)).apply(neighborhood_from_zipcode)
    prepared_df["description"] = prepared_df.apply(build_description, axis=1)
    prepared_df.to_pickle(prepared_df_path)
    embeddings = get_embeddings(prepared_df, _embedding_cache_key(sample_size, len(prepared_df)))
    enriched_rows = prepared_df["g_place_id"].notna().sum() if "g_place_id" in prepared_df.columns else 0
    return prepared_df, embeddings, {
        "prepared": True,
        "sample_size": sample_size,
        "rows": len(prepared_df),
        "base_rows": len(prepared_df),
        "enriched_rows": int(enriched_rows),
    }


def warm_search_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE):
    return load_prepared_search_assets(sample_size=sample_size, force_refresh=False)


def build_runtime_restaurant_df(prepared_df):
    if prepared_df.empty:
        return prepared_df.copy()

    runtime_df = prepared_df.copy()
    runtime_df["restaurant_id"] = runtime_df["camis"].astype(str)
    runtime_df["name"] = runtime_df["dba"]
    runtime_df["lat"] = pd.to_numeric(runtime_df.get("lat", runtime_df.get("latitude")), errors="coerce")
    runtime_df["lng"] = pd.to_numeric(runtime_df.get("lon", runtime_df.get("longitude")), errors="coerce")
    runtime_df["cuisine_type"] = runtime_df["cuisine"]
    runtime_df["price_tier"] = pd.to_numeric(runtime_df.get("g_price", 2), errors="coerce").fillna(2).clip(1, 4).astype(int)
    runtime_df["avg_rating"] = pd.to_numeric(runtime_df.get("g_rating", 3.0), errors="coerce").fillna(3.0)
    runtime_df["review_count"] = pd.to_numeric(runtime_df.get("g_reviews", 0), errors="coerce").fillna(0).astype(int)
    if "neighborhood" not in runtime_df.columns:
        runtime_df["neighborhood"] = runtime_df.get("zipcode", pd.Series([""] * len(runtime_df), index=runtime_df.index)).apply(neighborhood_from_zipcode)
    return runtime_df


def load_runtime_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=False):
    prepared_df, embeddings, cache_info = load_prepared_search_assets(
        sample_size=sample_size,
        force_refresh=force_refresh,
    )
    runtime_df = build_runtime_restaurant_df(prepared_df)
    return prepared_df, embeddings, runtime_df, cache_info
