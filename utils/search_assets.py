"""
utils/search_assets.py — Prepared search data loader and runtime asset manager.

Manages the lifecycle of prepared restaurant datasets: loading from cache,
enriching via Google Places, building descriptions, and computing embeddings.

Original author: Rahul Adusumalli
Integrated by: Ryan Han (PapTR)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data import load_nyc_base_safe
from utils.google_places import get_enriched_restaurants, get_google_api_key
from utils.search import build_description, get_embeddings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_SEARCH_SAMPLE_SIZE = 750


def _prepared_df_path(sample_size):
    return CACHE_DIR / f"prepared_search_{sample_size}.pkl"


def _embedding_cache_key(sample_size, row_count):
    return f"prepared_{sample_size}_{row_count}"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_prepared_search_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=False):
    _ensure_cache_dir()
    prepared_df_path = _prepared_df_path(sample_size)

    if prepared_df_path.exists() and not force_refresh:
        try:
            prepared_df = pd.read_pickle(prepared_df_path)
            if not prepared_df.empty:
                embeddings = get_embeddings(prepared_df, _embedding_cache_key(sample_size, len(prepared_df)))
                return prepared_df, embeddings, {"prepared": True, "sample_size": sample_size, "rows": len(prepared_df)}
        except Exception:
            pass

    api_key = get_google_api_key()
    base_df = load_nyc_base_safe(limit=8000)

    if base_df.empty:
        return pd.DataFrame(), None, {"prepared": False, "sample_size": sample_size}

    enriched_df = get_enriched_restaurants(base_df, sample_size, api_key, force_refresh=force_refresh)
    if enriched_df.empty:
        return pd.DataFrame(), None, {"prepared": False, "sample_size": sample_size}

    prepared_df = enriched_df.copy()
    prepared_df["description"] = prepared_df.apply(build_description, axis=1)
    prepared_df.to_pickle(prepared_df_path)
    embeddings = get_embeddings(prepared_df, _embedding_cache_key(sample_size, len(prepared_df)))
    return prepared_df, embeddings, {"prepared": True, "sample_size": sample_size, "rows": len(prepared_df)}


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
    runtime_df["neighborhood"] = runtime_df.get("boro", pd.Series(["NYC"] * len(runtime_df), index=runtime_df.index))
    return runtime_df


def load_runtime_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=False):
    prepared_df, embeddings, cache_info = load_prepared_search_assets(
        sample_size=sample_size,
        force_refresh=force_refresh,
    )
    runtime_df = build_runtime_restaurant_df(prepared_df)
    return prepared_df, embeddings, runtime_df, cache_info
