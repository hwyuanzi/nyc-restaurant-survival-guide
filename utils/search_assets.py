"""
Bug fix for load_prepared_search_assets in utils/search_assets.py.

The original function fetches the DOHMH "base" DataFrame from the NYC Open
Data live API on every app start, and only then looks at the on-disk
prepared cache.  If the API fails for any reason — network down, rate limit,
Socrata outage — the function returns an empty DataFrame even though a
perfectly good 2,835-restaurant `prepared_search_v4_3800.pkl` is sitting in
`data/cache/`.

This cascades into two visible errors:
  1. Page 1 (Semantic Search) shows "No restaurants returned valid Google
     Places data. Check your API key and quota."
  2. Page 3 (Restaurant Cluster Map) crashes with
     `KeyError: 'restaurant_id'` because the empty DataFrame from step 1
     gets written into `session_state['raw_df']` and every downstream page
     treats that as valid data.

Fix: check the prepared cache BEFORE hitting the live API.  Only fall
through to the live API if the cache is missing or stale.

Replace the body of load_prepared_search_assets in utils/search_assets.py
with the version below.  No other changes needed.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data import load_nyc_base_safe, normalize_borough_series
from utils.google_places import get_enriched_restaurants, get_google_api_key
from utils.search import build_description, get_embeddings, neighborhood_from_zipcode

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_SEARCH_SAMPLE_SIZE = 3800
BASE_DATASET_LIMIT = 8000
PREPARED_DATASET_VERSION = 4


def _prepared_df_path(sample_size):
    return CACHE_DIR / f"prepared_search_v{PREPARED_DATASET_VERSION}_{sample_size}.pkl"


def _embedding_cache_key(sample_size, row_count):
    return f"prepared_v{PREPARED_DATASET_VERSION}_{sample_size}_{row_count}"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_from_cache(sample_size):
    """Try to load the prepared DataFrame from the on-disk pkl.

    Returns (prepared_df, embeddings, cache_info) on success or None on any
    failure (missing file, read error, empty DataFrame).
    """
    prepared_df_path = _prepared_df_path(sample_size)
    if not prepared_df_path.exists():
        return None

    try:
        prepared_df = pd.read_pickle(prepared_df_path)
    except Exception:
        return None

    if prepared_df is None or prepared_df.empty:
        return None

    try:
        embeddings = get_embeddings(
            prepared_df, _embedding_cache_key(sample_size, len(prepared_df)),
        )
    except Exception:
        embeddings = None

    enriched_rows = (
        int(prepared_df["g_place_id"].notna().sum())
        if "g_place_id" in prepared_df.columns else 0
    )
    return prepared_df, embeddings, {
        "prepared": True,
        "sample_size": sample_size,
        "rows": len(prepared_df),
        "base_rows": len(prepared_df),
        "enriched_rows": enriched_rows,
        "source": "cache",
    }


def load_prepared_search_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=False):
    """Load the prepared restaurant + embedding assets.

    Priority order:
      1. If a fresh-enough pkl cache exists AND force_refresh is False → use it.
         This makes the app work offline and gracefully survives NYC Open
         Data or Google Places API outages.
      2. Otherwise try to build the DataFrame from the live DOHMH API +
         Google Places enrichment.
      3. If step 2 fails (e.g., no network) but an on-disk pkl exists →
         fall back to the pkl regardless of whether force_refresh was set,
         so the app still loads instead of presenting an empty-data error.
    """
    _ensure_cache_dir()

    # Step 1 — prefer the cache on cold starts.
    if not force_refresh:
        cached = _load_from_cache(sample_size)
        if cached is not None:
            return cached

    # Step 2 — rebuild from the live API.
    prepared_df_path = _prepared_df_path(sample_size)
    base_df = load_nyc_base_safe(limit=BASE_DATASET_LIMIT)

    if base_df.empty:
        # Step 3 — last-ditch fallback to the cache even if force_refresh
        # was requested.  Better to show stale data than nothing at all.
        cached = _load_from_cache(sample_size)
        if cached is not None:
            st.warning(
                "Could not reach the NYC Open Data API — serving the latest "
                "cached snapshot instead.  Re-run with 'Refresh cached sample' "
                "once the API is reachable to update."
            )
            return cached
        return pd.DataFrame(), None, {"prepared": False, "sample_size": sample_size}

    api_key = get_google_api_key()
    enriched_df = get_enriched_restaurants(base_df, sample_size, api_key, force_refresh=force_refresh)
    prepared_df = base_df.copy()

    if enriched_df.empty:
        # Same idea — if we got DOHMH rows but Places enrichment failed
        # (missing key, quota exhausted), fall back to the last good cache.
        cached = _load_from_cache(sample_size)
        if cached is not None:
            st.warning(
                "Google Places enrichment returned no data — serving the "
                "latest cached snapshot.  Check your GOOGLE_API_KEY and "
                "Places quota, then re-run with 'Refresh cached sample'."
            )
            return cached
        return pd.DataFrame(), None, {"prepared": False, "sample_size": sample_size}

    enrichment_columns = [
        "camis",
        "g_rating", "g_reviews", "g_price",
        "g_summary", "g_photo_ref", "g_maps_url", "g_place_id",
    ]
    available_columns = [c for c in enrichment_columns if c in enriched_df.columns]
    enriched_subset = (
        enriched_df[available_columns]
        .drop_duplicates(subset=["camis"], keep="first")
    )
    prepared_df = prepared_df.merge(enriched_subset, on="camis", how="left")

    required_google_columns = ["g_rating", "g_reviews", "g_price", "g_place_id"]
    available_required = [c for c in required_google_columns if c in prepared_df.columns]
    if available_required:
        prepared_df = prepared_df.dropna(subset=available_required).reset_index(drop=True)

    prepared_df["neighborhood"] = prepared_df.get(
        "zipcode", pd.Series([""] * len(prepared_df), index=prepared_df.index)
    ).apply(neighborhood_from_zipcode)
    prepared_df["description"] = prepared_df.apply(build_description, axis=1)
    prepared_df.to_pickle(prepared_df_path)

    embeddings = get_embeddings(
        prepared_df, _embedding_cache_key(sample_size, len(prepared_df)),
    )
    enriched_rows = (
        int(prepared_df["g_place_id"].notna().sum())
        if "g_place_id" in prepared_df.columns else 0
    )
    return prepared_df, embeddings, {
        "prepared": True,
        "sample_size": sample_size,
        "rows": len(prepared_df),
        "base_rows": len(prepared_df),
        "enriched_rows": enriched_rows,
        "source": "live",
    }


# --------------------------------------------------------------------------
# The other functions in search_assets.py stay exactly as-is.  Only replace
# load_prepared_search_assets in the existing file.
# --------------------------------------------------------------------------

def warm_search_assets(sample_size=DEFAULT_SEARCH_SAMPLE_SIZE):
    return load_prepared_search_assets(sample_size=sample_size, force_refresh=False)


def build_runtime_restaurant_df(prepared_df):
    if prepared_df.empty:
        return prepared_df.copy()

    runtime_df = prepared_df.copy()
    if "boro" in runtime_df.columns:
        runtime_df["boro"] = normalize_borough_series(runtime_df["boro"])
    runtime_df["restaurant_id"] = runtime_df["camis"].astype(str)
    runtime_df["name"] = runtime_df["dba"]
    runtime_df["lat"] = pd.to_numeric(runtime_df.get("lat", runtime_df.get("latitude")), errors="coerce")
    runtime_df["lng"] = pd.to_numeric(runtime_df.get("lon", runtime_df.get("longitude")), errors="coerce")
    runtime_df["cuisine_type"] = runtime_df["cuisine"]
    g_price_series = runtime_df.get("g_price", pd.Series(2, index=runtime_df.index))
    runtime_df["price_tier"] = pd.to_numeric(g_price_series, errors="coerce").fillna(2).clip(1, 4).astype(int)
    
    g_rating_series = runtime_df.get("g_rating", pd.Series(3.0, index=runtime_df.index))
    runtime_df["avg_rating"] = pd.to_numeric(g_rating_series, errors="coerce").fillna(3.0)
    
    g_reviews_series = runtime_df.get("g_reviews", pd.Series(0, index=runtime_df.index))
    runtime_df["review_count"] = pd.to_numeric(g_reviews_series, errors="coerce").fillna(0).astype(int)
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
