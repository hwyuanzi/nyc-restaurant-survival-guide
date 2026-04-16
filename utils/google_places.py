import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from utils.search import neighborhood_from_zipcode

PLACES_SEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_PHOTO = "https://maps.googleapis.com/maps/api/place/photo"
DEFAULT_GOOGLE_API_KEY = "AIzaSyBM_Td0_NgsHAmjOldP7AVH5pySmZH--I8"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ENRICHED_CACHE_DIR = DATA_DIR / "cache"


def get_google_api_key():
    try:
        secret_value = st.secrets.get("GOOGLE_API_KEY")
        if secret_value:
            return secret_value
    except Exception:
        pass
    return DEFAULT_GOOGLE_API_KEY


def _ensure_cache_dir():
    ENRICHED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(sample_size):
    return ENRICHED_CACHE_DIR / f"enriched_restaurants_{sample_size}.pkl"


def build_photo_url(photo_ref, api_key, max_width=400):
    return f"{PLACES_PHOTO}?maxwidth={max_width}&photo_reference={photo_ref}&key={api_key}"


def fetch_google_place(name, address, api_key):
    if not api_key:
        return None

    query = f"{name} restaurant {address}"
    try:
        search_response = requests.get(
            PLACES_SEARCH,
            params={
                "query": query,
                "key": api_key,
                "type": "restaurant",
                "region": "us",
            },
            timeout=10,
        )
        search_response.raise_for_status()
        results = search_response.json().get("results", [])
        if not results:
            return None

        place_id = results[0].get("place_id")
        if not place_id:
            return None

        details_response = requests.get(
            PLACES_DETAILS,
            params={
                "place_id": place_id,
                "key": api_key,
                "fields": "name,rating,user_ratings_total,price_level,editorial_summary,photos,url",
            },
            timeout=10,
        )
        details_response.raise_for_status()
        details = details_response.json().get("result", {})

        rating = details.get("rating")
        photos = details.get("photos", [])
        photo_ref = photos[0].get("photo_reference") if photos else None
        if not photo_ref or not rating:
            return None

        return {
            "g_rating": rating,
            "g_reviews": details.get("user_ratings_total"),
            "g_price": details.get("price_level"),
            "g_summary": details.get("editorial_summary", {}).get("overview", ""),
            "g_photo_ref": photo_ref,
            "g_maps_url": details.get("url", ""),
            "g_place_id": place_id,
        }
    except Exception:
        return None


def enrich_with_google(nyc_df, sample_size, api_key):
    if nyc_df.empty or not api_key:
        return pd.DataFrame()

    prioritized = nyc_df.copy()
    prioritized["zipcode_norm"] = prioritized.get("zipcode", pd.Series([""] * len(prioritized), index=prioritized.index)).fillna("").astype(str).str[:5]
    prioritized["neighborhood"] = prioritized["zipcode_norm"].apply(neighborhood_from_zipcode)
    prioritized["_quality_rank"] = pd.to_numeric(prioritized.get("score", 999), errors="coerce").fillna(999)
    prioritized["_group_neighborhood"] = prioritized["neighborhood"].where(prioritized["neighborhood"].astype(str).str.len() > 0, prioritized.get("boro", ""))
    prioritized = prioritized.sort_values(["_quality_rank", "dba", "camis"], ascending=[True, True, True]).reset_index(drop=True)

    coverage_frames = []
    coverage_frames.append(prioritized.drop_duplicates(subset=["_group_neighborhood", "cuisine"], keep="first"))
    coverage_frames.append(prioritized.drop_duplicates(subset=["_group_neighborhood"], keep="first"))
    coverage_frames.append(prioritized.drop_duplicates(subset=["cuisine"], keep="first"))
    coverage_pool = pd.concat(coverage_frames, ignore_index=True).drop_duplicates(subset=["camis"], keep="first")

    remaining = prioritized[~prioritized["camis"].isin(coverage_pool["camis"])]
    sample = pd.concat([coverage_pool, remaining], ignore_index=True).head(min(sample_size, len(prioritized))).copy()
    sample = sample.drop(columns=["zipcode_norm", "neighborhood", "_quality_rank", "_group_neighborhood"], errors="ignore")
    enriched_rows = []
    progress = st.progress(0, text="Fetching Google Places data...")
    total = len(sample)

    for i, (_, row) in enumerate(sample.iterrows()):
        google_data = fetch_google_place(row["dba"], row["address"], api_key)
        if google_data:
            merged = row.to_dict()
            merged.update(google_data)
            enriched_rows.append(merged)

        time.sleep(0.12)

        if i % 10 == 0 or i == total - 1:
            pct = min(int(((i + 1) / total) * 100), 99)
            progress.progress(
                pct,
                text=f"Google Places... {i + 1}/{total} checked · {len(enriched_rows)} enriched",
            )

    progress.progress(100, text=f"Done - {len(enriched_rows)} restaurants indexed")
    time.sleep(0.4)
    progress.empty()

    if not enriched_rows:
        return pd.DataFrame()
    return pd.DataFrame(enriched_rows).reset_index(drop=True)


def get_enriched_restaurants(nyc_df, sample_size, api_key, force_refresh=False):
    _ensure_cache_dir()
    cache_file = _cache_path(sample_size)

    if cache_file.exists() and not force_refresh:
        try:
            cached_df = pd.read_pickle(cache_file)
            if not cached_df.empty:
                return cached_df
        except Exception:
            pass

    enriched_df = enrich_with_google(nyc_df, sample_size, api_key)
    if not enriched_df.empty:
        enriched_df.to_pickle(cache_file)
    return enriched_df
