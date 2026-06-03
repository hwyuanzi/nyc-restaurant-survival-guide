"""
place_photos.py — Restaurant photo loading via Places API (New).

Kept separate from ``google_places.py`` so Streamlit pages can import photo
helpers without triggering circular imports through the search / enrichment stack.
"""

from __future__ import annotations

from functools import lru_cache

import requests

PLACES_NEW_BASE = "https://places.googleapis.com/v1"


def fetch_place_photo_bytes(place_id: str, api_key: str, max_px: int = 400) -> bytes | None:
    """Download one restaurant photo as raw bytes (Places API New)."""
    if not place_id or not api_key:
        return None

    try:
        headers = {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "photos"}
        details = requests.get(
            f"{PLACES_NEW_BASE}/places/{place_id}",
            headers=headers,
            timeout=12,
        )
        details.raise_for_status()
        photos = details.json().get("photos", [])
        if not photos:
            return None

        photo_name = photos[0].get("name")
        if not photo_name:
            return None

        media = requests.get(
            f"{PLACES_NEW_BASE}/{photo_name}/media",
            headers={"X-Goog-Api-Key": api_key},
            params={"maxHeightPx": max_px, "skipHttpRedirect": "true"},
            timeout=15,
        )
        media.raise_for_status()
        photo_uri = media.json().get("photoUri")
        if not photo_uri:
            return None

        image = requests.get(photo_uri, timeout=15)
        image.raise_for_status()
        return image.content or None
    except Exception:
        return None


@lru_cache(maxsize=256)
def get_restaurant_photo_bytes(place_id: str, api_key: str) -> bytes | None:
    """Cached wrapper for UI cards (keyed by place_id + api_key)."""
    return fetch_place_photo_bytes(place_id, api_key)
