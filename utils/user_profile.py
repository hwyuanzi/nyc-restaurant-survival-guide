import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
USER_PROFILES_PATH = DATA_DIR / "user_profiles.local.json"
DEFAULT_PROFILE_ID = "guest"

CUISINE_OPTIONS = [
    "American", "Bakery", "Cafe", "Caribbean", "Chinese", "French", "Greek",
    "Indian", "Italian", "Japanese", "Korean", "Mediterranean", "Mexican",
    "Middle Eastern", "Pizza", "Seafood", "Spanish", "Thai", "Vegan",
    "Vegetarian", "Vietnamese",
]
BOROUGH_OPTIONS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
VIBE_OPTIONS = ["Casual", "Cozy", "Date Night", "Family Friendly", "Late Night", "Trendy", "Upscale"]
BUDGET_OPTIONS = ["$", "$$", "$$$", "$$$$"]
GRADE_ORDER = {"A": 3, "B": 2, "C": 1, "N/A": 0}
SPICY_CUISINES = {"Indian", "Korean", "Mexican", "Sichuan", "Thai", "Caribbean"}
ADVENTUROUS_CUISINES = {"Ethiopian", "Indian", "Japanese", "Korean", "Mediterranean", "Spanish", "Thai", "Vietnamese"}


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _format_budget_slider_value(value):
    return r"\$" * int(value)


def format_budget_display(budget_value: str) -> str:
    labels = {
        "$": "Budget ($)",
        "$$": "Mid-Range ($$)",
        "$$$": "Premium ($$$)",
        "$$$$": "Luxury ($$$$)",
    }
    return labels.get(budget_value, budget_value or "Not set")


def get_valid_borough_options(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "boro" not in df.columns:
        return BOROUGH_OPTIONS.copy()
    present = set(
        df["boro"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )
    return [borough for borough in BOROUGH_OPTIONS if borough in present]


def get_valid_cuisine_options(df: pd.DataFrame, column: str = "cuisine_type") -> list[str]:
    if df is None or df.empty or column not in df.columns:
        return []
    invalid_values = {"", "0", "Unknown", "Nan", "None"}
    values = sorted(
        {
            str(value).strip()
            for value in df[column].dropna().tolist()
            if str(value).strip() and str(value).strip().title() not in invalid_values
            and str(value).strip() not in invalid_values
        }
    )
    return values


def _slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or DEFAULT_PROFILE_ID


def _default_profile(name="Guest", profile_id=None):
    profile_id = profile_id or _slugify(name)
    timestamp = _now_iso()
    return {
        "id": profile_id,
        "name": name,
        "likes": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def load_profiles():
    if not USER_PROFILES_PATH.exists():
        return {}
    try:
        with USER_PROFILES_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def find_profile_by_name(name):
    if not name:
        return None
    profiles = load_profiles()
    for existing in profiles.values():
        if existing.get("name", "").strip().lower() == name.strip().lower():
            return existing
    return None


def save_profiles(profiles):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with USER_PROFILES_PATH.open("w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=2)


def get_profile(profile_id=None, name=None):
    profiles = load_profiles()
    if profile_id and profile_id in profiles:
        return profiles[profile_id]

    if name:
        existing = find_profile_by_name(name)
        if existing:
            return existing

    if profiles:
        first_key = next(iter(profiles))
        return profiles[first_key]

    guest_profile = _default_profile()
    profiles[guest_profile["id"]] = guest_profile
    save_profiles(profiles)
    return guest_profile


def upsert_profile(profile):
    profiles = load_profiles()
    profile = profile.copy()
    profile.setdefault("created_at", _now_iso())
    profile["updated_at"] = _now_iso()
    profiles[profile["id"]] = profile
    save_profiles(profiles)
    return profile


def _series_from_candidates(df, candidates, default=""):
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _numeric_series(df, candidates, default):
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _restaurant_id_series(df):
    if "restaurant_id" in df.columns:
        return df["restaurant_id"].astype(str)
    if "g_place_id" in df.columns:
        return df["g_place_id"].astype(str)
    if "camis" in df.columns:
        return df["camis"].astype(str)
    return pd.Series(df.index.astype(str), index=df.index)


def get_default_user_history():
    return {
        "visited_ids": [],
        "rated": {},
        "cuisine_preferences": [],
        "price_preference": 2,
        "neighborhood_preference": [],
    }


def _normalize_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _resolve_like_restaurant_id(like_item, restaurant_df=None):
    candidate_ids = [
        like_item.get("restaurant_id"),
        like_item.get("camis"),
    ]
    for candidate in candidate_ids:
        if candidate is not None and str(candidate).strip():
            candidate_text = str(candidate).strip()
            if restaurant_df is None:
                return candidate_text
            if candidate_text in set(restaurant_df["restaurant_id"].astype(str)):
                return candidate_text

    if restaurant_df is None or restaurant_df.empty:
        return None

    normalized_name = _normalize_name(like_item.get("dba") or like_item.get("name"))
    normalized_boro = str(like_item.get("boro", "")).strip().lower()
    normalized_cuisine = str(like_item.get("cuisine", "")).strip().lower()

    candidates = restaurant_df.copy()
    candidates["_norm_name"] = candidates["name"].fillna("").map(_normalize_name)
    candidates["_norm_boro"] = candidates["boro"].fillna("").astype(str).str.lower()
    candidates["_norm_cuisine"] = candidates["cuisine_type"].fillna("").astype(str).str.lower()

    if normalized_name:
        candidates = candidates[candidates["_norm_name"] == normalized_name]
    if normalized_boro and not candidates.empty:
        boro_match = candidates[candidates["_norm_boro"] == normalized_boro]
        if not boro_match.empty:
            candidates = boro_match
    if normalized_cuisine and not candidates.empty:
        cuisine_match = candidates[candidates["_norm_cuisine"] == normalized_cuisine]
        if not cuisine_match.empty:
            candidates = cuisine_match

    if candidates.empty:
        return None
    return str(candidates.iloc[0]["restaurant_id"])


def profile_to_user_history(profile, restaurant_df=None):
    likes = profile.get("likes", [])
    visited_ids = []
    rated = {}
    for item in likes:
        restaurant_id = _resolve_like_restaurant_id(item, restaurant_df=restaurant_df)
        if not restaurant_id:
            continue
        if restaurant_id not in visited_ids:
            visited_ids.append(restaurant_id)
        rated[restaurant_id] = 1.0
    return {
        "visited_ids": visited_ids,
        "rated": rated,
        "cuisine_preferences": profile.get("favorite_cuisines", []),
        "price_preference": BUDGET_OPTIONS.index(profile.get("budget", "$$")) + 1 if profile.get("budget", "$$") in BUDGET_OPTIONS else 2,
        "neighborhood_preference": profile.get("preferred_boroughs", []),
    }


def build_profile_prompt(profile):
    profile = profile or {}
    parts = []
    if profile.get("favorite_cuisines"):
        parts.append("Favorite cuisines: " + ", ".join(profile["favorite_cuisines"]))
    if profile.get("preferred_boroughs"):
        parts.append("Preferred boroughs: " + ", ".join(profile["preferred_boroughs"]))
    if profile.get("budget"):
        parts.append(f"Budget: {profile['budget']}")
    if profile.get("favorite_vibes"):
        parts.append("Preferred vibes: " + ", ".join(profile["favorite_vibes"]))
    if profile.get("spice_tolerance"):
        parts.append(f"Spice tolerance: {profile['spice_tolerance']}/5")
    if profile.get("adventurousness"):
        parts.append(f"Adventurousness: {profile['adventurousness']}/5")
    if profile.get("min_grade"):
        parts.append(f"Minimum preferred health grade: {profile['min_grade']}")
    return ". ".join(parts)


def score_restaurants_for_user(df, profile):
    profile = profile or {}
    scored = df.copy()
    if scored.empty:
        scored["preference_score"] = pd.Series(dtype=float)
        return scored

    cuisine = _series_from_candidates(scored, ["cuisine", "cuisine_type"]).fillna("").astype(str)
    borough = _series_from_candidates(scored, ["boro", "neighborhood"]).fillna("").astype(str)
    grade = _series_from_candidates(scored, ["grade"], default="N/A").fillna("N/A").astype(str)
    description = _series_from_candidates(scored, ["description"], default="").fillna("").astype(str)
    price = _numeric_series(scored, ["g_price", "price_tier"], default=2).clip(1, 4)
    rating = _numeric_series(scored, ["g_rating", "avg_rating"], default=3.5).clip(0, 5)
    restaurant_ids = _restaurant_id_series(scored)

    favorite_cuisines = {value.lower() for value in profile.get("favorite_cuisines", [])}
    preferred_boroughs = {value.lower() for value in profile.get("preferred_boroughs", [])}
    favorite_vibes = {value.lower() for value in profile.get("favorite_vibes", [])}
    liked_items = profile.get("likes", [])
    liked_ids = {str(item.get("restaurant_id")) for item in liked_items if item.get("restaurant_id")}
    liked_cuisines = {str(item.get("cuisine", "")).lower() for item in liked_items if item.get("cuisine")}
    liked_boroughs = {str(item.get("boro", "")).lower() for item in liked_items if item.get("boro")}
    min_grade = profile.get("min_grade")
    budget = profile.get("budget")
    spice_tolerance = float(profile.get("spice_tolerance", 3))
    adventurousness = float(profile.get("adventurousness", 3))

    preference = pd.Series(np.full(len(scored), 3.5, dtype=float), index=scored.index)
    preference += 1.1 * (rating / 5)

    if favorite_cuisines:
        cuisine_match = cuisine.str.lower().isin(favorite_cuisines)
        preference += cuisine_match.astype(float) * 2.2

    if preferred_boroughs:
        borough_match = borough.str.lower().isin(preferred_boroughs)
        preference += borough_match.astype(float) * 1.3

    if budget in BUDGET_OPTIONS:
        budget_value = BUDGET_OPTIONS.index(budget) + 1
        preference += (1 - (price - budget_value).abs() / 3).clip(0, 1) * 1.2

    if min_grade in GRADE_ORDER:
        grade_floor = GRADE_ORDER[min_grade]
        grade_values = grade.map(lambda value: GRADE_ORDER.get(value, 0))
        preference += (grade_values >= grade_floor).astype(float) * 0.9
        preference -= (grade_values < grade_floor).astype(float) * 0.6

    if liked_cuisines:
        preference += cuisine.str.lower().isin(liked_cuisines).astype(float) * 1.0
    if liked_boroughs:
        preference += borough.str.lower().isin(liked_boroughs).astype(float) * 0.5
    if liked_ids:
        preference += restaurant_ids.isin(liked_ids).astype(float) * 3.0

    spicy_bias = cuisine.isin(SPICY_CUISINES).astype(float)
    adventurous_bias = cuisine.isin(ADVENTUROUS_CUISINES).astype(float)
    preference += spicy_bias * max(spice_tolerance - 3, 0) * 0.25
    preference += adventurous_bias * max(adventurousness - 3, 0) * 0.25

    if favorite_vibes:
        lowered_description = description.str.lower()
        vibe_matches = np.zeros(len(scored), dtype=float)
        for vibe in favorite_vibes:
            vibe_matches += lowered_description.str.contains(vibe, regex=False).astype(float).to_numpy()
        preference += np.clip(vibe_matches, 0, 1) * 0.8

    scored["preference_score"] = np.clip(np.round(preference, 2), 1, 10)
    return scored


def predict_user_cluster(user_history, df_clustered, kmeans, scaler):
    if df_clustered.empty or "restaurant_id" not in df_clustered.columns:
        return -1

    visited = df_clustered[df_clustered["restaurant_id"].isin(user_history.get("visited_ids", []))]
    if visited.empty:
        return -1

    cluster_votes = (
        visited.groupby("cluster_id")
        .size()
        .sort_values(ascending=False)
    )
    if not cluster_votes.empty and len(cluster_votes) == 1:
        return int(cluster_votes.index[0])
    if not cluster_votes.empty and cluster_votes.iloc[0] >= cluster_votes.iloc[-1] * 1.25:
        return int(cluster_votes.index[0])

    try:
        from utils.clustering import build_feature_matrix, prepare_clustering_space

        X, _, clustered = build_feature_matrix(df_clustered)
        visited_mask = clustered["restaurant_id"].isin(user_history["visited_ids"])
        visited_vecs = X[visited_mask.values]
        user_vec = visited_vecs.mean(axis=0).reshape(1, -1)
        _, user_vec_cluster, _ = prepare_clustering_space(user_vec, scaler=scaler, fit=False)
        return int(kmeans.predict(user_vec_cluster)[0])
    except Exception:
        return -1


def init_session_state():
    defaults = {
        "clustered_df": None,
        "kmeans_model": None,
        "scaler": None,
        "pca_model": None,
        "predicted_cluster": -1,
        "selected_cluster_label": "All Clusters",
        "active_profile_id": None,
        "user_history": get_default_user_history(),
        "optimal_k": 9,
        "raw_df": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _persist_profile_updates(profile_id, **updates):
    profile = get_profile(profile_id=profile_id)
    profile.update(updates)
    if updates:
        profile["survey_completed"] = True
    return upsert_profile(profile)


def render_profile_sidebar():
    init_session_state()
    profiles = load_profiles()
    if not profiles:
        guest_profile = _default_profile()
        profiles[guest_profile["id"]] = guest_profile
        save_profiles(profiles)

    profile_ids = list(profiles.keys())
    if st.session_state.get("authenticated_profile_id") not in profiles:
        # Fallback if somehow they are authenticated but profile is missing
        st.session_state["authenticated_profile_id"] = profile_ids[0]

    st.session_state["active_profile_id"] = st.session_state["authenticated_profile_id"]
    active_name = profiles[st.session_state["active_profile_id"]]["name"]

    st.title(f"👤 Welcome, {active_name}")
    
    if st.button("Logout", width="stretch"):
        st.session_state["authenticated_profile_id"] = None
        st.rerun()

    profile = get_profile(profile_id=st.session_state["active_profile_id"])

    favorite_cuisines = st.multiselect(
        "Favorite cuisines",
        options=CUISINE_OPTIONS,
        default=profile.get("favorite_cuisines", []),
    )
    preferred_boroughs = st.multiselect(
        "Preferred boroughs",
        options=BOROUGH_OPTIONS,
        default=profile.get("preferred_boroughs", []),
    )
    budget_value = BUDGET_OPTIONS.index(profile.get("budget", "$$")) + 1 if profile.get("budget", "$$") in BUDGET_OPTIONS else 2
    budget_value = st.select_slider(
        "Budget",
        options=range(1, len(BUDGET_OPTIONS) + 1),
        value=budget_value,
        format_func=_format_budget_slider_value,
    )
    budget = BUDGET_OPTIONS[budget_value - 1]
    min_grade = st.selectbox("Lowest acceptable health grade", ["A", "B", "C"], index=["A", "B", "C"].index(profile.get("min_grade", "B")))
    spice_tolerance = st.slider("Spice tolerance", 1, 5, int(profile.get("spice_tolerance", 3)))
    adventurousness = st.slider("Adventurousness", 1, 5, int(profile.get("adventurousness", 3)))
    favorite_vibes = st.multiselect(
        "Favorite vibes",
        options=VIBE_OPTIONS,
        default=profile.get("favorite_vibes", []),
    )

    if st.button("Save preferences", width="stretch"):
        profile = _persist_profile_updates(
            profile["id"],
            favorite_cuisines=favorite_cuisines,
            preferred_boroughs=preferred_boroughs,
            budget=budget,
            min_grade=min_grade,
            spice_tolerance=spice_tolerance,
            adventurousness=adventurousness,
            favorite_vibes=favorite_vibes,
        )
        st.success("Preferences saved.")

    profile = get_profile(profile_id=profile["id"])
    st.caption(f"Saved likes: {len(profile.get('likes', []))}")

    with st.expander("Account Management"):
        st.caption("Update your password or permanently delete this profile.")
        current_password = st.text_input("Current password", type="password", key="account_current_password")
        new_password = st.text_input("New password", type="password", key="account_new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="account_confirm_password")

        if st.button("Update password", width="stretch"):
            if new_password != confirm_password:
                st.error("New password and confirmation do not match.")
            else:
                from utils.auth import change_password

                success, message = change_password(profile["id"], current_password, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

        st.markdown("---")
        st.caption("Deleting your profile removes your saved likes and preferences.")
        delete_confirm_name = st.text_input(
            f"Type `{profile['name']}` to confirm deletion",
            key="account_delete_name",
        ).strip()
        delete_password = st.text_input("Password for deletion", type="password", key="account_delete_password")
        delete_disabled = delete_confirm_name != profile["name"]
        if st.button("Delete my profile", width="stretch", disabled=delete_disabled):
            from utils.auth import delete_user_account

            success, message = delete_user_account(profile["id"], delete_password)
            if success:
                st.session_state["authenticated_profile_id"] = None
                st.session_state["active_profile_id"] = None
                st.session_state["clustered_df"] = None
                st.session_state["user_history"] = get_default_user_history()
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.session_state["user_history"] = profile_to_user_history(profile)
    return profile


def get_active_profile():
    """Return the authenticated profile without rendering sidebar controls."""
    init_session_state()
    profiles = load_profiles()
    if not profiles:
        guest_profile = _default_profile()
        profiles[guest_profile["id"]] = guest_profile
        save_profiles(profiles)

    profile_ids = list(profiles.keys())
    authenticated_id = st.session_state.get("authenticated_profile_id")
    if authenticated_id not in profiles:
        authenticated_id = profile_ids[0]

    st.session_state["active_profile_id"] = authenticated_id
    profile = get_profile(profile_id=authenticated_id)
    st.session_state["user_history"] = profile_to_user_history(profile)
    return profile


def add_liked_restaurant(profile_name, restaurant_row, source="app"):
    profiles = load_profiles()
    profile = get_profile(name=profile_name)
    profile_id = profile["id"]
    likes = profile.get("likes", [])

    camis = restaurant_row.get("camis") or restaurant_row.get("restaurant_id")
    restaurant_id = str(
        restaurant_row.get("restaurant_id")
        or restaurant_row.get("camis")
        or restaurant_row.get("g_place_id")
        or restaurant_row.get("dba")
    )
    camis_text = str(camis).strip() if camis is not None and str(camis).strip() else ""

    if any(
        str(item.get("restaurant_id")) == restaurant_id
        or (camis_text and str(item.get("camis", "")).strip() == camis_text)
        for item in likes
    ):
        return False

    like_record = {
        "restaurant_id": restaurant_id,
        "camis": camis_text,
        "g_place_id": str(restaurant_row.get("g_place_id", "")).strip(),
        "dba": restaurant_row.get("dba") or restaurant_row.get("name", "Unknown"),
        "cuisine": restaurant_row.get("cuisine") or restaurant_row.get("cuisine_type", ""),
        "boro": restaurant_row.get("boro") or restaurant_row.get("neighborhood", ""),
        "grade": restaurant_row.get("grade", "N/A"),
        "score": int(pd.to_numeric(restaurant_row.get("score", 0), errors="coerce") or 0),
        "source": source,
        "liked_at": _now_iso(),
    }
    likes.append(like_record)
    profile["likes"] = likes
    profile["survey_completed"] = True
    profile["updated_at"] = _now_iso()
    profiles[profile_id] = profile
    save_profiles(profiles)

    st.session_state["user_history"] = profile_to_user_history(profile)
    return True


def is_restaurant_liked(profile_name, restaurant_row):
    profile = get_profile(name=profile_name)
    likes = profile.get("likes", [])
    camis = restaurant_row.get("camis") or restaurant_row.get("restaurant_id")
    restaurant_id = str(
        restaurant_row.get("restaurant_id")
        or restaurant_row.get("camis")
        or restaurant_row.get("g_place_id")
        or restaurant_row.get("dba")
    )
    camis_text = str(camis).strip() if camis is not None and str(camis).strip() else ""
    return any(
        str(item.get("restaurant_id")) == restaurant_id
        or (camis_text and str(item.get("camis", "")).strip() == camis_text)
        for item in likes
    )


def remove_liked_restaurant(profile_name, restaurant_row):
    profiles = load_profiles()
    profile = get_profile(name=profile_name)
    profile_id = profile["id"]
    likes = profile.get("likes", [])

    camis = restaurant_row.get("camis") or restaurant_row.get("restaurant_id")
    restaurant_id = str(
        restaurant_row.get("restaurant_id")
        or restaurant_row.get("camis")
        or restaurant_row.get("g_place_id")
        or restaurant_row.get("dba")
    )
    camis_text = str(camis).strip() if camis is not None and str(camis).strip() else ""

    new_likes = [
        item for item in likes
        if not (
            str(item.get("restaurant_id")) == restaurant_id
            or (camis_text and str(item.get("camis", "")).strip() == camis_text)
        )
    ]
    
    if len(new_likes) == len(likes):
        return False
        
    profile["likes"] = new_likes
    profile["updated_at"] = _now_iso()
    profiles[profile_id] = profile
    save_profiles(profiles)
    st.session_state["user_history"] = profile_to_user_history(profile)
    return True
