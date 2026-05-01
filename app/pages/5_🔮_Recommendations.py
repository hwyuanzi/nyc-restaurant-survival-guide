import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.ui_utils import apply_apple_theme
from utils.clustering import (
    apply_mmr,
    build_feature_matrix,
    collect_liked_vectors,
    get_clustered_data,
    prepare_clustering_space,
    recommend_per_liked_knn,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import (
    get_profile,
    get_valid_borough_options,
    get_valid_cuisine_options,
    init_session_state,
    profile_to_user_history,
    upsert_profile,
)

st.set_page_config(page_title="Personalized Recommendations", page_icon="🔮", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

st.title("🔮 Personalized Recommendations")
st.markdown(
    "Recommendations are learned from the restaurants you explicitly like. "
    "The model retrieves neighbors for each liked restaurant, fuses those rankings, "
    "and reranks for diversity so the list reflects your actual saved history."
)


def format_price_tier(value, escape_dollars=False):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or numeric_value <= 0:
        return "—"
    price_text = "$" * int(round(float(numeric_value)))
    return price_text.replace("$", r"\$") if escape_dollars else price_text


def build_restaurant_option(row):
    area = row.get("boro") or row.get("neighborhood") or "NYC"
    return f"{row.get('name', 'Unknown')} · {row.get('cuisine_type', 'Unknown')} · {area}"


def load_liked_entries(profile, df):
    restaurant_lookup = df.drop_duplicates(subset=["restaurant_id"]).set_index("restaurant_id", drop=False)
    entries_by_restaurant = {}

    for like in profile.get("likes", []):
        restaurant_id = str(like.get("restaurant_id", "")).strip()
        if not restaurant_id:
            continue

        row = restaurant_lookup.loc[restaurant_id].to_dict() if restaurant_id in restaurant_lookup.index else {}
        entries_by_restaurant[restaurant_id] = {
            "restaurant_id": restaurant_id,
            "name": row.get("name") or like.get("dba", "Unknown"),
            "cuisine_type": row.get("cuisine_type") or like.get("cuisine", ""),
            "boro": row.get("boro") or like.get("boro", ""),
            "address": row.get("address") or "",
        }

    entries = list(entries_by_restaurant.values())
    entries.sort(key=lambda item: (item["name"].lower(), item["restaurant_id"]))
    return entries


def persist_liked_entries(profile_id, liked_entries, raw_df):
    profile = get_profile(profile_id=profile_id)
    existing_likes = {
        str(item.get("restaurant_id")): item
        for item in profile.get("likes", [])
        if item.get("restaurant_id")
    }
    updated_likes_by_restaurant = {}

    for entry in liked_entries:
        restaurant_id = entry["restaurant_id"]
        existing_like = existing_likes.get(restaurant_id, {})
        updated_likes_by_restaurant[restaurant_id] = {
            **existing_like,
            "restaurant_id": restaurant_id,
            "dba": entry["name"],
            "cuisine": entry["cuisine_type"],
            "boro": entry["boro"],
            "source": existing_like.get("source", "manual_like_manager"),
            "liked_at": existing_like.get("liked_at") or pd.Timestamp.now().isoformat(timespec="seconds"),
        }

    profile["likes"] = list(updated_likes_by_restaurant.values())
    profile = upsert_profile(profile)
    st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
    st.session_state["clustered_df"] = None
    return profile


def liked_centroid_vector(liked_vectors: np.ndarray, fallback_vector: np.ndarray) -> np.ndarray:
    if liked_vectors is None or len(liked_vectors) == 0:
        return fallback_vector
    return np.asarray(liked_vectors, dtype=np.float32).mean(axis=0)

if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading prepared restaurant data..."):
        _, _, runtime_df, _ = load_runtime_assets(DEFAULT_SEARCH_SAMPLE_SIZE)
    if runtime_df.empty:
        st.error("Prepared restaurant data could not be loaded.")
        st.stop()
    st.session_state["raw_df"] = runtime_df

raw_df       = st.session_state["raw_df"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    profile = get_profile(profile_id=st.session_state.get("authenticated_profile_id"))
    st.session_state["active_profile_id"] = profile["id"]
    st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
    st.title(f"👤 Welcome, {profile.get('name', 'Guest')}")
    if st.button("Logout", width="stretch", key="recommendations_logout"):
        st.session_state["authenticated_profile_id"] = None
        st.rerun()
    st.caption("This page ranks restaurants from liked history only. Profile preference sliders are intentionally not used here.")

    profile_id = profile["id"]
    profile_updated_at = profile.get("updated_at", "")
    st.markdown("---")
    st.markdown("### Liked Restaurants")
    st.caption("Add or remove liked restaurants here. No ratings are needed; every liked restaurant is treated as a positive example.")

    session_profile_key = st.session_state.get("recommendations_history_profile_id")
    session_profile_updated_at = st.session_state.get("recommendations_history_profile_updated_at")
    if session_profile_key != profile_id or session_profile_updated_at != profile_updated_at:
        st.session_state["recommendations_history_profile_id"] = profile_id
        st.session_state["recommendations_history_profile_updated_at"] = profile_updated_at
        st.session_state["recommendations_liked_entries"] = load_liked_entries(profile, raw_df)

    liked_entries = st.session_state.get("recommendations_liked_entries", [])

    search_query = st.text_input(
        "Search restaurants",
        placeholder="Type a name, cuisine, borough, or address",
        key="recommendations_liked_restaurant_search",
    ).strip().lower()
    search_filter_col1, search_filter_col2 = st.columns(2)
    with search_filter_col1:
        add_boro_filter = st.selectbox(
            "Add-from borough",
            ["All"] + get_valid_borough_options(raw_df),
            key="recommendations_add_boro_filter",
        )
    with search_filter_col2:
        add_cuisine_filter = st.selectbox(
            "Add-from cuisine",
            ["All"] + get_valid_cuisine_options(raw_df),
            key="recommendations_add_cuisine_filter",
        )

    search_df = raw_df.drop_duplicates(subset=["restaurant_id"]).copy()
    if add_boro_filter != "All":
        search_df = search_df[search_df["boro"] == add_boro_filter]
    if add_cuisine_filter != "All":
        search_df = search_df[search_df["cuisine_type"] == add_cuisine_filter]
    if search_query:
        search_mask = (
            search_df["name"].fillna("").str.lower().str.contains(search_query, regex=False)
            | search_df["cuisine_type"].fillna("").str.lower().str.contains(search_query, regex=False)
            | search_df["boro"].fillna("").str.lower().str.contains(search_query, regex=False)
            | search_df["address"].fillna("").str.lower().str.contains(search_query, regex=False)
        )
        search_df = search_df[search_mask]

    search_df = search_df.sort_values(["name", "restaurant_id"]).head(75)
    restaurant_options = search_df["restaurant_id"].astype(str).tolist()

    selected_restaurant_id = None
    if restaurant_options:
        selected_restaurant_id = st.selectbox(
            "Restaurant",
            options=restaurant_options,
            format_func=lambda rid: build_restaurant_option(
                search_df.loc[search_df["restaurant_id"].astype(str) == str(rid)].iloc[0].to_dict()
            ),
            index=None,
            placeholder="Choose a restaurant",
        )
    else:
        st.caption("No matching restaurants found for the current search.")

    if st.button("Add liked restaurant", width="stretch", disabled=not selected_restaurant_id):
        selected_row = search_df.loc[search_df["restaurant_id"].astype(str) == str(selected_restaurant_id)].iloc[0]
        already_liked = False
        for entry in liked_entries:
            if entry["restaurant_id"] == str(selected_restaurant_id):
                already_liked = True
                break
        if not already_liked:
            liked_entries.append({
                "restaurant_id": str(selected_row["restaurant_id"]),
                "name": selected_row.get("name", "Unknown"),
                "cuisine_type": selected_row.get("cuisine_type", ""),
                "boro": selected_row.get("boro", ""),
                "address": selected_row.get("address", ""),
            })
            liked_entries.sort(key=lambda item: (item["name"].lower(), item["restaurant_id"]))
            st.session_state["recommendations_liked_entries"] = liked_entries
            profile = persist_liked_entries(profile_id, liked_entries, raw_df)
            st.success("Restaurant added to your liked list.")
        else:
            st.info("That restaurant is already in your liked list.")
        st.rerun()

    if liked_entries:
        st.markdown("#### Review And Edit Your Likes")
        liked_entries_df = pd.DataFrame(liked_entries)
        liked_borough_options = ["All"] + get_valid_borough_options(liked_entries_df)
        liked_cuisine_options = ["All"] + get_valid_cuisine_options(liked_entries_df, column="cuisine_type")
        liked_boro_filter = st.selectbox("Liked borough filter", liked_borough_options, key="liked_boro_filter")
        liked_cuisine_filter = st.selectbox("Liked cuisine filter", liked_cuisine_options, key="liked_cuisine_filter")

        filtered_liked_entries = [
            entry for entry in liked_entries
            if (liked_boro_filter == "All" or entry["boro"] == liked_boro_filter)
            and (liked_cuisine_filter == "All" or entry["cuisine_type"] == liked_cuisine_filter)
        ]
        st.caption(f"Showing {len(filtered_liked_entries)} of {len(liked_entries)} liked restaurants.")
        if filtered_liked_entries:
            liked_df = pd.DataFrame(filtered_liked_entries)[["name", "cuisine_type", "boro"]].rename(
                columns={"name": "Restaurant", "cuisine_type": "Cuisine", "boro": "Borough"}
            )
            st.dataframe(liked_df, width="stretch", hide_index=True)
        else:
            st.info("No liked restaurants match the current filters.")

        editable_entries = filtered_liked_entries
        edit_restaurant_id = None
        selected_edit_entry = None
        if editable_entries:
            edit_restaurant_id = st.selectbox(
                "Edit a liked restaurant",
                options=[entry["restaurant_id"] for entry in editable_entries],
                format_func=lambda rid: next(
                    (
                        entry["name"]
                        for entry in editable_entries
                        if entry["restaurant_id"] == rid
                    ),
                    rid,
                ),
                index=None,
                placeholder="Select a liked restaurant",
            )
            selected_edit_entry = next(
                (entry for entry in editable_entries if entry["restaurant_id"] == edit_restaurant_id),
                None,
            )

        if st.button("Remove like", width="stretch", disabled=not edit_restaurant_id):
            liked_entries = [
                entry for entry in liked_entries if entry["restaurant_id"] != edit_restaurant_id
            ]
            st.session_state["recommendations_liked_entries"] = liked_entries
            profile = persist_liked_entries(profile_id, liked_entries, raw_df)
            st.success("Liked restaurant removed.")
            st.rerun()
    else:
        st.caption("You have not liked any restaurants yet. Add a few above to personalize recommendations.")

user_history = st.session_state["user_history"]

# ── K-NN Recommendation Engine ────────────────────────────────────────────────
with st.spinner("Updating recommendation feature space..."):
    X_restaurants, _, df_feat = build_feature_matrix(raw_df)
    _, _, scaler = prepare_clustering_space(X_restaurants, fit=True)

st.markdown("---")
st.subheader("Your Liked-History Signal")

liked_vectors, liked_metadata = collect_liked_vectors(profile, X_restaurants, df_feat)

# Show the liked-history signal used by the recommender
n_likes = len(profile.get("likes", []))
liked_cuisine_count = (
    pd.DataFrame(profile.get("likes", [])).get("cuisine", pd.Series(dtype=str)).dropna().nunique()
    if n_likes else 0
)
liked_borough_count = (
    pd.DataFrame(profile.get("likes", [])).get("boro", pd.Series(dtype=str)).dropna().nunique()
    if n_likes else 0
)

p1, p2, p3 = st.columns(3)
p1.metric("Liked Restaurants", n_likes)
p2.metric("Liked Cuisines", liked_cuisine_count)
p3.metric("Liked Boroughs", liked_borough_count)

if n_likes == 0:
    st.info("Like a few restaurants in the sidebar or from Search/Home to generate personalized recommendations.")

# Run K-NN with the selected method
liked_ids = {str(v) for v in user_history.get("visited_ids", [])}

st.markdown("---")
st.subheader("Recommendation Settings")
st.caption(
    "Per-liked KNN retrieves neighbors for each saved like, Reciprocal Rank Fusion "
    "combines those lists, and MMR reranks the candidates for diversity."
)
mmr_lambda = st.slider(
    "MMR balance (λ)",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help=(
        "λ = 1.0 → pure relevance to liked restaurants.  "
        "λ = 0.0 → stronger diversity among the final picks.  "
        "0.7 is a standard default."
    ),
)

neutral_profile_vector = X_restaurants.mean(axis=0).astype(np.float32)
like_profile_vector = liked_centroid_vector(liked_vectors, neutral_profile_vector)

if n_likes > 0:
    candidates = recommend_per_liked_knn(
        liked_vectors=liked_vectors,
        profile_vector=like_profile_vector,
        restaurant_matrix=X_restaurants,
        restaurant_df=df_feat,
        visited_ids=liked_ids,
        liked_metadata=liked_metadata,
        k_per_liked=30,
        k_final=50,
        scaler=scaler,
        profile=None,
    )
    if len(candidates) > 0:
        id_to_position = {
            str(rid): pos
            for pos, rid in enumerate(df_feat["restaurant_id"].astype(str).values)
        }
        cand_matrix = X_restaurants[
            [id_to_position[str(rid)] for rid in candidates["restaurant_id"].astype(str)]
        ]
        recs = apply_mmr(
            candidates, cand_matrix,
            user_vector=like_profile_vector, k=15, lambda_=mmr_lambda, scaler=scaler,
        )
    else:
        recs = candidates
else:
    recs = pd.DataFrame()

# ── Recommendation cards ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Recommended Restaurants")
st.caption(
    "Restaurants are ranked from your liked restaurants only: per-liked cosine "
    "neighbors in the 18-dim feature space are fused with RRF, then reranked by MMR. "
    "The *Influenced by* column shows which saved like contributed each pick's best rank."
)

if recs.empty:
    st.info("No recommendations yet. Add at least one liked restaurant to train the recommender on your taste history.")
else:
    unique_cuisines = recs["cuisine_type"].fillna("Other").nunique()
    unique_boroughs = recs["boro"].fillna("NYC").nunique()
    d1, d2, d3 = st.columns(3)
    d1.metric("Cuisine diversity", f"{unique_cuisines} unique")
    d2.metric("Borough coverage", f"{unique_boroughs} / 5")
    d3.metric("Method", "Per-liked + MMR")

    display_cols = ["name", "cuisine_type", "avg_rating", "price_tier", "boro",
                    "knn_similarity", "primary_influencer"]
    display_df = recs[[c for c in display_cols if c in recs.columns]].copy()
    display_df["avg_rating"] = display_df["avg_rating"].apply(lambda x: f"{x:.2f} ★")
    display_df["price_tier"] = display_df["price_tier"].apply(format_price_tier)
    if "knn_similarity" in display_df.columns:
        display_df["knn_similarity"] = display_df["knn_similarity"].apply(lambda x: f"{x:.3f}")
    rename_map = {
        "name": "Restaurant",
        "cuisine_type": "Cuisine",
        "avg_rating": "Rating",
        "price_tier": "Price",
        "boro": "Borough",
        "knn_similarity": "Similarity",
        "primary_influencer": "Influenced by",
    }
    display_df = display_df.rename(columns=rename_map)
    st.dataframe(display_df, width="stretch", hide_index=True)

    # Expandable detail cards for top 5
    st.markdown("---")
    st.subheader("Top 5 Picks — Why These?")
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        sim_score = row.get("knn_similarity", 0)
        influencer = row.get("primary_influencer", "liked history")
        with st.expander(f"#{i+1} — {row.get('name', 'Unknown')} (Similarity: {sim_score:.3f})"):
            st.markdown(
                f"<span style='background:#E8F0FE; color:#1D4ED8; "
                f"padding:2px 8px; border-radius:999px; font-size:12px;'>"
                f"Because you liked: {influencer}</span>",
                unsafe_allow_html=True,
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cuisine", row.get("cuisine_type", "—"))
            col2.metric("Rating", f"{row.get('avg_rating', 0):.2f} ★")
            col3.metric("Price", format_price_tier(row.get("price_tier"), escape_dollars=True))
            col4.metric("Borough", row.get("boro", "—"))
            st.caption(f"📍 {row.get('address', row.get('boro', ''))}")
            
            # Explain WHY this restaurant was recommended
            reasons = []
            influencer = row.get("primary_influencer", "")
            if influencer and influencer != "liked history":
                reasons.append(f"nearest-neighbor match to {influencer}")
            if row.get("avg_rating", 0) >= 4.5:
                reasons.append("strong public rating")
            if reasons:
                st.markdown(f"**Why recommended:** {'; '.join(reasons)}")
            else:
                st.markdown("**Why recommended:** Similar feature profile to restaurants you liked.")

            if row.get("g_maps_url"):
                st.markdown(f"[📍 Open in Google Maps]({row['g_maps_url']})")

# ── Cluster visualization kept as interpretive context ───────────────────────
st.markdown("---")
st.subheader("Restaurant Space Cluster View")
st.caption(
    "Each dot is a restaurant colored by cluster. The diamond marks the average "
    "feature position of your liked restaurants, and dotted lines point to the "
    "top recommendations. This view is explanatory; recommendations are still "
    "ranked from your liked restaurants."
)

with st.spinner("Preparing cluster visualization..."):
    cluster_df, _, cluster_scaler, cluster_pca = get_clustered_data(
        raw_df,
        user_history,
        k=st.session_state.get("optimal_k", 10),
        force=(st.session_state.get("clustered_df") is None),
        algorithm=st.session_state.get("active_cluster_algorithm", "kmeans"),
    )

if cluster_pca is not None and hasattr(cluster_pca, "axis_labels_"):
    from utils.clustering import _scaled_space

    X_scaled, _, _ = prepare_clustering_space(
        X_restaurants, scaler=cluster_scaler, fit=False
    )
    X_pca_all = cluster_pca.transform(X_scaled)

    cluster_hex = [
        "#6c8fff", "#ff9f43", "#6dda7f", "#ff6b8a",
        "#b983ff", "#ffd32a", "#48dbfb", "#ff6348",
        "#34d399", "#fb923c", "#a78bfa", "#22d3ee",
        "#f87171", "#4ade80", "#facc15", "#818cf8",
    ]

    fig = go.Figure()
    for cid in sorted(cluster_df["cluster_id"].unique()):
        mask = cluster_df["cluster_id"] == cid
        label = cluster_df[mask]["cluster_label"].iloc[0]
        color = cluster_hex[int(cid) % len(cluster_hex)]
        indices = np.where(mask.values)[0]
        fig.add_trace(go.Scatter3d(
            x=X_pca_all[indices, 0],
            y=X_pca_all[indices, 1],
            z=X_pca_all[indices, 2],
            mode="markers",
            name=label,
            marker=dict(size=3, color=color, opacity=0.45),
            text=cluster_df[mask]["name"].values,
            hovertemplate="%{text}<extra>" + label + "</extra>",
        ))

    if n_likes > 0:
        liked_scaled = _scaled_space(like_profile_vector, cluster_scaler)
        liked_pca = cluster_pca.transform(liked_scaled)[0]
        fig.add_trace(go.Scatter3d(
            x=[liked_pca[0]],
            y=[liked_pca[1]],
            z=[liked_pca[2]],
            mode="markers+text",
            name="Liked-history center",
            marker=dict(
                size=12,
                color="#FFD700",
                symbol="diamond",
                line=dict(width=2, color="#000000"),
            ),
            text=["Your likes"],
            textposition="top center",
        ))

        for _, row in recs.head(5).iterrows():
            rid = str(row.get("restaurant_id", ""))
            idx_match = df_feat.index[df_feat["restaurant_id"].astype(str) == rid]
            if len(idx_match) == 0:
                continue
            idx = df_feat.index.get_loc(idx_match[0])
            fig.add_trace(go.Scatter3d(
                x=[liked_pca[0], X_pca_all[idx, 0]],
                y=[liked_pca[1], X_pca_all[idx, 1]],
                z=[liked_pca[2], X_pca_all[idx, 2]],
                mode="lines",
                line=dict(color="#FFD700", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    axis_labels = cluster_pca.axis_labels_
    var_explained = cluster_pca.explained_variance_ratio_[:3]
    fig.update_layout(
        height=600,
        scene=dict(
            xaxis_title=f"{axis_labels[0]} ({var_explained[0]:.0%} var.)",
            yaxis_title=f"{axis_labels[1]} ({var_explained[1]:.0%} var.)",
            zaxis_title=f"{axis_labels[2]} ({var_explained[2]:.0%} var.)",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(fig, width="stretch")
