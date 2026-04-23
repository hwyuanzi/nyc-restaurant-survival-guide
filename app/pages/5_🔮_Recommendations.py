import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

from app.ui_utils import apply_apple_theme
from utils.clustering import get_clustered_data
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import (
    format_budget_display,
    get_profile,
    get_valid_borough_options,
    get_valid_cuisine_options,
    init_session_state,
    predict_user_cluster,
    profile_to_user_history,
    render_profile_sidebar,
    upsert_profile,
)

st.set_page_config(page_title="Personalized Recommendations", page_icon="🔮", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

st.title("🔮 Personalized Recommendations")
st.markdown(
    "Recommendations are driven primarily by your **liked restaurants** using per-liked nearest-neighbor retrieval and diversity-aware reranking. "
    "Clusters are used for explanation and taste context, not as the sole ranking signal."
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
        rating_value = pd.to_numeric(like.get("rating", 5), errors="coerce")
        entries_by_restaurant[restaurant_id] = {
            "restaurant_id": restaurant_id,
            "name": row.get("name") or like.get("dba", "Unknown"),
            "cuisine_type": row.get("cuisine_type") or like.get("cuisine", ""),
            "boro": row.get("boro") or like.get("boro", ""),
            "address": row.get("address") or "",
            "rating": int(np.clip(rating_value if pd.notna(rating_value) else 5, 1, 5)),
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
            "rating": float(entry["rating"]),
            "source": existing_like.get("source", "manual_like_manager"),
            "liked_at": existing_like.get("liked_at") or pd.Timestamp.now().isoformat(timespec="seconds"),
        }

    profile["likes"] = list(updated_likes_by_restaurant.values())
    profile = upsert_profile(profile)
    st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
    st.session_state["clustered_df"] = None
    return profile

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
    profile = render_profile_sidebar()
    st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
    profile_id = profile["id"]
    profile_updated_at = profile.get("updated_at", "")
    st.markdown("---")
    st.markdown("### Liked Restaurants")
    st.caption("Recommendations learn from restaurants you explicitly like. Add, filter, update, or remove them here.")

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

    selected_rating = st.selectbox("Rating", options=[1, 2, 3, 4, 5], index=4)

    if st.button("Add liked restaurant", use_container_width=True, disabled=not selected_restaurant_id):
        selected_row = search_df.loc[search_df["restaurant_id"].astype(str) == str(selected_restaurant_id)].iloc[0]
        updated = False
        for entry in liked_entries:
            if entry["restaurant_id"] == str(selected_restaurant_id):
                entry["rating"] = selected_rating
                updated = True
                break
        if not updated:
            liked_entries.append({
                "restaurant_id": str(selected_row["restaurant_id"]),
                "name": selected_row.get("name", "Unknown"),
                "cuisine_type": selected_row.get("cuisine_type", ""),
                "boro": selected_row.get("boro", ""),
                "address": selected_row.get("address", ""),
                "rating": selected_rating,
            })
            liked_entries.sort(key=lambda item: (item["name"].lower(), item["restaurant_id"]))
        st.session_state["recommendations_liked_entries"] = liked_entries
        profile = persist_liked_entries(profile_id, liked_entries, raw_df)
        if updated:
            st.info("That restaurant was already liked, so its rating was updated instead.")
        else:
            st.success("Restaurant added to your liked list.")
        st.rerun()

    if liked_entries:
        st.markdown("#### Review And Filter Your Likes")
        liked_entries_df = pd.DataFrame(liked_entries)
        liked_borough_options = ["All"] + get_valid_borough_options(liked_entries_df)
        liked_cuisine_options = ["All"] + get_valid_cuisine_options(liked_entries_df, column="cuisine_type")
        liked_boro_filter = st.selectbox("Liked borough filter", liked_borough_options, key="liked_boro_filter")
        liked_cuisine_filter = st.selectbox("Liked cuisine filter", liked_cuisine_options, key="liked_cuisine_filter")
        liked_min_rating = st.slider("Minimum liked rating", 1, 5, 1, key="liked_min_rating")

        filtered_liked_entries = [
            entry for entry in liked_entries
            if (liked_boro_filter == "All" or entry["boro"] == liked_boro_filter)
            and (liked_cuisine_filter == "All" or entry["cuisine_type"] == liked_cuisine_filter)
            and int(entry["rating"]) >= liked_min_rating
        ]
        st.caption(f"Showing {len(filtered_liked_entries)} of {len(liked_entries)} liked restaurants.")
        if filtered_liked_entries:
            liked_df = pd.DataFrame(filtered_liked_entries)[["name", "cuisine_type", "boro", "rating"]].rename(
                columns={"name": "Restaurant", "cuisine_type": "Cuisine", "boro": "Borough", "rating": "Rating"}
            )
            st.dataframe(liked_df, use_container_width=True, hide_index=True)
        else:
            st.info("No liked restaurants match the current filters. Adjust the filters to review or edit a saved like.")

        editable_entries = filtered_liked_entries
        edit_restaurant_id = None
        selected_edit_entry = None
        if editable_entries:
            edit_restaurant_id = st.selectbox(
                "Edit a liked restaurant",
                options=[entry["restaurant_id"] for entry in editable_entries],
                format_func=lambda rid: next(
                    (
                        f"{entry['name']} ({entry['rating']}/5)"
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

        updated_rating = st.selectbox(
            "Updated rating",
            options=[1, 2, 3, 4, 5],
            index=max(0, int(selected_edit_entry["rating"]) - 1) if selected_edit_entry else 4,
            key="liked_updated_rating",
            disabled=not editable_entries,
        )
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("Update rating", use_container_width=True, disabled=not edit_restaurant_id):
                for entry in liked_entries:
                    if entry["restaurant_id"] == edit_restaurant_id:
                        entry["rating"] = updated_rating
                        break
                st.session_state["recommendations_liked_entries"] = liked_entries
                profile = persist_liked_entries(profile_id, liked_entries, raw_df)
                st.success("Liked restaurant rating updated.")
                st.rerun()
        with action_col2:
            if st.button("Remove like", use_container_width=True, disabled=not edit_restaurant_id):
                liked_entries = [
                    entry for entry in liked_entries if entry["restaurant_id"] != edit_restaurant_id
                ]
                st.session_state["recommendations_liked_entries"] = liked_entries
                profile = persist_liked_entries(profile_id, liked_entries, raw_df)
                st.success("Liked restaurant removed.")
                st.rerun()
    else:
        st.caption("You have not liked any restaurants yet. Add a few above to personalize recommendations.")

    st.markdown("---")
    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state.get("optimal_k", 10))
    shared_algo = st.session_state.get("active_cluster_algorithm", "kmeans")
    shared_algo_display = {
        "kmeans": "our NumPy K-Means",
        "gmm": "Gaussian Mixture",
        "agglomerative": "Hierarchical / Ward",
    }.get(shared_algo, "our NumPy K-Means")
    st.caption(f"Cluster context follows the GIS Map setup: `{shared_algo_display}` with `K = {k}`.")

    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None

user_history = st.session_state["user_history"]

# ── Run clustering ────────────────────────────────────────────────────────────
with st.spinner("Updating recommendation context..."):
    cdf, kmeans, scaler, pca = get_clustered_data(
        raw_df, user_history, k=k,
        force=(st.session_state["clustered_df"] is None),
        algorithm=st.session_state.get("active_cluster_algorithm", "kmeans"),
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

predicted_cluster = predict_user_cluster(user_history, cdf, kmeans, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

# ── K-NN Recommendation Engine ────────────────────────────────────────────────
from utils.clustering import (
    build_feature_matrix,
    build_user_feature_vector,
    recommend_knn,
    recommend_per_liked_knn,
    apply_mmr,
    collect_liked_vectors,
    TOP_CUISINES,
    BOROUGH_LIST,
)
import plotly.graph_objects as go

st.markdown("---")
st.subheader("🧠 Your Taste Profile")

# Build the user feature vector and restaurant feature matrix
X_restaurants, feature_columns, df_feat = build_feature_matrix(raw_df)
user_vec = build_user_feature_vector(profile, raw_df)
liked_vectors, liked_metadata = collect_liked_vectors(profile, X_restaurants, df_feat)

# Show the user's taste profile as metrics
pref_cuisines = profile.get("favorite_cuisines", [])
pref_boroughs = profile.get("preferred_boroughs", [])
budget = profile.get("budget", "$$")
n_likes = len(profile.get("likes", []))

p1, p2, p3, p4 = st.columns(4)
p1.metric("Budget", format_budget_display(budget))
p2.metric("Liked Restaurants", n_likes)
p3.metric("Cuisines", ", ".join(pref_cuisines[:3]) if pref_cuisines else "Any")
p4.metric("Boroughs", ", ".join(pref_boroughs[:2]) if pref_boroughs else "All NYC")

if n_likes == 0 and not pref_cuisines:
    st.info("💡 Like some restaurants or set your cuisine preferences in the sidebar to get personalized recommendations. Showing top-rated restaurants for now.")

# Run K-NN with the selected method
liked_ids = {str(v) for v in user_history.get("visited_ids", [])}

st.markdown("---")
st.subheader("⚙️ Recommendation Method")
method_col, lambda_col = st.columns([1.1, 1])
with method_col:
    rec_method = st.radio(
        "Algorithm",
        [
            "Per-liked KNN + MMR (recommended)",
            "Profile-averaged KNN (legacy)",
        ],
        help=(
            "**Per-liked KNN + MMR** runs cosine KNN separately from each "
            "liked restaurant, fuses the rankings via Reciprocal Rank Fusion, "
            "then re-ranks the top-50 with Maximal Marginal Relevance so the "
            "final list isn't dominated by one cuisine.  "
            "**Profile-averaged KNN** collapses all likes into a single mean "
            "vector and picks top-K by cosine similarity — simpler but "
            "dilutes diverse tastes."
        ),
    )
with lambda_col:
    if rec_method.startswith("Per-liked"):
        mmr_lambda = st.slider(
            "MMR balance (λ)",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help=(
                "λ = 1.0 → pure relevance (may repeat similar picks).  "
                "λ = 0.0 → pure diversity (may include weaker matches).  "
                "0.7 is a standard default."
            ),
        )
    else:
        mmr_lambda = 1.0

if rec_method.startswith("Per-liked"):
    candidates = recommend_per_liked_knn(
        liked_vectors=liked_vectors,
        profile_vector=user_vec,
        restaurant_matrix=X_restaurants,
        restaurant_df=df_feat,
        visited_ids=liked_ids,
        liked_metadata=liked_metadata,
        k_per_liked=30,
        k_final=50,
        scaler=scaler,
    )
    if len(candidates) > 0:
        cand_matrix = np.vstack([
            X_restaurants[df_feat.index[df_feat["restaurant_id"].astype(str) == str(rid)][0]]
            for rid in candidates["restaurant_id"].astype(str)
        ])
        recs = apply_mmr(
            candidates, cand_matrix,
            user_vector=user_vec, k=15, lambda_=mmr_lambda, scaler=scaler,
        )
    else:
        recs = candidates
else:
    recs = recommend_knn(
        user_vec, X_restaurants, df_feat,
        visited_ids=liked_ids,
        k=15,
        scaler=scaler,
    )
    recs["primary_influencer"] = "Profile preferences"

# ── Recommendation cards ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎯 Recommended Restaurants")
st.caption(
    "Restaurants ranked by cosine similarity to your taste profile in the "
    "interpretable 22-dim feature space (price, cuisine, borough, rating, "
    "health, location).  The *Influenced by* column shows which liked "
    "restaurant contributed each pick's best rank under RRF."
)

if recs.empty:
    st.info("No unseen restaurants found outside your liked list. Try broadening your preferences or removing a like.")
else:
    unique_cuisines = recs["cuisine_type"].fillna("Other").nunique()
    unique_boroughs = recs["boro"].fillna("NYC").nunique()
    d1, d2, d3 = st.columns(3)
    d1.metric("Cuisine diversity", f"{unique_cuisines} unique")
    d2.metric("Borough coverage", f"{unique_boroughs} / 5")
    d3.metric("Method", "Per-liked + MMR" if rec_method.startswith("Per-liked") else "Profile avg.")

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
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expandable detail cards for top 5
    st.markdown("---")
    st.subheader("Top 5 Picks — Why These?")
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        sim_score = row.get("knn_similarity", 0)
        influencer = row.get("primary_influencer", "Profile preferences")
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
            r_cuisine = row.get("cuisine_type", "")
            if r_cuisine in pref_cuisines:
                reasons.append(f"matches your favorite cuisine ({r_cuisine})")
            r_boro = row.get("boro", "")
            if r_boro in pref_boroughs:
                reasons.append(f"located in your preferred borough ({r_boro})")
            r_price = row.get("price_tier", 2)
            budget_val = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}.get(budget, 2)
            if abs(r_price - budget_val) <= 0.5:
                reasons.append(f"matches your budget ({budget})")
            if row.get("avg_rating", 0) >= 4.5:
                reasons.append("exceptionally high rated")
            if reasons:
                st.markdown(f"**Why recommended:** {'; '.join(reasons)}")
            else:
                st.markdown("**Why recommended:** Similar overall taste profile based on cuisine, price, location, and quality signals.")

            if row.get("g_maps_url"):
                st.markdown(f"[📍 Open in Google Maps]({row['g_maps_url']})")

# ── PCA visualization with user position ──────────────────────────────────────
st.markdown("---")
st.subheader("📍 Your Position in Restaurant Space")
st.caption("Each dot is a restaurant, colored by cluster. The ⭐ shows where your taste profile sits in the PCA projection.")

if pca is not None and hasattr(pca, 'axis_labels_'):
    # Project all restaurants into PCA space
    from utils.clustering import prepare_clustering_space, _scaled_space
    X_scaled, X_cluster, _ = prepare_clustering_space(X_restaurants, scaler=scaler, fit=False)
    X_pca_all = pca.transform(X_scaled)

    # Project user vector into PCA space
    user_scaled = _scaled_space(user_vec, scaler)
    user_pca = pca.transform(user_scaled)[0]

    # Cluster colors
    CLUSTER_HEX = [
        "#6c8fff", "#ff9f43", "#6dda7f", "#ff6b8a",
        "#b983ff", "#ffd32a", "#48dbfb", "#ff6348",
        "#34d399", "#fb923c", "#a78bfa", "#22d3ee",
        "#f87171", "#4ade80", "#facc15", "#818cf8",
    ]

    fig = go.Figure()

    # Plot restaurants by cluster
    for cid in sorted(cdf["cluster_id"].unique()):
        mask = cdf["cluster_id"] == cid
        label = cdf[mask]["cluster_label"].iloc[0]
        color = CLUSTER_HEX[cid % len(CLUSTER_HEX)]
        indices = np.where(mask.values)[0]
        fig.add_trace(go.Scatter3d(
            x=X_pca_all[indices, 0],
            y=X_pca_all[indices, 1],
            z=X_pca_all[indices, 2],
            mode="markers",
            name=label,
            marker=dict(size=3, color=color, opacity=0.5),
            text=cdf[mask]["name"].values,
            hovertemplate="%{text}<extra>" + label + "</extra>",
        ))

    # Plot user position
    fig.add_trace(go.Scatter3d(
        x=[user_pca[0]], y=[user_pca[1]], z=[user_pca[2]],
        mode="markers+text",
        name="⭐ You",
        marker=dict(size=12, color="#FFD700", symbol="diamond",
                    line=dict(width=2, color="#000")),
        text=["⭐ You"],
        textposition="top center",
    ))

    # Draw lines from user to top-5 recommendations
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        # Find this restaurant in df_feat
        rid = str(row.get("restaurant_id", ""))
        idx_match = df_feat.index[df_feat["restaurant_id"].astype(str) == rid]
        if len(idx_match) > 0:
            idx = df_feat.index.get_loc(idx_match[0])
            fig.add_trace(go.Scatter3d(
                x=[user_pca[0], X_pca_all[idx, 0]],
                y=[user_pca[1], X_pca_all[idx, 1]],
                z=[user_pca[2], X_pca_all[idx, 2]],
                mode="lines",
                line=dict(color="#FFD700", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

    axis_labels = pca.axis_labels_ if hasattr(pca, 'axis_labels_') else ["PC1", "PC2", "PC3"]
    var_explained = pca.explained_variance_ratio_[:3] if hasattr(pca, 'explained_variance_ratio_') else [0, 0, 0]

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
    st.plotly_chart(fig, use_container_width=True)

# ── All clusters overview ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("All Cluster Profiles")

cluster_summary = cdf.groupby(["cluster_id", "cluster_label"]).agg(
    Count=("restaurant_id", "count"),
    Avg_Rating=("avg_rating", "mean"),
    Avg_Price=("price_tier", "mean"),
    Top_Cuisine=("cuisine_type", lambda x: x.value_counts().index[0]),
).reset_index()

cluster_summary["Avg_Rating"] = cluster_summary["Avg_Rating"].round(2)
cluster_summary["Avg_Price"]  = cluster_summary["Avg_Price"].round(1)
cluster_summary["Is My Cluster"] = cluster_summary["cluster_id"].apply(
    lambda x: "🎯 You" if x == predicted_cluster else ""
)
cluster_summary.columns = [c.replace("_", " ") for c in cluster_summary.columns]
st.dataframe(cluster_summary.drop(columns=["cluster id"]), use_container_width=True, hide_index=True)
