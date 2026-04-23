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
    build_user_feature_vector,
    get_clustered_data,
    recommend_knn,
    recommend_per_liked_knn,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import get_profile, init_session_state, predict_user_cluster, profile_to_user_history, render_profile_sidebar, upsert_profile

st.set_page_config(page_title="Cluster-Based Recommendations", page_icon="🔮", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

st.title("🔮 Cluster-Based Recommendations")
st.markdown("Restaurants recommended based on your taste cluster profile.")


def format_price_tier(value, escape_dollars=False):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value) or numeric_value <= 0:
        return "—"
    price_text = "$" * int(round(float(numeric_value)))
    return price_text.replace("$", r"\$") if escape_dollars else price_text


def build_restaurant_option(row):
    area = row.get("boro") or row.get("neighborhood") or "NYC"
    return f"{row.get('name', 'Unknown')} · {row.get('cuisine_type', 'Unknown')} · {area}"


def scale_recommendation_vectors(vectors: np.ndarray, scaler):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if scaler is None:
        return vectors
    augmented = np.hstack([vectors, np.zeros((vectors.shape[0], 1), dtype=np.float32)])
    return scaler.transform(augmented)[:, :vectors.shape[1]]


def resolve_liked_ids_and_names(profile, feature_df):
    likes = profile.get("likes", [])
    lookup_df = feature_df.drop_duplicates(subset=["restaurant_id"]).copy()
    lookup_df["restaurant_id"] = lookup_df["restaurant_id"].astype(str)
    by_name = lookup_df.copy()
    by_name["_norm_name"] = by_name["name"].fillna("").str.strip().str.lower()

    liked_ids = []
    liked_name_by_id = {}

    for like in likes:
        raw_id = str(like.get("restaurant_id", "")).strip()
        if raw_id and raw_id in set(lookup_df["restaurant_id"]):
            liked_ids.append(raw_id)
            liked_name_by_id[raw_id] = like.get("dba") or like.get("name") or raw_id
            continue

        like_name = str(like.get("dba") or like.get("name") or "").strip().lower()
        if not like_name:
            continue
        matches = by_name[by_name["_norm_name"] == like_name]
        if matches.empty:
            continue
        resolved_id = str(matches.iloc[0]["restaurant_id"])
        liked_ids.append(resolved_id)
        liked_name_by_id[resolved_id] = like.get("dba") or like.get("name") or str(matches.iloc[0]["name"])

    # Stable de-duplication while preserving order
    seen = set()
    deduped = []
    for rid in liked_ids:
        if rid in seen:
            continue
        deduped.append(rid)
        seen.add(rid)
    return deduped, liked_name_by_id


def load_visit_entries(profile, df):
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
    st.markdown("### Visited Restaurants")
    st.caption("Add places you've been to so recommendations can learn from your actual history.")

    session_profile_key = st.session_state.get("recommendations_history_profile_id")
    session_profile_updated_at = st.session_state.get("recommendations_history_profile_updated_at")
    if session_profile_key != profile_id or session_profile_updated_at != profile_updated_at:
        st.session_state["recommendations_history_profile_id"] = profile_id
        st.session_state["recommendations_history_profile_updated_at"] = profile_updated_at
        st.session_state["recommendations_visit_entries"] = load_visit_entries(profile, raw_df)

    visit_entries = st.session_state.get("recommendations_visit_entries", [])

    search_query = st.text_input(
        "Search restaurants",
        placeholder="Type a name, cuisine, borough, or address",
        key="recommendations_restaurant_search",
    ).strip().lower()

    search_df = raw_df.drop_duplicates(subset=["restaurant_id"]).copy()
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

    if st.button("Add visited restaurant", use_container_width=True, disabled=not selected_restaurant_id):
        selected_row = search_df.loc[search_df["restaurant_id"].astype(str) == str(selected_restaurant_id)].iloc[0]
        updated = False
        for entry in visit_entries:
            if entry["restaurant_id"] == str(selected_restaurant_id):
                entry["rating"] = selected_rating
                updated = True
                break
        if not updated:
            visit_entries.append({
                "restaurant_id": str(selected_row["restaurant_id"]),
                "name": selected_row.get("name", "Unknown"),
                "cuisine_type": selected_row.get("cuisine_type", ""),
                "boro": selected_row.get("boro", ""),
                "address": selected_row.get("address", ""),
                "rating": selected_rating,
            })
            visit_entries.sort(key=lambda item: (item["name"].lower(), item["restaurant_id"]))
        st.session_state["recommendations_visit_entries"] = visit_entries
        if updated:
            st.info("That restaurant is already in your history, so its rating was updated instead.")
        else:
            st.success("Restaurant added to your visited history.")
        st.rerun()

    if visit_entries:
        visit_df = pd.DataFrame(visit_entries)[["name", "cuisine_type", "boro", "rating"]].rename(
            columns={"name": "Restaurant", "cuisine_type": "Cuisine", "boro": "Borough", "rating": "Rating"}
        )
        st.dataframe(visit_df, use_container_width=True, hide_index=True)

        remove_restaurant_id = st.selectbox(
            "Remove a saved visit",
            options=[entry["restaurant_id"] for entry in visit_entries],
            format_func=lambda rid: next(
                (
                    f"{entry['name']} ({entry['rating']}/5)"
                    for entry in visit_entries
                    if entry["restaurant_id"] == rid
                ),
                rid,
            ),
            index=None,
            placeholder="Select a restaurant to remove",
        )
        if st.button("Remove selected visit", use_container_width=True):
            st.session_state["recommendations_visit_entries"] = [
                entry for entry in visit_entries if entry["restaurant_id"] != remove_restaurant_id
            ]
            st.rerun()

    if st.button("Save visited history", use_container_width=True):
        profile = get_profile(profile_id=profile_id)
        existing_likes = {str(item.get("restaurant_id")): item for item in profile.get("likes", []) if item.get("restaurant_id")}
        updated_likes_by_restaurant = {}

        for entry in st.session_state.get("recommendations_visit_entries", []):
            restaurant_id = entry["restaurant_id"]
            existing_like = existing_likes.get(restaurant_id, {})
            updated_likes_by_restaurant[restaurant_id] = {
                **existing_like,
                "restaurant_id": restaurant_id,
                "dba": entry["name"],
                "cuisine": entry["cuisine_type"],
                "boro": entry["boro"],
                "rating": float(entry["rating"]),
                "source": existing_like.get("source", "manual_history"),
                "liked_at": existing_like.get("liked_at") or pd.Timestamp.now().isoformat(timespec="seconds"),
            }

        profile["likes"] = list(updated_likes_by_restaurant.values())
        profile = upsert_profile(profile)
        st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
        st.session_state["clustered_df"] = None
        st.success("Visited history saved. Recommendations updated.")
        st.rerun()

    st.markdown("---")
    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state.get("optimal_k", 10))

    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None

user_history = st.session_state["user_history"]

# ── Run clustering ────────────────────────────────────────────────────────────
with st.spinner("Running K-Means clustering..."):
    cdf, cluster_model, scaler, pca = get_clustered_data(
        raw_df, user_history, k=k,
        force=(st.session_state["clustered_df"] is None)
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = cluster_model
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

predicted_cluster = predict_user_cluster(user_history, cdf, cluster_model, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

# ── K-NN Recommendation Engine ────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Your Taste Profile")

# Build the user feature vector and restaurant feature matrix
X_restaurants, feature_columns, df_feat = build_feature_matrix(raw_df)
user_vec = build_user_feature_vector(profile, raw_df)

if X_restaurants.shape[0] == 0:
    st.warning("No restaurants are available in the current dataset slice, so recommendations cannot be generated yet.")
    st.stop()

# Show the user's taste profile as metrics
pref_cuisines = profile.get("favorite_cuisines", [])
pref_boroughs = profile.get("preferred_boroughs", [])
budget = profile.get("budget", "$$")
n_likes = len(profile.get("likes", []))

p1, p2, p3, p4 = st.columns(4)
p1.metric("Budget", budget)
p2.metric("Liked Restaurants", n_likes)
p3.metric("Cuisines", ", ".join(pref_cuisines[:3]) if pref_cuisines else "Any")
p4.metric("Boroughs", ", ".join(pref_boroughs[:2]) if pref_boroughs else "All NYC")

if n_likes == 0 and not pref_cuisines:
    st.info("💡 Like some restaurants or set your cuisine preferences in the sidebar to get personalized recommendations. Showing top-rated restaurants for now.")

# Recommendation method controls
st.markdown("---")
method = st.radio(
    "Recommendation method",
    options=["Per-liked KNN + MMR (default)", "Profile-average KNN (legacy)"],
    index=0,
    horizontal=True,
)
mmr_lambda = st.slider(
    "MMR λ (relevance ↔ diversity)",
    min_value=0.3,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Lower λ increases diversity; λ=1.0 disables diversity reranking.",
)

visited_ids = {str(v) for v in user_history.get("visited_ids", [])}
liked_ids, liked_name_by_id = resolve_liked_ids_and_names(profile, df_feat)
liked_rows = df_feat[df_feat["restaurant_id"].astype(str).isin(liked_ids)]
liked_vectors = X_restaurants[liked_rows.index.values]
liked_ids_aligned = liked_rows["restaurant_id"].astype(str).tolist()

if method.startswith("Per-liked"):
    candidates = recommend_per_liked_knn(
        liked_vectors=liked_vectors,
        profile_vector=user_vec,
        restaurant_matrix=X_restaurants,
        restaurant_df=df_feat,
        visited_ids=visited_ids,
        k_per_liked=30,
        k_final=50,
        scaler=scaler,
        liked_ids=liked_ids_aligned,
    )

    if candidates.empty:
        recs = candidates
    else:
        matrix_lookup = pd.Series(np.arange(len(df_feat)), index=df_feat["restaurant_id"].astype(str))
        candidate_indices = [int(matrix_lookup[str(rid)]) for rid in candidates["restaurant_id"].astype(str)]
        candidate_matrix_raw = X_restaurants[candidate_indices]
        candidate_matrix_scaled = scale_recommendation_vectors(candidate_matrix_raw, scaler=scaler)
        user_vec_scaled = scale_recommendation_vectors(user_vec, scaler=scaler)[0]
        recs = apply_mmr(
            candidates_df=candidates,
            candidate_matrix=candidate_matrix_scaled,
            user_vector=user_vec_scaled,
            k=15,
            lambda_=mmr_lambda,
        )
else:
    recs = recommend_knn(
        user_vec,
        X_restaurants,
        df_feat,
        visited_ids=visited_ids,
        k=15,
        scaler=scaler,
    )
    recs["primary_influencer_id"] = None
    recs["primary_influencer_idx"] = -1

if "primary_influencer_id" in recs.columns:
    recs["primary_influencer_name"] = recs["primary_influencer_id"].map(liked_name_by_id).fillna("")
else:
    recs["primary_influencer_name"] = ""

# ── Recommendation cards ──────────────────────────────────────────────────────
st.subheader("🎯 Recommended Restaurants")
if method.startswith("Per-liked"):
    st.caption("Candidates are fused across each liked restaurant using reciprocal-rank fusion, then MMR reranks for diversity.")
else:
    st.caption("Legacy baseline: profile-average cosine KNN in interpretable feature space.")

if recs.empty:
    st.info("No unvisited restaurants found. Try clearing your visit history or adjusting preferences.")
else:
    cuisine_diversity = int(recs["cuisine_type"].fillna("Unknown").nunique())
    st.metric("Cuisine diversity (top 15)", f"{cuisine_diversity} unique cuisines")

    display_cols = ["name", "cuisine_type", "avg_rating", "price_tier", "boro", "knn_similarity"]
    if "primary_influencer_name" in recs.columns:
        display_cols.append("primary_influencer_name")
    display_df = recs[display_cols].copy()
    display_df["avg_rating"] = display_df["avg_rating"].apply(lambda x: f"{x:.2f} ★")
    display_df["price_tier"] = display_df["price_tier"].apply(format_price_tier)
    display_df["knn_similarity"] = display_df["knn_similarity"].apply(lambda x: f"{x:.3f}")
    if "primary_influencer_name" in display_df.columns:
        display_df["primary_influencer_name"] = display_df["primary_influencer_name"].replace("", "—")
        display_df.columns = ["Restaurant", "Cuisine", "Rating", "Price", "Borough", "Similarity", "Because you liked"]
    else:
        display_df.columns = ["Restaurant", "Cuisine", "Rating", "Price", "Borough", "Similarity"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Expandable detail cards for top 5
    st.markdown("---")
    st.subheader("Top 5 Picks — Why These?")
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        sim_score = row.get("knn_similarity", 0)
        with st.expander(f"#{i+1} — {row.get('name', 'Unknown')} (Similarity: {sim_score:.3f})"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cuisine", row.get("cuisine_type", "—"))
            col2.metric("Rating", f"{row.get('avg_rating', 0):.2f} ★")
            col3.metric("Price", format_price_tier(row.get("price_tier"), escape_dollars=True))
            col4.metric("Borough", row.get("boro", "—"))
            st.caption(f"📍 {row.get('address', row.get('boro', ''))}")

            influencer_name = str(row.get("primary_influencer_name", "")).strip()
            if influencer_name:
                st.markdown(
                    f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
                    f"background:#eef4ff;color:#2f5fd1;font-size:12px;'>Because you liked {influencer_name}</span>",
                    unsafe_allow_html=True,
                )
            
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
    from utils.clustering import apply_user_weights, prepare_clustering_space
    X_aug = apply_user_weights(X_restaurants, df_feat, user_history)
    projection_feature_columns = feature_columns + ["user_affinity"]
    X_scaled, X_cluster, _ = prepare_clustering_space(X_aug, scaler=scaler, fit=False)
    X_pca_all = pca.transform(X_scaled)

    # Project user vector into PCA space
    user_aug = np.append(user_vec, [0.5])  # user_affinity placeholder
    user_scaled = scaler.transform(user_aug.reshape(1, -1))
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
