import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np

from utils.clustering import get_clustered_data
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import get_profile, init_session_state, predict_user_cluster, profile_to_user_history, render_profile_sidebar, upsert_profile

init_session_state()
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


def format_match_percent(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "—"
    clipped = float(np.clip(numeric_value, 0, 1))
    return f"{round(clipped * 100):d}%"


def describe_cluster_traits(cluster_df):
    if cluster_df.empty:
        return "Mixed restaurant attributes"

    parts = []
    borough_counts = cluster_df["boro"].fillna("").astype(str).value_counts()
    if not borough_counts.empty:
        top_borough = borough_counts.index[0]
        borough_share = borough_counts.iloc[0] / max(len(cluster_df), 1)
        if top_borough and borough_share >= 0.4:
            parts.append(f"{top_borough}-leaning")

    avg_rating = pd.to_numeric(cluster_df.get("avg_rating"), errors="coerce").mean()
    if pd.notna(avg_rating):
        if avg_rating >= 4.5:
            parts.append("top rated")
        elif avg_rating >= 4.1:
            parts.append("well rated")
        elif avg_rating >= 3.7:
            parts.append("reliable")

    avg_price = pd.to_numeric(cluster_df.get("price_tier"), errors="coerce").mean()
    if pd.notna(avg_price):
        if avg_price <= 1.5:
            parts.append("budget friendly")
        elif avg_price >= 3.0:
            parts.append("upscale")
        else:
            parts.append("mid-range")

    avg_reviews = pd.to_numeric(cluster_df.get("review_count"), errors="coerce").mean()
    if pd.notna(avg_reviews):
        if avg_reviews >= 400:
            parts.append("crowd favorites")
        elif avg_reviews >= 150:
            parts.append("local staples")
        else:
            parts.append("hidden gems")

    return ", ".join(dict.fromkeys(parts)) if parts else "Mixed restaurant attributes"


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
        st.session_state["user_history"] = profile_to_user_history(profile)
        st.session_state["clustered_df"] = None
        st.success("Visited history saved. Recommendations updated.")
        st.rerun()

    st.markdown("---")
    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state["optimal_k"])

    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None

user_history = st.session_state["user_history"]

# ── Run clustering ────────────────────────────────────────────────────────────
with st.spinner("Running K-Means clustering..."):
    cdf, kmeans, scaler, pca = get_clustered_data(
        raw_df, user_history, k=k,
        force=(st.session_state["clustered_df"] is None)
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

predicted_cluster = predict_user_cluster(user_history, cdf, kmeans, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

# ── Cluster profile banner ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 Your Cluster Profile")

if predicted_cluster == -1:
    st.info("No visit history found. Add visited restaurants in the sidebar to get personalized recommendations. Showing top-rated restaurants across all clusters instead.")
    recs = cdf.sort_values("avg_rating", ascending=False).head(10)
    cluster_label = "All Clusters"
else:
    cluster_label    = cdf[cdf["cluster_id"] == predicted_cluster]["cluster_label"].iloc[0]
    cluster_df       = cdf[cdf["cluster_id"] == predicted_cluster]
    n_in_cluster     = len(cluster_df)
    avg_cluster_r    = cluster_df["avg_rating"].mean()
    avg_cluster_p    = cluster_df["price_tier"].mean()
    cluster_traits   = describe_cluster_traits(cluster_df)

    st.success(f"🎯 You're a **{cluster_label}** explorer — {n_in_cluster} restaurants match your taste profile.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your Cluster", cluster_label)
    c2.metric("Restaurants", n_in_cluster)
    c3.metric("Avg Rating", f"{avg_cluster_r:.2f} ★")
    c4.metric("Avg Price", format_price_tier(avg_cluster_p, escape_dollars=True))

    st.markdown(f"**Cluster traits:** {cluster_traits}")

    unvisited = cluster_df[
        ~cluster_df["restaurant_id"].isin(user_history.get("visited_ids", []))
    ]
    recs = unvisited.sort_values("user_affinity_score", ascending=False)

# ── Recommendation cards ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Recommended for You — {cluster_label}")

if recs.empty:
    st.info("You've visited all restaurants in your cluster! Try increasing K or exploring other clusters.")
else:
    display_cols = ["name", "cuisine_type", "avg_rating", "price_tier", "cluster_label"]
    display_cols = [c for c in display_cols if c in recs.columns]

    if "user_affinity_score" in recs.columns:
        display_cols += ["user_affinity_score"]

    top_recs = recs.head(15)[display_cols].copy()

    if "avg_rating" in top_recs.columns:
        top_recs["avg_rating"] = top_recs["avg_rating"].apply(lambda x: f"{x:.2f} ★")
    if "price_tier" in top_recs.columns:
        top_recs["price_tier"] = top_recs["price_tier"].apply(format_price_tier)
    if "user_affinity_score" in top_recs.columns:
        top_recs["user_affinity_score"] = top_recs["user_affinity_score"].apply(
            format_match_percent
        )

    top_recs.columns = [c.replace("_", " ").title() for c in top_recs.columns]
    st.dataframe(top_recs, use_container_width=True, hide_index=True)

    # Expandable detail cards for top 5
    st.markdown("---")
    st.subheader("Top 5 Picks")
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        with st.expander(f"#{i+1} — {row.get('name', 'Unknown')}"):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cuisine", row.get("cuisine_type", "—"))
            col2.metric("Rating", f"{row.get('avg_rating', 0):.2f} ★")
            col3.metric("Price", format_price_tier(row.get("price_tier"), escape_dollars=True))
            col4.metric("Affinity", format_match_percent(row.get("user_affinity_score")))
            st.caption(f"📍 {row.get('address', row.get('boro', ''))}")
            st.caption(f"Cluster: {row.get('cluster_label', '—')}")
            if row.get("g_maps_url"):
                st.markdown(f"[📍 Open in Google Maps]({row['g_maps_url']})")

# ── All clusters overview ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("All Cluster Profiles")

cluster_summary = cdf.groupby(["cluster_id", "cluster_label"]).agg(
    Count=("restaurant_id", "count"),
    Avg_Rating=("avg_rating", "mean"),
    Avg_Price=("price_tier", "mean"),
    Top_Borough=("boro", lambda x: x.fillna("").value_counts().index[0] if not x.fillna("").empty else "NYC"),
).reset_index()

cluster_summary["Avg_Rating"] = cluster_summary["Avg_Rating"].round(2)
cluster_summary["Avg_Price"]  = cluster_summary["Avg_Price"].round(1)
cluster_summary["Is My Cluster"] = cluster_summary["cluster_id"].apply(
    lambda x: "🎯 You" if x == predicted_cluster else ""
)
cluster_summary.columns = [c.replace("_", " ") for c in cluster_summary.columns]
st.dataframe(cluster_summary.drop(columns=["cluster id"]), use_container_width=True, hide_index=True)
