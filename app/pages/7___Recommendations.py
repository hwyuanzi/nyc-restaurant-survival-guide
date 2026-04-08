"""
7_🔮_Recommendations.py — Cluster-based personalized recommendations.

Reclusters the real NYC restaurant dataset via K-Means, predicts the user's
taste cluster from their saved history, and ranks restaurants by affinity.

Original clustering & recommendation pipeline: Rahul Adusumalli
Integrated by: Ryan Han (PapTR)

Course topics: Week 7 (K-Means Clustering)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np

from utils.clustering import get_clustered_data
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import init_session_state, predict_user_cluster

st.set_page_config(page_title="Recommendations", page_icon="🔮", layout="wide")

from app.ui_utils import apply_apple_theme
apply_apple_theme()

init_session_state()
st.title("🔮 Cluster-Based Recommendations")
st.markdown("Restaurants recommended based on your taste cluster profile, powered by real NYC data.")

if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading prepared restaurant data..."):
        _, _, runtime_df, _ = load_runtime_assets(DEFAULT_SEARCH_SAMPLE_SIZE)
    if runtime_df.empty:
        st.error("Prepared restaurant data could not be loaded.")
        st.stop()
    st.session_state["raw_df"] = runtime_df

raw_df       = st.session_state["raw_df"]
user_history = st.session_state["user_history"]

with st.sidebar:
    st.markdown("### Clustering Controls")
    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state["optimal_k"])
    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None

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

st.markdown("---")
st.subheader("🧠 Your Cluster Profile")

if predicted_cluster == -1:
    st.info("No visit history found. Showing top-rated restaurants across all clusters. Use the Live Search page to like some restaurants first!")
    recs = cdf.sort_values("avg_rating", ascending=False).head(10)
    cluster_label = "All Clusters"
else:
    cluster_label    = cdf[cdf["cluster_id"] == predicted_cluster]["cluster_label"].iloc[0]
    cluster_df       = cdf[cdf["cluster_id"] == predicted_cluster]
    n_in_cluster     = len(cluster_df)
    top_cuisines     = cluster_df["cuisine_type"].value_counts().head(3).index.tolist()
    avg_cluster_r    = cluster_df["avg_rating"].mean()
    avg_cluster_p    = cluster_df["price_tier"].mean()

    st.success(f"🎯 You're a **{cluster_label}** explorer — {n_in_cluster} restaurants match your taste profile.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Your Cluster", cluster_label)
    c2.metric("Restaurants", n_in_cluster)
    c3.metric("Avg Rating", f"{avg_cluster_r:.2f} ★")
    c4.metric("Avg Price", "$" * round(avg_cluster_p) if pd.notna(avg_cluster_p) else "—")

    st.markdown(f"**Top cuisines in your cluster:** {', '.join(top_cuisines)}")

    unvisited = cluster_df[
        ~cluster_df["restaurant_id"].isin(user_history.get("visited_ids", []))
    ]
    recs = unvisited.sort_values("user_affinity_score", ascending=False)

st.markdown("---")
st.subheader(f"Recommended for You — {cluster_label}")

if recs.empty:
    st.info("You've visited all restaurants in your cluster! Try increasing K.")
else:
    display_cols = ["name", "cuisine_type", "avg_rating", "price_tier", "cluster_label"]
    display_cols = [c for c in display_cols if c in recs.columns]
    if "user_affinity_score" in recs.columns:
        display_cols += ["user_affinity_score"]

    top_recs = recs.head(15)[display_cols].copy()
    if "avg_rating" in top_recs.columns:
        top_recs["avg_rating"] = top_recs["avg_rating"].apply(lambda x: f"{x:.2f} ★")
    if "price_tier" in top_recs.columns:
        top_recs["price_tier"] = top_recs["price_tier"].apply(lambda x: "$" * int(x) if pd.notna(x) and x > 0 else "—")
    if "user_affinity_score" in top_recs.columns:
        top_recs["user_affinity_score"] = top_recs["user_affinity_score"].apply(lambda x: f"{x:.3f}")

    top_recs.columns = [c.replace("_", " ").title() for c in top_recs.columns]
    st.dataframe(top_recs, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top 5 Picks")
    for i, (_, row) in enumerate(recs.head(5).iterrows()):
        with st.expander(f"#{i+1} — {row.get('name', 'Unknown')}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Cuisine", row.get("cuisine_type", "—"))
            col2.metric("Rating", f"{row.get('avg_rating', 0):.2f} ★")
            col3.metric("Price", "$" * int(row.get("price_tier", 1)) if pd.notna(row.get("price_tier")) else "—")
            st.caption(f"📍 {row.get('address', row.get('boro', ''))}")
            st.caption(f"Cluster: {row.get('cluster_label', '—')}")

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
