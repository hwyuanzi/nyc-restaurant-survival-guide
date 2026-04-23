import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

from utils.clustering import (
    apply_user_weights,
    build_feature_matrix,
    find_optimal_k,
    get_clustered_data,
    prepare_clustering_space,
    run_clustering,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import init_session_state, predict_user_cluster

init_session_state()

from utils.auth import require_auth
require_auth()

try:
    import pydeck as pdk
    PYDECK_OK = True
except ImportError:
    PYDECK_OK = False

CLUSTER_COLORS = [
    [108, 143, 255, 200],
    [255, 159, 67,  200],
    [109, 218, 127, 200],
    [255, 107, 138, 200],
    [185, 131, 255, 200],
    [255, 211, 42,  200],
    [72,  219, 251, 200],
    [255, 99,  72,  200],
    [52,  211, 153, 200],
    [251, 146, 60,  200],
    [167, 139, 250, 200],
    [34,  211, 238, 200],
    [248, 113, 113, 200],
    [74,  222, 128, 200],
    [250, 204, 21,  200],
    [129, 140, 248, 200],
]

st.title("📍 Restaurant Cluster GIS Map")
st.markdown("Restaurants colored by interpretable taste clusters on a real NYC map.")

ALGORITHM_OPTIONS = [
    ("KMeans", "kmeans"),
    ("GMM (tied covariance)", "gmm"),
    ("Hierarchical (Ward)", "agglomerative"),
]
ALGO_HELP = {
    "kmeans": "KMeans: fast and stable when clusters are approximately spherical.",
    "gmm": "GMM: soft probabilistic membership and elliptical clusters via tied covariance.",
    "agglomerative": "Hierarchical (Ward): bottom-up grouping that can better follow non-spherical structure.",
}

default_algo = st.session_state.get("clustering_algorithm", "kmeans")
default_algo_idx = [key for _, key in ALGORITHM_OPTIONS].index(default_algo) if default_algo in [key for _, key in ALGORITHM_OPTIONS] else 0
algorithm_label = st.radio(
    "Clustering algorithm",
    options=[label for label, _ in ALGORITHM_OPTIONS],
    index=default_algo_idx,
    horizontal=True,
)
algorithm_key = dict(ALGORITHM_OPTIONS)[algorithm_label]
st.session_state["clustering_algorithm"] = algorithm_key
st.caption(ALGO_HELP[algorithm_key])

if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading prepared restaurant data..."):
        _, _, runtime_df, _ = load_runtime_assets(DEFAULT_SEARCH_SAMPLE_SIZE)
    if runtime_df.empty:
        st.error("Prepared restaurant data could not be loaded.")
        st.stop()
    st.session_state["raw_df"] = runtime_df

raw_df       = st.session_state["raw_df"]
user_history = st.session_state["user_history"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Clustering Controls")
    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state["optimal_k"])
    cluster_filter_placeholder = st.empty()

    if st.button("🔍 Find Optimal K"):
        with st.spinner("Computing silhouette scores..."):
            X, _, _ = build_feature_matrix(raw_df)
            X_aug   = apply_user_weights(X, raw_df, user_history)
            _, X_cluster, _ = prepare_clustering_space(X_aug, fit=True)
            best_k  = find_optimal_k(X_cluster)
            st.session_state["optimal_k"] = best_k
            st.success(f"Optimal K = {best_k}")
            k = best_k

    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None
        st.session_state["kmeans_model"] = None

    st.markdown("---")
    st.markdown("### Map Controls")
    layer_type   = st.radio("Layer type", ["3D Columns", "Scatter Dots"])
    height_metric = st.selectbox("Column height by", ["avg_rating", "review_count", "user_affinity_score", "price_tier"])
    map_style_name = st.selectbox("Map style", ["Dark", "Light"])

MAP_STYLES = {
    "Dark":      "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Light":     "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
}

# ── Run clustering ────────────────────────────────────────────────────────────
last_algo = st.session_state.get("cluster_cache_algorithm")
algorithm_changed = last_algo != algorithm_key
with st.spinner(f"Running {algorithm_label} clustering..."):
    cdf, cluster_model, scaler, pca = get_clustered_data(
        raw_df,
        user_history,
        k=k,
        algorithm=algorithm_key,
        force=(st.session_state["clustered_df"] is None or algorithm_changed),
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = cluster_model
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca
    st.session_state["cluster_cache_algorithm"] = algorithm_key

predicted_cluster = predict_user_cluster(user_history, cdf, cluster_model, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

# ── User cluster banner ───────────────────────────────────────────────────────
if predicted_cluster != -1:
    cl_label = cdf[cdf["cluster_id"] == predicted_cluster]["cluster_label"].iloc[0]
    n_match  = len(cdf[cdf["cluster_id"] == predicted_cluster])
    st.success(f"🎯 Based on your history, you belong to **{cl_label}** — {n_match} restaurants match your taste profile.")

with st.expander("Compare silhouette"):
    st.caption("Run all three clustering methods on the same feature space and compare separation quality.")
    if st.button("Run silhouette comparison", key="run_silhouette_comparison"):
        X_base, _, _ = build_feature_matrix(raw_df)
        X_aug = apply_user_weights(X_base, raw_df, user_history)
        _, X_cluster_eval, _ = prepare_clustering_space(X_aug, fit=True)

        rows = []
        for label, algo in ALGORITHM_OPTIONS:
            cmp_df, _, _, _ = run_clustering(raw_df, user_history, k=k, algorithm=algo)
            labels_vec = cmp_df["cluster_id"].values
            score = -1.0
            if len(np.unique(labels_vec)) > 1:
                try:
                    score = float(
                        silhouette_score(
                            X_cluster_eval,
                            labels_vec,
                            sample_size=min(1500, len(X_cluster_eval)),
                            random_state=42,
                        )
                    )
                except Exception:
                    score = -1.0

            top_labels = (
                cmp_df.groupby("cluster_label")["restaurant_id"]
                .count()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            rows.append(
                {
                    "Algorithm": label,
                    "Silhouette": score,
                    "Clusters": int(cmp_df["cluster_id"].nunique()),
                    "Top-3 Cluster Labels": ", ".join(top_labels),
                }
            )

        st.dataframe(
            pd.DataFrame(rows).sort_values("Silhouette", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ── Prepare map data ──────────────────────────────────────────────────────────
map_df = cdf.dropna(subset=["lat", "lng"]).copy()
map_df = map_df[map_df["lat"].between(40.4774, 40.9176) & map_df["lng"].between(-74.2591, -73.7004)]

cluster_name_map = (
    map_df[["cluster_id", "cluster_label"]]
    .drop_duplicates()
    .sort_values(["cluster_label", "cluster_id"])
)
all_cluster_labels = cluster_name_map["cluster_label"].tolist()
with cluster_filter_placeholder.container():
    cluster_filter = st.multiselect("Show clusters", options=all_cluster_labels, default=all_cluster_labels)

if cluster_filter:
    map_df = map_df[map_df["cluster_label"].isin(cluster_filter)]

# Cluster colors with dimming
def get_color(cid):
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)].copy()
    if predicted_cluster != -1 and cid != predicted_cluster:
        return color[:3] + [60]
    return color

map_df["cluster_color_rgba"] = map_df["cluster_id"].apply(get_color)
map_df["scaled_radius"]      = map_df["cluster_id"].apply(lambda c: 80 if c == predicted_cluster else 40)

# Elevation scaling
h_col = height_metric if height_metric in map_df.columns else "avg_rating"
h_vals = pd.to_numeric(map_df[h_col], errors="coerce").fillna(0)
h_min, h_max = h_vals.min(), h_vals.max()
if h_max > h_min:
    map_df["elevation_value"] = ((h_vals - h_min) / (h_max - h_min) * 500).astype(int)
else:
    map_df["elevation_value"] = 100

# Tooltip-friendly columns
map_df["price_tier_str"] = map_df["price_tier"].apply(lambda x: "$" * int(x) if pd.notna(x) else "")
map_df["neighborhood"]   = map_df.get("neighborhood", map_df.get("boro", "NYC"))

if not PYDECK_OK:
    st.error("pydeck not installed. Run: pip install pydeck")
    st.stop()

view_state = pdk.ViewState(latitude=40.7128, longitude=-74.0060, zoom=11, pitch=45, bearing=0)

tooltip = {
    "html": "<b>{name}</b><br/>🍽 {cuisine_type} · {price_tier_str}<br/>⭐ {avg_rating}<br/>📍 {neighborhood}<br/><span style='color:#aaa'>Cluster: {cluster_label}</span>",
    "style": {"backgroundColor": "#1a1a2e", "color": "white", "fontSize": "13px", "padding": "8px 12px", "borderRadius": "8px"},
}

if layer_type == "3D Columns":
    layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position="[lng, lat]",
        get_elevation="elevation_value",
        elevation_scale=1,
        radius=60,
        get_fill_color="cluster_color_rgba",
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )
else:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lng, lat]",
        get_color="cluster_color_rgba",
        get_radius="scaled_radius",
        radius_min_pixels=3,
        radius_max_pixels=20,
        pickable=True,
        auto_highlight=True,
    )

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_provider="carto",
    map_style=MAP_STYLES[map_style_name],
)
st.pydeck_chart(deck)

# ── Cluster summary cards ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Summaries")

unique_clusters = sorted(cdf["cluster_id"].unique().tolist())
cols = st.columns(min(len(unique_clusters), 4))

for i, cid in enumerate(unique_clusters):
    col = cols[i % 4]
    subset = cdf[cdf["cluster_id"] == cid]
    label  = subset["cluster_label"].iloc[0]
    color  = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
    hex_c  = "#{:02x}{:02x}{:02x}".format(*color[:3])
    top3   = subset["cuisine_type"].value_counts().head(3).index.tolist()
    avg_r  = subset["avg_rating"].mean()

    with col:
        st.markdown(f"""
        <div style="border-left: 4px solid {hex_c}; padding: 0.6rem 0.8rem;
                    background:#1a1a2e; border-radius:8px; margin-bottom:0.5rem;">
          <div style="font-weight:700; color:#e0e0f0">{label}</div>
          <div style="font-size:.78rem; color:#7a7a9a">{len(subset)} restaurants</div>
          <div style="font-size:.78rem; color:#a0a0c0">{", ".join(top3)}</div>
          <div style="font-size:.78rem; color:#d19900">⭐ {avg_r:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Explore →", key=f"explore_{cid}"):
            st.session_state["selected_cluster_label"] = label
            st.info(f"Use the taste-cluster filter on Home or Semantic Search to explore {label}.")
