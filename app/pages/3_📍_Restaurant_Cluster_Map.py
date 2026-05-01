import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui_utils import apply_apple_theme
from utils.clustering import (
    get_clustered_data, find_optimal_k,
    find_silhouette_knee, find_inertia_elbow, CLUSTER_SCHEMA_VERSION,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import init_session_state

st.set_page_config(
    page_title="Restaurant Cluster Map",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_apple_theme()
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

ALGO_OPTIONS = {
    "K-Means (fast, spherical clusters)": "kmeans",
    "Gaussian Mixture (tied cov., elliptical soft clusters)": "gmm",
    "Hierarchical / Ward (merges by variance)": "agglomerative",
}

st.title("📍 Restaurant Cluster GIS Map")
st.markdown("Restaurants colored by cluster on a real NYC map.  Switch the "
            "clustering algorithm in the sidebar to see how the geometry of "
            "the feature space changes the groupings.  The default K-Means path "
            "uses our own NumPy implementation, while GMM and Ward are shown as "
            "comparison baselines on the same interpretable restaurant features.")

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
    algo_label = st.radio(
        "Algorithm",
        list(ALGO_OPTIONS.keys()),
        help=(
            "KMeans minimises Euclidean variance within each cluster — fast "
            "and tends to produce spherical groups.  GMM models each cluster "
            "as a Gaussian (tied covariance shares one shape across "
            "components).  Hierarchical / Ward greedily merges the pair of "
            "clusters that minimises total within-cluster variance, which "
            "often produces more natural groupings on mixed feature types."
        ),
    )
    algorithm = ALGO_OPTIONS[algo_label]
    if st.session_state.get("active_cluster_algorithm") != algorithm:
        # Algorithm changed — drop the cached clustered_df so the new algo runs.
        st.session_state["clustered_df"] = None
        st.session_state["kmeans_model"] = None
        st.session_state["active_cluster_algorithm"] = algorithm

    k = st.slider("Number of Clusters (K)", 4, 16, st.session_state["optimal_k"])
    cluster_filter_placeholder = st.empty()

    if st.button("🔍 Find Optimal K (K=4…15)"):
        with st.spinner("Sweeping K=4…15, computing silhouette + inertia…"):
            from utils.clustering import build_feature_matrix, prepare_clustering_space
            X, _, _ = build_feature_matrix(raw_df)
            _, X_cluster, _ = prepare_clustering_space(X, fit=True)
            best_k, k_scores = find_optimal_k(
                X_cluster, algorithm=algorithm, return_scores=True
            )
            st.session_state["optimal_k"] = best_k
            st.session_state["k_selection_scores"] = k_scores
            st.session_state["k_selection_algo"] = algorithm
            k = best_k

    cluster_request = (algorithm, int(k), CLUSTER_SCHEMA_VERSION)
    if st.session_state.get("active_cluster_request") != cluster_request:
        st.session_state["clustered_df"] = None
        st.session_state["kmeans_model"] = None
        st.session_state["active_cluster_request"] = cluster_request
        st.session_state["optimal_k"] = int(k)

    st.markdown("---")
    st.markdown("### Map Controls")
    height_metric = st.selectbox("Column height by", ["avg_rating", "review_count", "user_affinity_score", "price_tier"])
    map_style_name = st.selectbox("Map style", ["Dark", "Light"])

MAP_STYLES = {
    "Dark":      "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    "Light":     "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
}

# ── Run clustering ────────────────────────────────────────────────────────────
algo_display = {v: k_ for k_, v in ALGO_OPTIONS.items()}[algorithm]
with st.spinner(f"Running {algo_display}..."):
    cdf, kmeans, scaler, pca = get_clustered_data(
        raw_df, user_history, k=k,
        force=(st.session_state["clustered_df"] is None),
        algorithm=algorithm,
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

# ── Effective cluster count note ─────────────────────────────────────────────
_n_effective = cdf["cluster_id"].nunique()
if _n_effective < k:
    st.info(
        f"ℹ️ Requested K = {k}, effective clusters = **{_n_effective}** "
        f"because the selected algorithm produced empty duplicate assignments. "
        f"No post-hoc cluster merging is applied."
    )

# ── K-selection analysis (shown after "Find Optimal K" is clicked) ────────────
_k_scores = st.session_state.get("k_selection_scores")
if _k_scores:
    _k_algo = st.session_state.get("k_selection_algo", algorithm)
    _best_k = st.session_state["optimal_k"]
    _k_sil_knee  = find_silhouette_knee(_k_scores)
    _k_elbow     = find_inertia_elbow(_k_scores)

    with st.expander("📐 K Selection Analysis — Elbow & Silhouette", expanded=True):
        st.caption(
            "**How we choose K (two independent signals):** \n\n"
            "① **Silhouette knee** — first local maximum in the silhouette curve "
            "**where no single cluster exceeds 35% of the dataset** (red dots = "
            "degenerate catch-all; skipped as candidates). We avoid the global maximum "
            "because silhouette rises when K is too small and one cluster absorbs half "
            "the restaurants — that is not good clustering, just artificial tightness.\n\n"
            "② **Inertia elbow** (K-Means only) — first K where the per-step WCSS "
            "improvement drops below 15% of the largest single-step improvement. "
            "This marks the 'diminishing returns' knee in the elbow curve.\n\n"
            "**Final K = min(silhouette knee, inertia elbow)** — the more conservative "
            "estimate. 🟡 gold = chosen K · 🟢 green = silhouette knee · "
            "🔴 red = catch-all cluster (> 35%, skipped) · 🔵 blue = normal."
        )
        _ks    = [r["k"] for r in _k_scores]
        _sils  = [r["silhouette"] for r in _k_scores]
        _inerts = [r["inertia"] for r in _k_scores]
        _fracs  = [r.get("max_cluster_fraction", 0.0) for r in _k_scores]
        has_inertia = any(v is not None for v in _inerts)

        from utils.clustering import MAX_CLUSTER_FRACTION_THRESHOLD
        # Highlight dots: chosen=gold, knee=green, balance-fail=red, else blue
        _sil_colors = []
        for k_, frac in zip(_ks, _fracs):
            if k_ == _best_k:
                _sil_colors.append("#facc15")    # gold = final chosen
            elif k_ == _k_sil_knee:
                _sil_colors.append("#4ade80")    # green = silhouette knee
            elif frac > MAX_CLUSTER_FRACTION_THRESHOLD:
                _sil_colors.append("#f87171")    # red = catch-all cluster (degenerate)
            else:
                _sil_colors.append("#6c8fff")    # blue = normal

        if has_inertia:
            _fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "① Silhouette Score (↑ better) — pick the first local max",
                    "② Inertia / WCSS (↓ better) — pick the elbow",
                ],
            )
        else:
            _fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=["① Silhouette Score (↑ better) — pick the first local max"],
            )

        _fig.add_trace(go.Scatter(
            x=_ks, y=_sils, mode="lines+markers",
            name="Silhouette",
            line=dict(color="#6c8fff", width=2),
            marker=dict(size=9, color=_sil_colors),
            hovertemplate="K=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
        ), row=1, col=1)
        # Annotation: silhouette knee
        if _k_sil_knee in _ks:
            _sil_val = _sils[_ks.index(_k_sil_knee)]
            _fig.add_annotation(
                x=_k_sil_knee, y=_sil_val, text=f"Knee K={_k_sil_knee}",
                showarrow=True, arrowhead=2, ax=30, ay=-30,
                font=dict(color="#4ade80", size=11), row=1, col=1,
            )
        # Annotation: chosen
        _fig.add_vline(x=_best_k, line_dash="dot", line_color="#facc15",
                       annotation_text=f"Chosen K={_best_k}",
                       annotation_font_color="#facc15", row=1, col=1)

        if has_inertia:
            _inert_colors = [
                "#facc15" if k_ == _best_k else
                "#fb923c" if k_ == _k_elbow else
                "#ff9f43"
                for k_ in _ks
            ]
            _fig.add_trace(go.Scatter(
                x=_ks, y=_inerts, mode="lines+markers",
                name="Inertia",
                line=dict(color="#ff9f43", width=2),
                marker=dict(size=9, color=_inert_colors),
                hovertemplate="K=%{x}<br>Inertia=%{y:,.0f}<extra></extra>",
            ), row=1, col=2)
            if _k_elbow is not None and _k_elbow in _ks:
                _iert_val = _inerts[_ks.index(_k_elbow)]
                _fig.add_annotation(
                    x=_k_elbow, y=_iert_val, text=f"Elbow K={_k_elbow}",
                    showarrow=True, arrowhead=2, ax=30, ay=-30,
                    font=dict(color="#fb923c", size=11), row=1, col=2,
                )
            _fig.add_vline(x=_best_k, line_dash="dot", line_color="#facc15",
                           row=1, col=2)

        _fig.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0"),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
        )
        for _axis in ["xaxis", "xaxis2"]:
            _fig.update_layout(**{_axis: dict(
                gridcolor="#2a2a38", title="K (number of clusters)", tickvals=_ks,
            )})
        for _axis in ["yaxis", "yaxis2"]:
            _fig.update_layout(**{_axis: dict(gridcolor="#2a2a38")})
        st.plotly_chart(_fig, width="stretch")

        _algo_display = {
            "kmeans": "K-Means (scratch NumPy)", "gmm": "GMM (tied covariance)",
            "agglomerative": "Hierarchical Ward",
        }.get(_k_algo, _k_algo)
        _chosen_sil = _sils[_ks.index(_best_k)] if _best_k in _ks else float("nan")

        if _k_elbow is not None:
            st.success(
                f"**Chosen K = {_best_k}** = min(silhouette knee K={_k_sil_knee}, "
                f"inertia elbow K={_k_elbow}) for {_algo_display} "
                f"on {len(raw_df):,} restaurants "
                f"(silhouette at K={_best_k}: **{_chosen_sil:.4f}**)."
            )
        else:
            st.success(
                f"**Chosen K = {_best_k}** (silhouette knee) "
                f"for {_algo_display} on {len(raw_df):,} restaurants "
                f"(silhouette: **{_chosen_sil:.4f}**)."
            )

# ── Prepare map data ──────────────────────────────────────────────────────────
map_df = cdf.dropna(subset=["lat", "lng"]).copy()
map_df = map_df[map_df["lat"].between(40.4774, 40.9176) & map_df["lng"].between(-74.2591, -73.7004)]

cluster_name_map = (
    map_df[["cluster_id", "cluster_label"]]
    .drop_duplicates()
    .sort_values(["cluster_id"])
)
cluster_label_by_id = dict(zip(cluster_name_map["cluster_id"], cluster_name_map["cluster_label"]))
count_by_cluster_id = cdf["cluster_id"].value_counts().to_dict()
all_cluster_ids = cluster_name_map["cluster_id"].astype(int).tolist()
pending_filter_ids = st.session_state.pop("pending_map_cluster_filter_ids", None)

if pending_filter_ids is not None:
    st.session_state["map_cluster_filter_ids"] = [
        int(cid) for cid in pending_filter_ids if int(cid) in set(all_cluster_ids)
    ]
    st.session_state["map_cluster_filter_k"] = int(k)
elif st.session_state.get("map_cluster_filter_k") != int(k):
    st.session_state["map_cluster_filter_ids"] = all_cluster_ids
    st.session_state["map_cluster_filter_k"] = int(k)
elif "map_cluster_filter_ids" not in st.session_state:
    st.session_state["map_cluster_filter_ids"] = all_cluster_ids
else:
    valid_ids = set(all_cluster_ids)
    st.session_state["map_cluster_filter_ids"] = [
        int(cid) for cid in st.session_state["map_cluster_filter_ids"] if int(cid) in valid_ids
    ]

def format_cluster_option(cluster_id):
    label = cluster_label_by_id.get(int(cluster_id), f"Cluster {int(cluster_id) + 1}")
    count = count_by_cluster_id.get(int(cluster_id), 0)
    return f"Cluster {int(cluster_id) + 1} · {label} ({count} restaurants)"

with cluster_filter_placeholder.container():
    cluster_filter_ids = st.multiselect(
        "Show clusters",
        options=all_cluster_ids,
        format_func=format_cluster_option,
        key="map_cluster_filter_ids",
    )

if cluster_filter_ids:
    map_df = map_df[map_df["cluster_id"].isin(cluster_filter_ids)]
else:
    st.warning("No clusters selected. Use the filter in the sidebar to show clusters again.")
    map_df = map_df.iloc[0:0]

# Cluster colors with dimming
def get_color(cid):
    color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)].copy()
    return color

map_df["cluster_color_rgba"] = map_df["cluster_id"].apply(get_color)
map_df["scaled_radius"]      = 40

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
    story  = subset["cluster_story"].iloc[0] if "cluster_story" in subset.columns else ""
    drivers = subset["cluster_key_drivers"].iloc[0] if "cluster_key_drivers" in subset.columns else ""

    with col:
        st.markdown(f"""
        <div style="border-left: 4px solid {hex_c}; padding: 0.6rem 0.8rem;
                    background:#1a1a2e; border-radius:8px; margin-bottom:0.5rem;">
          <div style="font-weight:700; color:#e0e0f0">{label}</div>
          <div style="font-size:.78rem; color:#7a7a9a">{len(subset)} restaurants</div>
          <div style="font-size:.78rem; color:#a0a0c0">{", ".join(top3)}</div>
          <div style="font-size:.78rem; color:#d19900">⭐ {avg_r:.2f}</div>
          <div style="font-size:.74rem; color:#93c5fd; margin-top:0.25rem;">{drivers}</div>
        </div>
        """, unsafe_allow_html=True)
        if story:
            st.caption(story)
