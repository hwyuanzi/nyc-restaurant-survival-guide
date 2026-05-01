import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_score

from app.ui_utils import apply_apple_theme
from utils.clustering import (
    build_feature_matrix,
    get_clustered_data,
    get_pca_axis_labels,
    prepare_clustering_space,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import (
    get_active_profile,
    get_valid_borough_options,
    get_valid_cuisine_options,
    init_session_state,
    profile_to_user_history,
)

st.set_page_config(page_title="PCA Embedding Explorer", page_icon="📊", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

CLUSTER_HEX = [
    "#6c8fff", "#ff9f43", "#6dda7f", "#ff6b8a",
    "#b983ff", "#ffd32a", "#48dbfb", "#ff6348",
    "#34d399", "#fb923c", "#a78bfa", "#22d3ee",
    "#f87171", "#4ade80", "#facc15", "#818cf8",
]

ALGO_OPTIONS = {
    "K-Means (our NumPy implementation)": "kmeans",
    "Gaussian Mixture (tied covariance)": "gmm",
    "Hierarchical / Ward": "agglomerative",
}

st.title("📊 PCA Embedding Explorer")
st.markdown("""
This page uses the **same clustering pipeline as Restaurant Cluster GIS Map** and projects those clusters into 3D for analysis. The model works on **fully interpretable features**: cuisine type, price tier, Google rating, review volume, health inspection score, borough, and geographic location.

💡 **How to read this chart:** In **Principal Components** layout, each axis is a PCA component that combines the original 18 input features. In **Cleaner Cluster View**, the points are arranged from distances to cluster centroids so cluster separation is easier to see; it is a visualization of the same learned clusters, not a different clustering model.
Check the **Feature Loadings** and **Cluster Evidence** sections below to see exactly which features drive each cluster and which restaurants sit closest to each centroid.
""")


def format_price_tier_mean(value):
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    nearest_tier = int(np.clip(round(float(numeric_value)), 1, 4))
    return f"{float(numeric_value):.2f} / 4 (~{'$' * nearest_tier})"

if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading prepared restaurant data..."):
        _, _, runtime_df, _ = load_runtime_assets(DEFAULT_SEARCH_SAMPLE_SIZE)
    if runtime_df.empty:
        st.error("Prepared restaurant data could not be loaded.")
        st.stop()
    st.session_state["raw_df"] = runtime_df

raw_df       = st.session_state["raw_df"]
profile = get_active_profile()
st.session_state["user_history"] = profile_to_user_history(profile, raw_df)
user_history = st.session_state["user_history"]
algorithm = st.session_state.get("active_cluster_algorithm", "kmeans")
k = st.session_state.get("optimal_k", 10)
algo_display = {v: k_ for k_, v in ALGO_OPTIONS.items()}.get(algorithm, "K-Means (our NumPy implementation)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Visualization Controls")
    st.caption(
        f"Shared clustering setup: `{algo_display}` with `K = {k}`. "
        "Change these on the GIS Map page; this page mirrors the same clustering result."
    )
    color_by  = st.selectbox("Color by", ["Cluster", "Cuisine type", "Price tier", "Rating"])
    size_by   = st.selectbox("Size by", ["Review count", "User affinity score", "Uniform"])
    show_axes = st.toggle("Show axis labels", value=True)

    st.markdown("---")
    st.markdown("### Data Filters")
    st.caption("Isolate specific categories to see where they land in the latent space.")
    
    all_boros = get_valid_borough_options(raw_df)
    filter_boros = st.multiselect("Filter by Borough", all_boros, default=[])
    
    all_cuisines = get_valid_cuisine_options(raw_df)
    filter_cuisines = st.multiselect("Filter by Cuisine", all_cuisines, default=[])

# ── Run clustering ────────────────────────────────────────────────────────────
with st.spinner(f"Running {algo_display}..."):
    cdf, kmeans, scaler, pca = get_clustered_data(
        raw_df,
        user_history,
        k=k,
        force=False,
        algorithm=algorithm,
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

X_features, feature_columns, clustered_features_df = build_feature_matrix(cdf)
X_scaled_cluster, X_cluster_space, _ = prepare_clustering_space(X_features, scaler=scaler, fit=False)
distance_matrix = kmeans.transform(X_cluster_space)
cdf = cdf.copy()
cdf["distance_to_centroid"] = distance_matrix[np.arange(len(cdf)), cdf["cluster_id"].to_numpy(dtype=int)]

# ── Dot sizing ────────────────────────────────────────────────────────────────
pca_model = st.session_state.get("pca_model")
pca_axis_labels = get_pca_axis_labels(pca_model) if pca_model is not None else ["PC1", "PC2", "PC3"]
pca_component_summaries = getattr(pca_model, "component_summaries_", ["", "", ""]) if pca_model is not None else ["", "", ""]

with st.sidebar:
    projection_mode = st.selectbox(
        "Layout",
        ["Cleaner Cluster View", "Principal Components", "t-SNE (visualization only)"],
        index=1,
    )

if projection_mode == "Cleaner Cluster View":
    st.caption(
        "Layout note: this view uses each restaurant's distances to all cluster centroids, "
        "then projects that distance profile into 3D. It is cleaner for seeing cluster separation, "
        "but the axes are not raw PCA components."
    )
elif projection_mode == "Principal Components":
    st.caption(
        "Layout note: this is the direct 3D PCA projection of the scaled 18-feature restaurant space. "
        "The first three PCs are only a compressed view, so visual overlap does not mean the high-dimensional clusters are identical."
    )
else:
    st.caption(
        "Layout note: t-SNE is for visualization only. It preserves local neighborhoods but should not be used to explain the actual K-Means objective."
    )

projection_columns = (
    ["cluster_view_x", "cluster_view_y", "cluster_view_z"]
    if projection_mode == "Cleaner Cluster View"
    else ["pca_x", "pca_y", "pca_z"]
    if projection_mode == "Principal Components"
    else ["tsne_x", "tsne_y", "tsne_z"]
)
plot_df = cdf.dropna(subset=projection_columns).copy()
plot_df["plot_x"] = plot_df[projection_columns[0]]
plot_df["plot_y"] = plot_df[projection_columns[1]]
plot_df["plot_z"] = plot_df[projection_columns[2]]
plot_df["grade"] = plot_df.get("grade", pd.Series(["N/A"] * len(plot_df))).fillna("N/A")

if filter_boros:
    plot_df = plot_df[plot_df["boro"].isin(filter_boros)]
if filter_cuisines:
    plot_df = plot_df[plot_df["cuisine_type"].isin(filter_cuisines)]

if size_by == "Review count":
    raw_size = pd.to_numeric(plot_df["review_count"], errors="coerce").fillna(0)
elif size_by == "User affinity score":
    raw_size = pd.to_numeric(plot_df.get("user_affinity_score", 0), errors="coerce").fillna(0)
else:
    raw_size = pd.Series([1.0] * len(plot_df))

s_min, s_max = raw_size.min(), raw_size.max()
if s_max > s_min:
    plot_df["dot_size"] = 3 + ((raw_size - s_min) / (s_max - s_min) * 9)
else:
    plot_df["dot_size"] = 5.0

# ── Build Plotly figure ───────────────────────────────────────────────────────
fig = go.Figure()

if color_by == "Cluster":
    for cid in sorted(plot_df["cluster_id"].unique()):
        subset = plot_df[plot_df["cluster_id"] == cid]
        label  = subset["cluster_label"].iloc[0]

        fig.add_trace(go.Scatter3d(
            x=subset["plot_x"], y=subset["plot_y"], z=subset["plot_z"],
            mode="markers",
            name=label,
            marker=dict(
                size=subset["dot_size"],
                color=CLUSTER_HEX[cid % len(CLUSTER_HEX)],
                opacity=0.8,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "🍽️ %{customdata[1]} · 💰 %{customdata[2]}<br>"
                "⭐ %{customdata[3]} · 🏥 Grade %{customdata[4]}<br>"
                "<extra>%{fullData.name}</extra>"
            ),
            customdata=subset[["name", "cuisine_type", "price_tier", "avg_rating", "grade"]].values,
        ))
else:
    color_col = {
        "Cuisine type": "cuisine_type",
        "Price tier":   "price_tier",
        "Rating":       "avg_rating",
    }.get(color_by, "cluster_id")

    if plot_df[color_col].dtype == object or color_by == "Cuisine type":
        unique_vals = sorted(plot_df[color_col].dropna().unique())
        for i, val in enumerate(unique_vals):
            subset = plot_df[plot_df[color_col] == val]
            fig.add_trace(go.Scatter3d(
                x=subset["plot_x"], y=subset["plot_y"], z=subset["plot_z"],
                mode="markers",
                name=str(val),
                marker=dict(
                    size=subset["dot_size"],
                    color=CLUSTER_HEX[i % len(CLUSTER_HEX)],
                    opacity=0.8,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "🍽️ %{customdata[1]} · 💰 %{customdata[2]}<br>"
                    "⭐ %{customdata[3]} · 🏥 Grade %{customdata[4]}<br>"
                    "<extra></extra>"
                ),
                customdata=subset[["name", "cuisine_type", "price_tier", "avg_rating", "grade"]].values,
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=plot_df["plot_x"], y=plot_df["plot_y"], z=plot_df["plot_z"],
            mode="markers",
            marker=dict(
                size=plot_df["dot_size"],
                color=plot_df[color_col],
                colorscale="Viridis",
                colorbar=dict(title=color_by, x=-0.1),
                opacity=0.8,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "🍽️ %{customdata[1]} · 💰 %{customdata[2]}<br>"
                "⭐ %{customdata[3]} · 🏥 Grade %{customdata[4]}<br>"
                "<extra></extra>"
            ),
            customdata=plot_df[["name", "cuisine_type", "price_tier", "avg_rating", "grade"]].values,
        ))

if show_axes:
    if projection_mode == "Principal Components":
        _ax = pca_axis_labels if pca_axis_labels else []
        x_title = f"PC1: {_ax[0]}" if len(_ax) > 0 else "PC1"
        y_title = f"PC2: {_ax[1]}" if len(_ax) > 1 else "PC2"
        z_title = f"PC3: {_ax[2]}" if len(_ax) > 2 else "PC3"
    elif projection_mode == "t-SNE (visualization only)":
        x_title, y_title, z_title = ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
    else:
        x_title, y_title, z_title = ["Cluster Axis 1", "Cluster Axis 2", "Cluster Axis 3"]
else:
    x_title = y_title = z_title = ""

fig.update_layout(
    height=650,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(title=x_title, backgroundcolor="rgba(20,20,30,0.8)",
                   gridcolor="#2a2a38", color="#7a7a9a"),
        yaxis=dict(title=y_title, backgroundcolor="rgba(20,20,30,0.8)",
                   gridcolor="#2a2a38", color="#7a7a9a"),
        zaxis=dict(title=z_title, backgroundcolor="rgba(20,20,30,0.8)",
                   gridcolor="#2a2a38", color="#7a7a9a"),
        bgcolor="rgba(13,13,16,0.95)",
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="left",
        x=0,
        bgcolor="rgba(20,20,30,0.8)",
        bordercolor="#2a2a38",
        font=dict(color="#e0e0f0"),
    ),
    font=dict(color="#e0e0f0"),
    margin=dict(l=0, r=0, t=40, b=60),
)

chart_key = f"pca_chart_{len(plot_df)}_{k}_{color_by}_{projection_mode}"
try:
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key=chart_key)
    if event and event.selection and event.selection.points:
        idx  = event.selection.points[0].get("point_index", 0)
        rest = plot_df.iloc[idx]
        st.markdown("---")
        st.subheader(f"📋 {rest.get('name','')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cuisine", rest.get("cuisine_type", "—"))
        c2.metric("Rating", f"{rest.get('avg_rating', 0):.1f} ★")
        c3.metric("Cluster", rest.get("cluster_label", "—"))
        st.caption(f"📍 {rest.get('address', '')}")
except Exception:
    st.plotly_chart(fig, width="stretch", key=chart_key)

# ── Clustering quality metrics ────────────────────────────────────────────────
st.markdown("---")
try:
    _sil = float(silhouette_score(X_cluster_space, cdf["cluster_id"].to_numpy(dtype=int),
                                  sample_size=min(1500, len(cdf)), random_state=42))
except Exception:
    _sil = None
_inertia = getattr(kmeans, "inertia_", None)
cluster_sizes = cdf["cluster_id"].value_counts()
largest_cluster_share = (cluster_sizes.max() / len(cdf)) if len(cdf) else 0.0
_m1, _m2, _m3, _m4 = st.columns(4)
if _sil is not None:
    _m1.metric("Silhouette Score", f"{_sil:.3f}",
               help="Ranges −1 to 1. Values > 0.2 indicate reasonable separation; > 0.5 is strong.")
if _inertia is not None:
    _m2.metric("Inertia (WCSS)", f"{_inertia:,.0f}",
               help="Sum of squared distances from each point to its assigned centroid (lower = tighter clusters).")
_m3.metric("K (clusters)", k)
_m4.metric("Largest Cluster Share", f"{largest_cluster_share * 100:.1f}%",
           help="Checks whether one cluster is swallowing the dataset. Lower is usually easier to interpret.")

# ── PCA component interpretation + loadings (merged, vertical) ───────────────
_feature_cols = getattr(pca_model, "feature_columns_", None) if pca_model is not None else None

if pca_model is not None and _feature_cols and pca_model.components_ is not None:
    st.markdown("---")
    st.subheader("Principal Components: Interpretation & Feature Loadings")
    st.caption(
        "PCA finds orthogonal axes that capture the most variance in the 18-dimensional "
        "feature space. Each PC is a weighted combination of all input features. "
        "The bar chart shows the **top 8 features by absolute loading** — "
        "blue = positive (high feature value pushes the PC score up), "
        "red = negative (high feature value pushes the PC score down). "
        "These 3 PCs are used only for the 3D scatter above; the actual clustering "
        "runs on a higher-dimensional PCA subspace (≥ 92% variance retained)."
    )

    def _fmt(name: str) -> str:
        if name == "lat_norm":
            return "Latitude"
        if name == "lng_norm":
            return "Longitude"
        name = name.replace("_norm", "").replace("_", " ")
        if name.startswith("cuisine "):
            return name.replace("cuisine ", "") + " cuisine"
        if name.startswith("boro "):
            return name.replace("boro ", "") + " borough"
        return name.title()

    _n_pcs = min(3, len(pca_model.components_))
    for _pc_idx in range(_n_pcs):
        st.markdown("---")
        _loadings = pd.Series(pca_model.components_[_pc_idx], index=_feature_cols)
        _var_pct = pca_model.explained_variance_ratio_[_pc_idx] * 100
        _top8_idx = _loadings.abs().nlargest(8).index
        _top8 = _loadings[_top8_idx]

        _pos = _loadings[_loadings > 0.05].nlargest(3)
        _neg = _loadings[_loadings < -0.05].nsmallest(3)
        _pos_labels = [_fmt(n) for n in _pos.index]
        _neg_labels = [_fmt(n) for n in _neg.index]

        _col_text, _col_chart = st.columns([1, 1.4])

        with _col_text:
            st.markdown(f"#### PC{_pc_idx + 1} &nbsp; <span style='font-size:0.85rem;color:#7a7a9a'>({_var_pct:.1f}% of variance)</span>", unsafe_allow_html=True)

            if _pos_labels:
                st.markdown(
                    "**High score →** restaurants with strong "
                    + ", ".join(f"**{l}**" for l in _pos_labels[:3]) + " signals."
                )
            if _neg_labels:
                st.markdown(
                    "**Low score →** restaurants that lean toward "
                    + ", ".join(f"**{l}**" for l in _neg_labels[:3]) + "."
                )

            _summary = pca_component_summaries[_pc_idx] if _pc_idx < len(pca_component_summaries) else ""
            if _summary:
                st.caption(f"Overall axis direction: {_summary.lower()}")

            # Clarifying note for the first two PCs which both involve geographic features
            if _pc_idx == 0:
                st.info(
                    "PC1 separates restaurants primarily by **which borough** they are in "
                    "(discrete one-hot) and their **price tier**. "
                    "It is a prestige/price axis, not a raw coordinate axis.",
                    icon="ℹ️",
                )
            elif _pc_idx == 1:
                st.info(
                    "PC2 also involves geographic features, but captures **continuous "
                    "latitude/longitude coordinates** (east–west spread) rather than borough identity. "
                    "PC1 and PC2 are orthogonal — they explain different, non-overlapping "
                    "variance in the data.",
                    icon="ℹ️",
                )

        with _col_chart:
            _colors = ["#6c8fff" if v >= 0 else "#ff6b8a" for v in _top8.values]
            _short_names = [_fmt(n) for n in _top8_idx]
            _load_fig = go.Figure(go.Bar(
                x=_top8.values,
                y=_short_names,
                orientation="h",
                marker_color=_colors,
                text=[f"{v:+.3f}" for v in _top8.values],
                textposition="outside",
            ))
            _load_fig.update_layout(
                title=f"PC{_pc_idx + 1} — top 8 feature loadings",
                height=300,
                margin=dict(l=0, r=60, t=40, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0e0f0", size=11),
                xaxis=dict(
                    title="Loading magnitude",
                    gridcolor="#2a2a38",
                    zeroline=True, zerolinecolor="#555",
                ),
                yaxis=dict(gridcolor="#2a2a38", autorange="reversed"),
            )
            st.plotly_chart(_load_fig, width="stretch")

# ── Explained variance bar ────────────────────────────────────────────────────
if st.session_state["pca_model"] is not None:
    st.markdown("---")
    st.subheader("Variance Explained by Each Principal Component (Visualization)")
    pca_model = st.session_state["pca_model"]
    _vis_var_total = sum(pca_model.explained_variance_ratio_[:3]) * 100
    st.caption(
        f"These 3 components (used for the 3D scatter above) capture "
        f"**{_vis_var_total:.1f}%** of the variance in the scaled feature space. "
        f"The **clustering** itself runs in a higher-dimensional PCA subspace "
        f"(up to 24 components, retaining ≥ 92% variance) before K-Means assigns "
        f"labels — so the 3D view is an approximate projection, not the full "
        f"clustering space."
    )
    var_df = pd.DataFrame({
        "Component": [f"PC{idx + 1}" for idx in range(3)],
        "Variance Explained (%)": [round(v * 100, 2) for v in pca_model.explained_variance_ratio_],
    })
    bar_fig = px.bar(var_df, x="Component", y="Variance Explained (%)",
                     color_discrete_sequence=["#6c8fff"])
    bar_fig.update_layout(
        height=200,
        margin=dict(t=20, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0"),
        xaxis=dict(gridcolor="#2a2a38"),
        yaxis=dict(gridcolor="#2a2a38"),
    )
    st.plotly_chart(bar_fig, width="stretch")

# ── Cluster distance heatmap ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Distance Matrix")
st.caption(
    "Pairwise Euclidean distances between cluster centroids. "
    "Small values (dark blue) mean the clusters are close in feature space — "
    "they share similar cuisine, price, and quality profiles."
)

cluster_ids_sorted = sorted(cdf["cluster_id"].unique())
cluster_labels_sorted = [cdf[cdf["cluster_id"] == cid]["cluster_label"].iloc[0] for cid in cluster_ids_sorted]
centroids = kmeans.cluster_centers_
n_clusters = len(cluster_ids_sorted)

dist_matrix = np.zeros((n_clusters, n_clusters))
for i in range(n_clusters):
    for j in range(n_clusters):
        dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

heat_fig = go.Figure(go.Heatmap(
    z=dist_matrix,
    x=cluster_labels_sorted,
    y=cluster_labels_sorted,
    colorscale="Blues_r",
    text=np.round(dist_matrix, 2),
    texttemplate="%{text:.1f}",
    hovertemplate="%{x} ↔ %{y}<br>Distance: %{z:.2f}<extra></extra>",
))
heat_fig.update_layout(
    height=400,
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0f0", size=11),
    xaxis=dict(tickangle=30),
)
st.plotly_chart(heat_fig, width="stretch")

# ── Cluster evidence panel ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Evidence")
st.caption(
    "To make the analysis defensible, each cluster is paired with a textual explanation, summary statistics, and prototype restaurants nearest to the centroid."
)

focus_cluster_id = int(cdf["cluster_id"].value_counts().idxmax())
focus_df = cdf[cdf["cluster_id"] == focus_cluster_id].copy()
focus_row = focus_df.iloc[0]

st.info(f"Evidence for **{focus_row['cluster_label']}** (largest cluster): {focus_row.get('cluster_story', 'No explanation available.')}")
if focus_row.get("cluster_cuisine_mix"):
    st.caption(f"🍽️ Cuisine mix: {focus_row['cluster_cuisine_mix']}")
if focus_row.get("cluster_boro_mix"):
    st.caption(f"📍 Borough mix: {focus_row['cluster_boro_mix']}")
if focus_row.get("cluster_key_drivers"):
    st.caption(f"Key drivers: {focus_row['cluster_key_drivers']}")

overall_rating = pd.to_numeric(cdf["avg_rating"], errors="coerce").mean()
overall_price = pd.to_numeric(cdf["price_tier"], errors="coerce").mean()
overall_health = pd.to_numeric(cdf.get("score", pd.Series(np.nan, index=cdf.index)), errors="coerce").mean()

e1, e2, e3, e4 = st.columns(4)
e1.metric("Restaurants", f"{len(focus_df)}")
e2.metric("Avg Rating", f"{focus_df['avg_rating'].mean():.2f}", delta=f"{focus_df['avg_rating'].mean() - overall_rating:+.2f} vs all")
e3.metric(
    "Avg Price Tier",
    format_price_tier_mean(focus_df["price_tier"].mean()),
    delta=f"{focus_df['price_tier'].mean() - overall_price:+.2f} vs all",
    help="Google Places price tier: 1=$, 2=$$, 3=$$$, 4=$$$$. This is an average tier, not a dollar amount.",
)
focus_health = pd.to_numeric(focus_df.get("score", pd.Series(np.nan, index=focus_df.index)), errors="coerce").mean()
e4.metric("Avg Inspection Score", f"{focus_health:.1f}", delta=f"{focus_health - overall_health:+.1f} vs all")

prototype_df = focus_df.sort_values("distance_to_centroid").head(6).copy()
prototype_df["price_tier"] = prototype_df["price_tier"].apply(lambda x: "$" * int(x) if pd.notna(x) else "")
prototype_df["avg_rating"] = prototype_df["avg_rating"].map(lambda x: f"{x:.2f}")
prototype_df["distance_to_centroid"] = prototype_df["distance_to_centroid"].map(lambda x: f"{x:.3f}")
st.markdown("**Prototype restaurants nearest to the centroid**")
st.dataframe(
    prototype_df[["name", "cuisine_type", "boro", "price_tier", "avg_rating", "distance_to_centroid"]]
    .rename(columns={
        "name": "Restaurant",
        "cuisine_type": "Cuisine",
        "boro": "Borough",
        "price_tier": "Price",
        "avg_rating": "Rating",
        "distance_to_centroid": "Distance to centroid",
    }),
    width="stretch",
    hide_index=True,
)

# ── Cluster table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Breakdown")
summary = cdf.groupby(["cluster_id", "cluster_label"]).agg(
    Restaurants=("restaurant_id", "count"),
    Avg_Rating=("avg_rating", "mean"),
    Avg_Price=("price_tier", "mean"),
    Top_Cuisine=("cuisine_type", lambda x: x.value_counts().index[0]),
    Key_Drivers=("cluster_key_drivers", "first"),
).reset_index()
summary["Avg_Rating"] = summary["Avg_Rating"].round(2)
summary["Avg_Price_Tier"] = summary["Avg_Price"].map(format_price_tier_mean)
st.caption(
    "Average price is the mean Google Places price tier, where 1=$ and 4=$$$$. "
    "Most restaurants in this dataset are between tier 1 and 2, so many clusters are naturally budget-to-mid-range rather than upscale."
)
st.dataframe(summary[["cluster_label", "Restaurants", "Avg_Rating", "Avg_Price_Tier", "Top_Cuisine", "Key_Drivers"]],
             width="stretch", hide_index=True)
