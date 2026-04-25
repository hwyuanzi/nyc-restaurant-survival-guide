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
    compute_silhouette,
    get_clustered_data,
    prepare_clustering_space,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import (
    get_active_profile,
    get_valid_borough_options,
    get_valid_cuisine_options,
    init_session_state,
    predict_user_cluster,
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

💡 **How to read this chart:** Each axis is a Principal Component that combines the original features.
Check the **Feature Loadings** and **Cluster Evidence** sections below to see exactly which features drive each cluster and which restaurants sit closest to each centroid.
""")

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
    color_by       = st.selectbox("Color by", ["Cluster", "Cuisine type", "Price tier", "Rating"])
    size_by        = st.selectbox("Size by", ["Review count", "User affinity score", "Uniform"])
    highlight_mode = st.toggle("Highlight my cluster only", value=False)
    show_user      = st.toggle("Show my position", value=True)
    show_axes      = st.toggle("Show axis labels", value=True)

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
        force=(st.session_state["clustered_df"] is None),
        algorithm=algorithm,
    )
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

predicted_cluster = predict_user_cluster(user_history, cdf, kmeans, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

if predicted_cluster != -1:
    cl_label = cdf[cdf["cluster_id"] == predicted_cluster]["cluster_label"].iloc[0]
    st.success(f"🎯 Your predicted cluster: **{cl_label}**")

X_features, feature_columns, clustered_features_df = build_feature_matrix(cdf)
X_scaled_cluster, X_cluster_space, _ = prepare_clustering_space(X_features, scaler=scaler, fit=False)
distance_matrix = kmeans.transform(X_cluster_space)
cdf = cdf.copy()
cdf["distance_to_centroid"] = distance_matrix[np.arange(len(cdf)), cdf["cluster_id"].to_numpy(dtype=int)]

# ── Dot sizing ────────────────────────────────────────────────────────────────
pca_model = st.session_state.get("pca_model")
pca_axis_labels = getattr(pca_model, "axis_labels_", ["PC1", "PC2", "PC3"]) if pca_model is not None else ["PC1", "PC2", "PC3"]
pca_component_summaries = getattr(pca_model, "component_summaries_", ["", "", ""]) if pca_model is not None else ["", "", ""]

with st.sidebar:
    projection_mode = st.selectbox(
        "Layout",
        ["Cleaner Cluster View", "Principal Components", "t-SNE (visualization only)"],
        index=0,
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
        is_user_cluster = (cid == predicted_cluster)
        opacity = 1.0 if (not highlight_mode or is_user_cluster) else 0.1

        fig.add_trace(go.Scatter3d(
            x=subset["plot_x"], y=subset["plot_y"], z=subset["plot_z"],
            mode="markers",
            name=label,
            marker=dict(
                size=subset["dot_size"],
                color=CLUSTER_HEX[cid % len(CLUSTER_HEX)],
                opacity=opacity,
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

# User position marker
if show_user and predicted_cluster != -1:
    user_pos = plot_df[plot_df["cluster_id"] == predicted_cluster][["plot_x", "plot_y", "plot_z"]].mean()
    fig.add_trace(go.Scatter3d(
        x=[user_pos["plot_x"]], y=[user_pos["plot_y"]], z=[user_pos["plot_z"]],
        mode="markers+text",
        name="You",
        text=["📍 You"],
        textposition="top center",
        marker=dict(size=14, color="white", symbol="diamond", line=dict(color="#6c8fff", width=3)),
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
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
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
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

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

with st.expander("🔬 Compare clustering algorithms on this dataset"):
    st.caption(
        "This is the same comparison logic as the GIS map. Use it to argue why your chosen algorithm is a reasonable modeling choice."
    )
    run_compare = st.button("Run comparison for this page", key="pca_compare_btn")
    if run_compare:
        comparison_rows = []
        progress = st.progress(0.0, text="Starting…")
        algos_to_run = [("kmeans", "K-Means"), ("gmm", "GMM (tied)"), ("agglomerative", "Hierarchical (Ward)")]
        for i, (algo_key, algo_name) in enumerate(algos_to_run):
            progress.progress(i / len(algos_to_run), text=f"Running {algo_name}…")
            try:
                summary = compute_silhouette(raw_df, user_history, algo_key, k=k)
                comparison_rows.append({
                    "Algorithm": algo_name,
                    "Silhouette": round(summary["silhouette"], 4),
                    "Clusters": summary["n_clusters"],
                    "Top labels": " · ".join(summary["top_labels"]),
                })
            except Exception as exc:
                comparison_rows.append({
                    "Algorithm": algo_name,
                    "Silhouette": float("nan"),
                    "Clusters": 0,
                    "Top labels": f"⚠️ error: {exc}",
                })
        progress.progress(1.0, text="Done.")
        st.session_state["pca_cluster_comparison"] = comparison_rows
    if st.session_state.get("pca_cluster_comparison"):
        cmp_df = pd.DataFrame(st.session_state["pca_cluster_comparison"])
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ── PCA component interpretation ─────────────────────────────────────────────
if pca_model is not None:
    st.markdown("---")
    st.subheader("What The Principal Components Mean")
    component_cols = st.columns(3)
    for idx, col in enumerate(component_cols):
        with col:
            st.metric(f"PC{idx + 1}", pca_axis_labels[idx] if idx < len(pca_axis_labels) else f"PC{idx + 1}")
            if idx < len(pca_component_summaries) and pca_component_summaries[idx]:
                st.caption(pca_component_summaries[idx])

# ── Feature loadings per component ───────────────────────────────────────────
_feature_cols = getattr(pca_model, "feature_columns_", None) if pca_model is not None else None
if _feature_cols and pca_model is not None and pca_model.components_ is not None:
    st.markdown("---")
    st.subheader("Top Feature Loadings per Component")
    st.caption(
        "Each bar shows how strongly a raw feature drives the component. "
        "Blue = positive loading (high feature value → high PC score); "
        "red = negative loading."
    )
    _n_pcs = min(3, len(pca_model.components_))
    _load_cols = st.columns(_n_pcs)
    for _pc_idx, _load_col in enumerate(_load_cols):
        _loadings = pd.Series(pca_model.components_[_pc_idx], index=_feature_cols)
        _top_idx = _loadings.abs().nlargest(8).index
        _top_loadings = _loadings[_top_idx]
        _pc_label = pca_axis_labels[_pc_idx] if _pc_idx < len(pca_axis_labels) else f"PC{_pc_idx+1}"
        _colors = ["#6c8fff" if v >= 0 else "#ff6b8a" for v in _top_loadings.values]
        _short_names = [
            n.replace("cuisine_", "").replace("family_", "").replace("_norm", "").replace("_", " ").title()
            for n in _top_idx
        ]
        _load_fig = go.Figure(go.Bar(
            x=_top_loadings.values,
            y=_short_names,
            orientation="h",
            marker_color=_colors,
        ))
        _load_fig.update_layout(
            title=f"PC{_pc_idx+1}: {_pc_label}",
            height=280,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0f0", size=11),
            xaxis=dict(gridcolor="#2a2a38", zeroline=True, zerolinecolor="#4a4a58"),
            yaxis=dict(gridcolor="#2a2a38"),
        )
        with _load_col:
            st.plotly_chart(_load_fig, use_container_width=True)

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
    st.plotly_chart(bar_fig, use_container_width=True)

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
st.plotly_chart(heat_fig, use_container_width=True)

# ── Cluster evidence panel ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Evidence")
st.caption(
    "To make the analysis defensible, each cluster is paired with a textual explanation, summary statistics, and prototype restaurants nearest to the centroid."
)

cluster_options = (
    cdf[["cluster_id", "cluster_label"]]
    .drop_duplicates()
    .sort_values(["cluster_label", "cluster_id"])
)
default_cluster_id = predicted_cluster if predicted_cluster != -1 else int(cdf["cluster_id"].value_counts().idxmax())
focus_cluster_id = default_cluster_id
focus_df = cdf[cdf["cluster_id"] == focus_cluster_id].copy()
focus_row = focus_df.iloc[0]

label_prefix = "your current cluster" if predicted_cluster != -1 else "the largest cluster in the current analysis"
st.info(f"Evidence for **{focus_row['cluster_label']}** ({label_prefix}): {focus_row.get('cluster_story', 'No explanation available.')}")
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
e3.metric("Avg Price Tier", f"{focus_df['price_tier'].mean():.2f}", delta=f"{focus_df['price_tier'].mean() - overall_price:+.2f} vs all")
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
    use_container_width=True,
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
summary["Avg_Price"] = summary["Avg_Price"].round(2)
st.dataframe(summary[["cluster_label", "Restaurants", "Avg_Rating", "Avg_Price", "Top_Cuisine", "Key_Drivers"]],
             use_container_width=True, hide_index=True)
