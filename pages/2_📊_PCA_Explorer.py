import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.clustering import get_clustered_data
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import init_session_state, predict_user_cluster

init_session_state()
CLUSTER_HEX = [
    "#6c8fff", "#ff9f43", "#6dda7f", "#ff6b8a",
    "#b983ff", "#ffd32a", "#48dbfb", "#ff6348",
    "#34d399", "#fb923c", "#a78bfa", "#22d3ee",
    "#f87171", "#4ade80", "#facc15", "#818cf8",
]

st.title("📊 PCA Embedding Explorer")
st.markdown("Restaurants plotted in 3D taste space. The default layout emphasizes cleaner cluster separation, while the PCA summary below explains the main latent dimensions.")

with st.expander("How clustering works"):
    st.markdown(
        """
        We cluster restaurants in a reduced latent space instead of clustering directly on raw columns.

        1. **TF-IDF text features**
        We turn restaurant text like summaries, tags, and descriptions into a sparse feature matrix so the model can capture cuisine, vibe, and menu language.

        2. **TruncatedSVD**
        We reduce those sparse text features into dense semantic factors. This keeps the strongest restaurant-style patterns while removing a lot of noise.

        3. **Structured restaurant signals**
        We add practical numeric features like price tier, rating, review volume, health score, and user affinity so the clusters reflect both semantics and quality.

        4. **Standard scaling**
        We normalize the merged features so no one numeric input overwhelms the others.

        5. **PCA for clustering space**
        We apply PCA to the combined dense matrix and cluster in that reduced space. This gives K-Means a cleaner, denoised representation than the full raw matrix.

        6. **K-Means with multiple seeds**
        We try several K-Means initializations and keep the solution with the best silhouette separation, which makes clusters more stable and distinct.

        7. **PCA 3D for visualization**
        The 3D chart is a display projection of the learned cluster space, so the plot is for interpretation, while the clustering itself uses the fuller reduced representation.
        """
    )

with st.expander("How optimal K works"):
    st.markdown(
        """
        The **Find Optimal K** button tests several cluster counts and recommends the one with the best **silhouette score**.

        1. We build the same reduced clustering space used by the real clustering pipeline.
        2. We test K values from 4 to 15.
        3. For each K, we run K-Means and compute the silhouette score.
        4. The silhouette score rewards clusters that are both tight internally and well separated from each other.
        5. We pick the K with the highest score.

        It is a strong heuristic, not a hard truth. Sometimes the most mathematically separated K is not the most intuitive visually, so it is best used as a recommendation.
        """
    )

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

    if st.button("🔍 Find Optimal K"):
        with st.spinner("Computing silhouette scores..."):
            from utils.clustering import build_feature_matrix, apply_user_weights, prepare_clustering_space, find_optimal_k
            X, _, _ = build_feature_matrix(raw_df)
            X_aug = apply_user_weights(X, raw_df, user_history)
            _, X_cluster, _ = prepare_clustering_space(X_aug, fit=True)
            best_k = find_optimal_k(X_cluster)
            st.session_state["optimal_k"] = best_k
            st.success(f"Optimal K = {best_k}")
            k = best_k

    if st.button("🔄 Re-run Clustering"):
        st.session_state["clustered_df"] = None

    st.markdown("---")
    st.markdown("### Visualization Controls")
    color_by       = st.selectbox("Color by", ["Cluster", "Cuisine type", "Price tier", "Rating"])
    size_by        = st.selectbox("Size by", ["Review count", "User affinity score", "Uniform"])
    highlight_mode = st.toggle("Highlight my cluster only", value=False)
    show_user      = st.toggle("Show my position", value=True)
    show_axes      = st.toggle("Show axis labels", value=True)

# ── Run clustering ────────────────────────────────────────────────────────────
with st.spinner("Running K-Means clustering..."):
    cdf, kmeans, scaler, pca = get_clustered_data(raw_df, user_history, k=k,
                                                   force=(st.session_state["clustered_df"] is None))
    st.session_state["clustered_df"] = cdf
    st.session_state["kmeans_model"] = kmeans
    st.session_state["scaler"]       = scaler
    st.session_state["pca_model"]    = pca

predicted_cluster = predict_user_cluster(user_history, cdf, kmeans, scaler)
st.session_state["predicted_cluster"] = predicted_cluster

if predicted_cluster != -1:
    cl_label = cdf[cdf["cluster_id"] == predicted_cluster]["cluster_label"].iloc[0]
    st.success(f"🎯 Your predicted cluster: **{cl_label}**")

# ── Dot sizing ────────────────────────────────────────────────────────────────
pca_model = st.session_state.get("pca_model")
pca_axis_labels = getattr(pca_model, "axis_labels_", ["PC1", "PC2", "PC3"]) if pca_model is not None else ["PC1", "PC2", "PC3"]
pca_component_summaries = getattr(pca_model, "component_summaries_", ["", "", ""]) if pca_model is not None else ["", "", ""]

with st.sidebar:
    projection_mode = st.selectbox(
        "Layout",
        ["Cleaner Cluster View", "Principal Components"],
        index=0,
    )

projection_columns = ["cluster_view_x", "cluster_view_y", "cluster_view_z"] if projection_mode == "Cleaner Cluster View" else ["pca_x", "pca_y", "pca_z"]
plot_df = cdf.dropna(subset=projection_columns).copy()
plot_df["plot_x"] = plot_df[projection_columns[0]]
plot_df["plot_y"] = plot_df[projection_columns[1]]
plot_df["plot_z"] = plot_df[projection_columns[2]]

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
                "%{customdata[1]} · %{customdata[2]}<br>"
                "⭐ %{customdata[3]}<extra>%{fullData.name}</extra>"
            ),
            customdata=subset[["name", "cuisine_type", "price_tier", "avg_rating"]].values,
        ))
else:
    color_col = {
        "Cuisine type": "cuisine_type",
        "Price tier":   "price_tier",
        "Rating":       "avg_rating",
    }.get(color_by, "cluster_id")

    fig.add_trace(go.Scatter3d(
        x=plot_df["plot_x"], y=plot_df["plot_y"], z=plot_df["plot_z"],
        mode="markers",
        marker=dict(
            size=plot_df["dot_size"],
            color=pd.Categorical(plot_df[color_col]).codes if plot_df[color_col].dtype == object else plot_df[color_col],
            colorscale="Viridis",
            opacity=0.8,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} · %{customdata[2]}<br>"
            "⭐ %{customdata[3]}<extra></extra>"
        ),
        customdata=plot_df[["name", "cuisine_type", "price_tier", "avg_rating"]].values,
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
        x_title, y_title, z_title = ["PC1", "PC2", "PC3"]
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

try:
    event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="pca_chart")
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
    st.plotly_chart(fig, use_container_width=True)

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

# ── Explained variance bar ────────────────────────────────────────────────────
if st.session_state["pca_model"] is not None:
    st.markdown("---")
    st.subheader("Variance Explained by Each Principal Component")
    pca_model = st.session_state["pca_model"]
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

# ── Cluster table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Breakdown")
summary = cdf.groupby(["cluster_id", "cluster_label"]).agg(
    Restaurants=("restaurant_id", "count"),
    Avg_Rating=("avg_rating", "mean"),
    Top_Cuisine=("cuisine_type", lambda x: x.value_counts().index[0]),
).reset_index()
summary["Avg_Rating"] = summary["Avg_Rating"].round(2)
st.dataframe(summary[["cluster_label", "Restaurants", "Avg_Rating", "Top_Cuisine"]],
             use_container_width=True, hide_index=True)
