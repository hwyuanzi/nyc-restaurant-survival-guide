import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.clustering import (
    build_feature_matrix, apply_user_weights, prepare_clustering_space,
    find_optimal_k, _assign_cluster_labels,
    _component_axis_label, _component_summary, _humanize_feature,
    TASTE_LABELS, VIBE_LABELS, FEATURE_LABELS,
)
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import init_session_state

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

init_session_state()

from utils.auth import require_auth
require_auth()

# ── Constants ─────────────────────────────────────────────────────────────────
CLUSTER_HEX = [
    "#6c8fff", "#ff9f43", "#6dda7f", "#ff6b8a",
    "#b983ff", "#ffd32a", "#48dbfb", "#ff6348",
    "#34d399", "#fb923c", "#a78bfa", "#22d3ee",
    "#f87171", "#4ade80", "#facc15", "#818cf8",
]
CLUSTER_RGBA = [
    [108,143,255,200],[255,159,67,200],[109,218,127,200],[255,107,138,200],
    [185,131,255,200],[255,211,42,200],[72,219,251,200],[255,99,72,200],
    [52,211,153,200],[251,146,60,200],[167,139,250,200],[34,211,238,200],
    [248,113,113,200],[74,222,128,200],[250,204,21,200],[129,140,248,200],
]

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🍽 Taste & Cuisine Cluster Explorer")
st.markdown(
    "Restaurants grouped **purely by what they are** — cuisine, price, quality, "
    "vibe, and taste — with **no geographic influence**. Two restaurants on "
    "opposite sides of the city land in the same cluster if they feel and taste "
    "similar."
)

# ── Load data ─────────────────────────────────────────────────────────────────
if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    with st.spinner("Loading prepared restaurant data..."):
        _, _, runtime_df, _ = load_runtime_assets(DEFAULT_SEARCH_SAMPLE_SIZE)
    if runtime_df.empty:
        st.error("Prepared restaurant data could not be loaded.")
        st.stop()
    st.session_state["raw_df"] = runtime_df

raw_df = st.session_state["raw_df"]
user_history = st.session_state["user_history"]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Clustering Controls")
    k = st.slider("Number of Clusters (K)", 4, 16, 8, key="taste_k")
    color_by = st.selectbox("Color by", ["Cluster", "Cuisine type", "Price tier", "Rating"], key="taste_color")
    size_by = st.selectbox("Size by", ["Review count", "Uniform"], key="taste_size")

    if st.button("🔄 Re-cluster", key="taste_rerun"):
        st.session_state.pop("taste_cdf", None)

    st.markdown("---")
    st.markdown("### Data Filters")
    all_cuisines = sorted([str(c) for c in raw_df["cuisine_type"].dropna().unique() if str(c).strip()])
    filter_cuisines = st.multiselect("Filter by Cuisine", all_cuisines, default=[], key="taste_filter_cuisine")

# ── Run location-agnostic clustering ──────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _run_taste_clustering(_df_hash, k, raw_df_arg):
    """Build features without location and run KMeans."""
    X, feature_columns, df = build_feature_matrix(raw_df_arg, use_location=False)
    X_aug = apply_user_weights(X, df, user_history)
    proj_cols = feature_columns + ["user_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    # PCA reduction for clustering
    max_comp = min(20, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)
    if max_comp >= 3:
        probe = PCA(n_components=max_comp, random_state=42)
        probe.fit(X_scaled)
        target = int(np.searchsorted(np.cumsum(probe.explained_variance_ratio_), 0.92) + 1)
        n_comp = int(np.clip(target, 3, max_comp))
        reducer = PCA(n_components=n_comp, random_state=42)
        X_cluster = reducer.fit_transform(X_scaled)
    else:
        X_cluster = X_scaled

    max_k = max(2, min(16, len(df) - 1))
    k = max(2, min(k, max_k))

    KMC = MiniBatchKMeans if len(df) > 10000 else KMeans
    best_model, best_labels, best_score = None, None, -1
    for seed in [42, 52, 62]:
        m = KMC(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=seed)
        labels = m.fit_predict(X_cluster)
        if len(set(labels)) < 2:
            continue
        try:
            sc = silhouette_score(X_cluster, labels, sample_size=min(1500, len(X_cluster)), random_state=seed)
        except Exception:
            sc = -1
        if sc > best_score:
            best_score, best_model, best_labels = sc, m, labels

    if best_model is None:
        best_model = KMC(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42).fit(X_cluster)
        best_labels = best_model.labels_

    df = df.copy()
    df["cluster_id"] = best_labels

    # Merge tiny clusters
    centroids = best_model.cluster_centers_
    min_size = max(1, int(round(len(df) * 0.03)))
    while True:
        sizes = df["cluster_id"].value_counts()
        smalls = sizes[sizes < min_size].index.tolist()
        bigs = sizes[sizes >= min_size].index.tolist()
        if not smalls or not bigs:
            break
        smallest = sizes.idxmin()
        dists = np.linalg.norm(centroids - centroids[smallest], axis=1)
        dists[smallest] = np.inf
        for c in range(len(centroids)):
            if c not in bigs:
                dists[c] = np.inf
        df.loc[df["cluster_id"] == smallest, "cluster_id"] = int(np.argmin(dists))

    # PCA 3D for visualization
    pca3 = PCA(n_components=3, random_state=42)
    coords = pca3.fit_transform(X_scaled)
    df["pca_x"], df["pca_y"], df["pca_z"] = coords[:, 0], coords[:, 1], coords[:, 2]
    pca3.axis_labels_ = [_component_axis_label(c, proj_cols) for c in pca3.components_[:3]]
    pca3.component_summaries_ = [_component_summary(c, proj_cols) for c in pca3.components_[:3]]

    # Cluster-view coords
    cdists = best_model.transform(X_cluster)
    cv_pca = PCA(n_components=3, random_state=42)
    cv = cv_pca.fit_transform(StandardScaler().fit_transform(cdists))
    df["cv_x"], df["cv_y"], df["cv_z"] = cv[:, 0], cv[:, 1], cv[:, 2]

    df["cluster_label"] = _assign_cluster_labels(df)
    df["user_affinity_score"] = X_aug[:, -1]

    return df, best_model, scaler, pca3, proj_cols, best_score


df_hash = pd.util.hash_pandas_object(raw_df[["restaurant_id"]]).sum()
with st.spinner("Clustering by taste & cuisine (no geography)..."):
    cdf, model, scaler, pca3, proj_cols, sil_score = _run_taste_clustering(df_hash, k, raw_df)

# ── Filter ────────────────────────────────────────────────────────────────────
plot_df = cdf.copy()
if filter_cuisines:
    plot_df = plot_df[plot_df["cuisine_type"].isin(filter_cuisines)]

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric("Clusters", int(cdf["cluster_id"].nunique()))
m2.metric("Silhouette", f"{sil_score:.3f}" if sil_score > -1 else "—")
m3.metric("Features", "Cuisine · Price · Quality · Vibe · Taste")

# ── 3D Scatter ────────────────────────────────────────────────────────────────
st.markdown("---")

view_mode = st.radio("Layout", ["Cluster View", "PCA View"], horizontal=True, key="taste_view")
xcol = "cv_x" if view_mode == "Cluster View" else "pca_x"
ycol = "cv_y" if view_mode == "Cluster View" else "pca_y"
zcol = "cv_z" if view_mode == "Cluster View" else "pca_z"

if size_by == "Review count":
    raw_s = pd.to_numeric(plot_df["review_count"], errors="coerce").fillna(0)
    smin, smax = raw_s.min(), raw_s.max()
    plot_df["dot_size"] = 3 + ((raw_s - smin) / (smax - smin) * 9) if smax > smin else 5.0
else:
    plot_df["dot_size"] = 5.0

fig = go.Figure()

if color_by == "Cluster":
    for cid in sorted(plot_df["cluster_id"].unique()):
        sub = plot_df[plot_df["cluster_id"] == cid]
        label = sub["cluster_label"].iloc[0]
        fig.add_trace(go.Scatter3d(
            x=sub[xcol], y=sub[ycol], z=sub[zcol],
            mode="markers", name=label,
            marker=dict(size=sub["dot_size"], color=CLUSTER_HEX[cid % len(CLUSTER_HEX)], opacity=0.85, line=dict(width=0)),
            hovertemplate="<b>%{customdata[0]}</b><br>🍽 %{customdata[1]} · 💰 %{customdata[2]}<br>⭐ %{customdata[3]}<extra>%{fullData.name}</extra>",
            customdata=sub[["name","cuisine_type","price_tier","avg_rating"]].values,
        ))
elif color_by == "Cuisine type":
    for i, val in enumerate(sorted(plot_df["cuisine_type"].dropna().unique())):
        sub = plot_df[plot_df["cuisine_type"] == val]
        fig.add_trace(go.Scatter3d(
            x=sub[xcol], y=sub[ycol], z=sub[zcol],
            mode="markers", name=str(val),
            marker=dict(size=sub["dot_size"], color=CLUSTER_HEX[i % len(CLUSTER_HEX)], opacity=0.8, line=dict(width=0)),
            hovertemplate="<b>%{customdata[0]}</b><br>⭐ %{customdata[1]}<extra></extra>",
            customdata=sub[["name","avg_rating"]].values,
        ))
else:
    col_map = {"Price tier": "price_tier", "Rating": "avg_rating"}
    cc = col_map.get(color_by, "avg_rating")
    fig.add_trace(go.Scatter3d(
        x=plot_df[xcol], y=plot_df[ycol], z=plot_df[zcol],
        mode="markers",
        marker=dict(size=plot_df["dot_size"], color=pd.to_numeric(plot_df[cc], errors="coerce"), colorscale="Viridis", colorbar=dict(title=color_by), opacity=0.8, line=dict(width=0)),
        hovertemplate="<b>%{customdata[0]}</b><br>🍽 %{customdata[1]}<extra></extra>",
        customdata=plot_df[["name","cuisine_type"]].values,
    ))

ax_labels = getattr(pca3, "axis_labels_", ["PC1","PC2","PC3"])
if view_mode == "PCA View":
    xt = f"PC1: {ax_labels[0]}" if ax_labels else "PC1"
    yt = f"PC2: {ax_labels[1]}" if len(ax_labels) > 1 else "PC2"
    zt = f"PC3: {ax_labels[2]}" if len(ax_labels) > 2 else "PC3"
else:
    xt, yt, zt = "Cluster Axis 1", "Cluster Axis 2", "Cluster Axis 3"

fig.update_layout(
    height=620,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    scene=dict(
        xaxis=dict(title=xt, backgroundcolor="rgba(20,20,30,0.8)", gridcolor="#2a2a38", color="#7a7a9a"),
        yaxis=dict(title=yt, backgroundcolor="rgba(20,20,30,0.8)", gridcolor="#2a2a38", color="#7a7a9a"),
        zaxis=dict(title=zt, backgroundcolor="rgba(20,20,30,0.8)", gridcolor="#2a2a38", color="#7a7a9a"),
        bgcolor="rgba(13,13,16,0.95)",
    ),
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0,
                bgcolor="rgba(20,20,30,0.8)", bordercolor="#2a2a38", font=dict(color="#e0e0f0")),
    font=dict(color="#e0e0f0"),
    margin=dict(l=0, r=0, t=40, b=60),
)

chart_key = f"taste3d_{len(plot_df)}_{k}_{color_by}_{view_mode}"
try:
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
    if ev and ev.selection and ev.selection.points:
        r = plot_df.iloc[ev.selection.points[0].get("point_index", 0)]
        st.markdown("---")
        st.subheader(f"📋 {r.get('name','')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Cuisine", r.get("cuisine_type", "—"))
        c2.metric("Rating", f"{r.get('avg_rating', 0):.1f} ★")
        c3.metric("Cluster", r.get("cluster_label", "—"))
except Exception:
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

# ── Cluster summary cards ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Summaries")

unique_clusters = sorted(cdf["cluster_id"].unique().tolist())
cols = st.columns(min(len(unique_clusters), 4))

MIN_RATE = 0.05
ENRICHMENT = 1.5

for i, cid in enumerate(unique_clusters):
    col = cols[i % 4]
    subset = cdf[cdf["cluster_id"] == cid]
    label = subset["cluster_label"].iloc[0]
    hex_c = CLUSTER_HEX[cid % len(CLUSTER_HEX)]
    top3 = subset["cuisine_type"].value_counts().head(3).index.tolist()
    avg_r = subset["avg_rating"].mean()

    vibe_tags, taste_tags = [], []
    for vc in [c for c in subset.columns if c.startswith("vibe_")]:
        cm, gm = subset[vc].mean(), cdf[vc].mean() if vc in cdf.columns else 0.0
        if cm >= MIN_RATE and (gm == 0 or cm >= ENRICHMENT * gm):
            vibe_tags.append(vc.replace("vibe_", "").title())
    for tc in [c for c in subset.columns if c.startswith("taste_")]:
        cm, gm = subset[tc].mean(), cdf[tc].mean() if tc in cdf.columns else 0.0
        if cm >= MIN_RATE and (gm == 0 or cm >= ENRICHMENT * gm):
            taste_tags.append(tc.replace("taste_", "").title())

    pills = ""
    for t in vibe_tags:
        pills += f'<span style="display:inline-block;background:#3d2963;color:#c9a0ff;padding:1px 6px;border-radius:10px;font-size:.68rem;margin:1px 2px">🎭 {t}</span>'
    for t in taste_tags:
        pills += f'<span style="display:inline-block;background:#2a4035;color:#7ae8a0;padding:1px 6px;border-radius:10px;font-size:.68rem;margin:1px 2px">🍽 {t}</span>'

    with col:
        st.markdown(f"""
        <div style="border-left:4px solid {hex_c};padding:.6rem .8rem;background:#1a1a2e;border-radius:8px;margin-bottom:.5rem">
          <div style="font-weight:700;color:#e0e0f0">{label}</div>
          <div style="font-size:.78rem;color:#7a7a9a">{len(subset)} restaurants</div>
          <div style="font-size:.78rem;color:#a0a0c0">{", ".join(top3)}</div>
          {f'<div style="margin-top:3px">{pills}</div>' if pills else ''}
          <div style="font-size:.78rem;color:#d19900">⭐ {avg_r:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# ── PCA component interpretation ──────────────────────────────────────────────
st.markdown("---")
st.subheader("What The Principal Components Mean (No Location)")
st.caption("Since geographic features are excluded, the PCs capture cuisine, price, quality, and vibe/taste patterns only.")
pc_cols = st.columns(3)
ax_labels = getattr(pca3, "axis_labels_", [])
summaries = getattr(pca3, "component_summaries_", [])
for idx, pcol in enumerate(pc_cols):
    with pcol:
        st.metric(f"PC{idx+1}", ax_labels[idx] if idx < len(ax_labels) else f"PC{idx+1}")
        if idx < len(summaries) and summaries[idx]:
            st.caption(summaries[idx])

# ── Feature loadings ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Top Feature Loadings per Component")
load_cols = st.columns(min(3, len(pca3.components_)))
for pc_idx, lc in enumerate(load_cols):
    loadings = pd.Series(pca3.components_[pc_idx], index=proj_cols)
    top_idx = loadings.abs().nlargest(8).index
    top_vals = loadings[top_idx]
    colors = ["#6c8fff" if v >= 0 else "#ff6b8a" for v in top_vals.values]
    names = [n.replace("cuisine_","").replace("_norm","").replace("taste_","🍴 ").replace("vibe_","🎭 ").replace("_"," ").title() for n in top_idx]
    lf = go.Figure(go.Bar(x=top_vals.values, y=names, orientation="h", marker_color=colors))
    lf.update_layout(
        title=f"PC{pc_idx+1}", height=260, margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0f0", size=11),
        xaxis=dict(gridcolor="#2a2a38", zeroline=True, zerolinecolor="#4a4a58"),
        yaxis=dict(gridcolor="#2a2a38"),
    )
    with lc:
        st.plotly_chart(lf, use_container_width=True)

# ── Variance explained ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Variance Explained by Each Component")
var_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(3)],
    "Variance Explained (%)": [round(v * 100, 2) for v in pca3.explained_variance_ratio_],
})
vf = px.bar(var_df, x="Component", y="Variance Explained (%)", color_discrete_sequence=["#6c8fff"])
vf.update_layout(
    height=200, margin=dict(t=20,b=0,l=0,r=0),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e0e0f0"),
    xaxis=dict(gridcolor="#2a2a38"), yaxis=dict(gridcolor="#2a2a38"),
)
st.plotly_chart(vf, use_container_width=True)
