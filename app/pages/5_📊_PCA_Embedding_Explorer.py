"""
PCA Embedding Explorer — Multi-model PCA, geographic filtering, and k-means density clustering.
Applies Principal Component Analysis to Transformer (384-D) and Autoencoder (32-D) embeddings,
then runs k-means to identify the densest clusters and extract their shared characteristics.
Course topics: Week 4 (Embeddings), Week 6 (PCA, Autoencoders), Week 7 (k-means Clustering).
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="PCA Embedding Explorer", page_icon="📊", layout="wide")
from app.ui_utils import apply_apple_theme
apply_apple_theme()

st.title("📊 PCA Embedding Explorer")

st.markdown("""
### 🎯 What is this?
**Principal Component Analysis (PCA)** applied to high-dimensional embeddings from our neural networks,
combined with **k-means clustering** to automatically discover the densest restaurant groups and their
shared characteristics. Instead of just viewing geographic locations on a map, we project each restaurant's
*learned representation* into 2D and ask: do restaurants cluster by borough? By cuisine? By grade?

### 🧠 Why PCA + k-means?
Each restaurant is a **high-dimensional vector** (384-D for Transformers, 32-D for the Autoencoder).
PCA finds the axes of maximum variance, compressing these vectors into 2 interpretable principal components.
**k-means clustering** then groups the PCA-projected restaurants into *k* clusters, letting us identify
the most tightly packed groups and extract what makes them similar.

### 💡 How to explore?
1. **Select Embedding Source:** Choose between the Transformer (384-D) or Autoencoder (32-D) embeddings.
2. **Geographic Filtering:** Filter by Borough (e.g., Manhattan only) to compare restaurant profiles across neighborhoods.
3. **Run k-means:** Adjust *k* and examine the densest cluster's shared characteristics (cuisine, grade, borough breakdown).
4. **Inspect Scree Plots:** Higher explained variance = PCA captured more meaningful structure.
---
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1. SHARED SYNTHETIC CORPUS (same seed as other pages)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def build_pca_corpus(n=220, seed=42):
    """Generates the same diverse NYC restaurant corpus used across the app."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    templates = {
        "Italian": [
            "{adj} Italian trattoria in {boro}. Known for hand-made {pasta}, {atm}. {extra}",
            "Classic red-sauce Italian joint in {boro}. {adj}, {atm}, with a great wine list. {extra}",
            "Modern Italian osteria in {boro} focusing on {pasta} and seasonal ingredients. {adj} and {atm}. {extra}",
        ],
        "Japanese": [
            "{adj} Japanese restaurant in {boro} famous for {jp_dish}. {atm}. {extra}",
            "Cozy ramen shop in {boro}. Slow-simmered broth, perfect noodles, {atm}. {extra}",
            "Omakase sushi counter in {boro}. {adj} experience with {jp_dish}. {atm}. {extra}",
        ],
        "Mexican": [
            "Authentic Mexican taqueria in {boro}. {adj} tacos, {atm}. {extra}",
            "Vibrant {boro} Mexican spot with handmade tortillas and mezcal cocktails. {adj} and lively. {extra}",
            "Street-food style Mexican in {boro}. {adj}, {atm}, with bold flavors. {extra}",
        ],
        "Chinese": [
            "{adj} Chinese restaurant in {boro} featuring {cn_dish}. {atm}. {extra}",
            "Dim sum hall in {boro}. Bustling, loud, and packed on weekends. {adj} food. {extra}",
            "Sichuan specialist in {boro} known for numbing spicy flavors and {cn_dish}. {atm}. {extra}",
        ],
        "American": [
            "{adj} American diner in {boro}. Classic burgers, fries, and milkshakes. {atm}. {extra}",
            "Farm-to-table American bistro in {boro}. Seasonal menu, {adj} atmosphere. {extra}",
            "Old-school American steakhouse in {boro}. Dry-aged beef, {adj} and {atm}. {extra}",
        ],
        "French": [
            "Charming French bistro in {boro}. {adj} ambiance, steak frites, and crème brûlée. {extra}",
            "{adj} French brasserie in {boro}. Elegant, {atm}, with classic Gallic cooking. {extra}",
            "Intimate French fine dining in {boro}. {adj}, hushed, with an extraordinary tasting menu. {extra}",
        ],
        "Indian": [
            "Aromatic Indian restaurant in {boro} with rich curries and tandoor breads. {adj} and {atm}. {extra}",
            "Modern Indian cuisine in {boro}. Inventive spice blends, {adj} decor. {extra}",
            "Casual Indian street food stall in {boro}. Cheap, flavorful, and {atm}. {extra}",
        ],
        "Thai": [
            "Lively Thai restaurant in {boro}. Fragrant curries, pad thai, {adj} atmosphere. {extra}",
            "Authentic Thai in {boro} with {adj} flavors and a great vegetarian menu. {extra}",
        ],
        "Korean": [
            "Korean BBQ in {boro}. Tabletop grills, marinated meats, {adj} and {atm}. {extra}",
            "Modern Korean in {boro} with {adj} fusion twists and rich kimchi-based sides. {extra}",
        ],
        "Mediterranean": [
            "{adj} Mediterranean mezze restaurant in {boro}. Hummus, grilled meats, {atm}. {extra}",
        ],
        "Steakhouse": [
            "Classic New York steakhouse in {boro}. Massive portions, {adj}, {atm}, legendary cuts. {extra}",
        ],
        "Pizza": [
            "New York pizza slice joint in {boro}. {adj}, {atm}, cash only, loyal regulars. {extra}",
            "Artisan wood-fired pizza in {boro}. Creative toppings, {adj}, great natural wine. {extra}",
        ],
        "Bakery": [
            "Neighborhood bakery in {boro}. {adj} pastries, fresh bread, excellent espresso. {atm}. {extra}",
        ],
        "Cafe": [
            "Specialty coffee shop in {boro}. {adj} atmosphere, single-origin pour-overs, {atm}. {extra}",
            "Cozy neighborhood cafe in {boro}, perfect for remote work, {adj} and quiet. {extra}",
        ],
        "Street Food": [
            "Street cart in {boro}. Cheap, quick, {adj}, and always busy. {atm}. {extra}",
        ],
    }

    adj_pool = ["casual", "upscale", "vibrant", "quiet", "intimate", "bustling", "elegant",
                "rustic", "trendy", "no-frills", "cozy", "minimalist", "loud", "bright",
                "dimly lit", "romantic", "family-friendly", "hipster", "old-school"]
    atm_pool = ["great for a date night", "perfect for groups", "ideal for solo dining",
                "wonderful for business lunches", "packed on weekends", "always a long wait",
                "reservations essential", "walk-ins welcome", "cash only", "BYOB friendly",
                "outdoor seating available"]
    extra_pool = ["Extensive wine list.", "Cash only.", "Open late.", "Dog friendly.",
                  "Live music on weekends.", "Vegan options available.", "BYOB welcome.",
                  "Michelin recommended.", "Zagat-rated.", "James Beard award-winning chef.",
                  "Known for very long lines.", "Reservations required months ahead.",
                  "No reservations taken.", "Best value in the neighborhood."]
    pasta_pool = ["pappardelle", "tagliatelle", "cacio e pepe", "carbonara",
                  "orecchiette", "gnocchi", "risotto", "lasagna"]
    jp_pool = ["wagyu sashimi", "tonkotsu ramen", "omakase nigiri",
               "yakitori skewers", "tempura", "chirashi bowls"]
    cn_pool = ["Peking duck", "xiao long bao", "mapo tofu",
               "hand-pulled noodles", "pork belly buns", "crispy duck"]
    boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    boro_weights = [0.38, 0.28, 0.20, 0.10, 0.04]

    prefixes = ["Casa", "Bistro", "Café", "Osteria", "Trattoria", "Kitchen", "Bar", "The",
                "Little", "Big", "Old", "New", "Corner", "Village", "Market", "Garden"]
    suffixes = ["& Co.", "NYC", "House", "Room", "Table", "Place", "Corner", "Spot",
                "Lane", "Street", "Ave", "Court"]
    middles = ["Bella", "Luna", "Verde", "Rosso", "Primo", "Oro", "Mare", "Terra", "Cielo",
               "Vino", "Fuego", "Azul", "Rouge", "Blanc", "Spice", "Smoke", "Salt", "Coal",
               "Oak", "Ember"]

    cuisines = list(templates.keys())
    cuisine_weights = [0.12, 0.10, 0.09, 0.10, 0.12, 0.08, 0.08, 0.07, 0.07,
                       0.05, 0.04, 0.04, 0.02, 0.01, 0.01]

    rows = []
    used_names = set()
    for i in range(n):
        cuisine = rng.choices(cuisines, weights=cuisine_weights, k=1)[0]
        boro = rng.choices(boros, weights=boro_weights, k=1)[0]

        for _ in range(20):
            name = f"{rng.choice(prefixes)} {rng.choice(middles)}"
            if rng.random() > 0.5:
                name += f" {rng.choice(suffixes)}"
            if name not in used_names:
                used_names.add(name)
                break

        adj = rng.choice(adj_pool)
        atm = rng.choice(atm_pool)
        extra = rng.choice(extra_pool)

        tmpl = rng.choice(templates[cuisine])
        desc = tmpl.format(
            adj=adj, atm=atm, extra=extra, boro=boro,
            pasta=rng.choice(pasta_pool),
            jp_dish=rng.choice(jp_pool),
            cn_dish=rng.choice(cn_pool),
        )

        rows.append({"Restaurant": name, "Cuisine": cuisine, "Borough": boro, "Desc": desc})

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 2. COMPUTE EMBEDDINGS
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading Transformer model & computing 384-D embeddings…")
def compute_transformer_embeddings():
    """Embed all 220 restaurants into 384-dimensional Transformer space."""
    from retrieval.vector_search import SemanticSearchModel
    corpus_df = build_pca_corpus(n=220)
    model = SemanticSearchModel(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.embed_texts(corpus_df["Desc"].tolist())
    return corpus_df, embeddings.numpy()


@st.cache_resource(show_spinner="Training Autoencoder & extracting 32-D intermediate embeddings…")
def compute_autoencoder_embeddings():
    """Train AE on synthetic operational features, extract 32-D intermediate representations."""
    from models.autoencoder import RestaurantAutoencoder, train_autoencoder

    np_rng = np.random.default_rng(42)
    n = 1200

    records = []
    for i in range(n):
        archetype = np_rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])

        if archetype == 0:    # Grade A — clean
            viol = max(0, np_rng.normal(1.5, 1.2))
            pest = max(1, np_rng.normal(14, 5))
            train_hrs = min(100, max(30, np_rng.normal(70, 12)))
            op_hrs = max(20, np_rng.normal(55, 15))
            cuisine_idx = np_rng.uniform(0.0, 0.4)
            boro_idx = np_rng.uniform(0.0, 0.4)
        elif archetype == 1:  # Grade B — mediocre
            viol = max(0, np_rng.normal(5, 2))
            pest = max(1, np_rng.normal(45, 12))
            train_hrs = min(100, max(10, np_rng.normal(30, 10)))
            op_hrs = max(20, np_rng.normal(80, 15))
            cuisine_idx = np_rng.uniform(0.3, 0.7)
            boro_idx = np_rng.uniform(0.3, 0.7)
        else:                 # Grade C — hazardous
            viol = max(0, np_rng.normal(11, 3))
            pest = max(1, np_rng.normal(120, 25))
            train_hrs = min(100, max(0, np_rng.normal(5, 5)))
            op_hrs = max(20, np_rng.normal(110, 20))
            cuisine_idx = np_rng.uniform(0.6, 1.0)
            boro_idx = np_rng.uniform(0.6, 1.0)

        grade = ['A', 'B', 'C'][archetype]
        boro = np_rng.choice(['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'])

        records.append({
            'viol': viol, 'pest': pest, 'train_hrs': train_hrs,
            'op_hrs': op_hrs, 'cuisine_idx': cuisine_idx, 'boro_idx': boro_idx,
            'Grade': grade, 'Borough': boro,
            'Violations': round(viol, 1),
            'Pest_Control_Days': round(pest, 1),
            'Training_Hours': round(train_hrs, 1),
        })

    meta_df = pd.DataFrame(records)

    feature_cols = ['viol', 'pest', 'train_hrs', 'op_hrs', 'cuisine_idx', 'boro_idx']
    X = torch.tensor(meta_df[feature_cols].values, dtype=torch.float32)

    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    ae_model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    ae_model = train_autoencoder(ae_model, X_norm, epochs=80, lr=0.003)

    intermediate_emb = ae_model.get_intermediate_embedding(X_norm).numpy()

    return meta_df, intermediate_emb


# ══════════════════════════════════════════════════════════════
# 3. PCA + K-MEANS UTILITIES
# ══════════════════════════════════════════════════════════════
def run_pca(embedding_matrix, n_components=2):
    """Apply PCA to a high-dimensional embedding matrix."""
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(embedding_matrix)
    return projected, pca.explained_variance_ratio_, pca


def run_full_pca(embedding_matrix, max_components=None):
    """Run PCA with all possible components to get the full explained variance curve (Scree plot)."""
    if max_components is None:
        max_components = min(embedding_matrix.shape[0], embedding_matrix.shape[1])
    pca_full = PCA(n_components=max_components)
    pca_full.fit(embedding_matrix)
    return pca_full.explained_variance_ratio_


def run_kmeans(projected_2d, k=5):
    """Run k-means clustering on PCA-projected 2D coordinates and return cluster labels."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(projected_2d)
    return labels, kmeans


def find_densest_cluster(projected_2d, labels, k):
    """
    Identify the densest cluster: the one whose points have the smallest average distance
    to their centroid. Returns the cluster id and a summary of its density.
    """
    from sklearn.metrics import pairwise_distances
    best_cluster = -1
    best_density = float('inf')
    cluster_stats = {}

    for c in range(k):
        mask = labels == c
        points = projected_2d[mask]
        if len(points) < 2:
            continue
        centroid = points.mean(axis=0)
        distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))
        avg_dist = distances.mean()
        cluster_stats[c] = {"count": len(points), "avg_dist": float(avg_dist)}
        if avg_dist < best_density:
            best_density = avg_dist
            best_cluster = c

    return best_cluster, cluster_stats


def summarize_cluster(df_subset, categorical_cols):
    """Extract the top characteristics of a cluster subset as a readable summary."""
    summary = {}
    for col in categorical_cols:
        counts = df_subset[col].value_counts()
        top = counts.head(3)
        summary[col] = {val: int(cnt) for val, cnt in top.items()}
    return summary


# ══════════════════════════════════════════════════════════════
# 4. INTERACTIVE UI — TWO TABS
# ══════════════════════════════════════════════════════════════
tab_transformer, tab_autoencoder = st.tabs([
    "🤖 Transformer Embeddings (384-D → 2D)",
    "🧬 Autoencoder Intermediate Embeddings (32-D → 2D)"
])

BORO_COLORS = {
    "Manhattan": "#FF3B30", "Brooklyn": "#007AFF",
    "Queens": "#34C759", "Bronx": "#FF9500", "Staten Island": "#AF52DE"
}

# ────────────────────────────────────────
# TAB 1: TRANSFORMER PCA
# ────────────────────────────────────────
with tab_transformer:
    corpus_df, transformer_emb = compute_transformer_embeddings()

    col_ctrl, col_viz = st.columns([1.2, 3.5])

    with col_ctrl:
        st.subheader("🎛️ Controls")
        t_color = st.selectbox(
            "Color PCA scatter by:",
            ["Borough", "Cuisine"],
            key="t_color",
            help="See if the Transformer model's learned representations naturally cluster by geographic or categorical features."
        )

        # ── GEOGRAPHIC FILTER (TA requirement) ──
        st.divider()
        st.subheader("🌍 Geographic Filter")
        t_boro_filter = st.multiselect(
            "Show only selected boroughs:",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            default=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            key="t_boro_filter",
            help="Filter the PCA scatter by borough to compare geographic sub-populations."
        )

        # ── K-MEANS CONTROLS (TA requirement) ──
        st.divider()
        st.subheader("🔬 k-means Clustering")
        t_k = st.slider("Number of clusters (k):", 2, 10, 5, key="t_k")
        t_show_clusters = st.toggle("Overlay k-means clusters", value=False, key="t_show_clusters")

        st.divider()
        st.subheader("📐 PCA Statistics")

        # Run PCA on FULL embeddings, then filter for display
        t_projected, t_var_ratio, t_pca = run_pca(transformer_emb, n_components=2)
        corpus_df['PCA_1'] = t_projected[:, 0]
        corpus_df['PCA_2'] = t_projected[:, 1]

        st.metric("PC1 Explained Variance", f"{t_var_ratio[0]*100:.1f}%")
        st.metric("PC2 Explained Variance", f"{t_var_ratio[1]*100:.1f}%")
        st.metric("Total (2 PCs)", f"{sum(t_var_ratio)*100:.1f}%")

        st.divider()

        # Scree plot
        st.subheader("📉 Scree Plot")
        full_var = run_full_pca(transformer_emb, max_components=min(20, transformer_emb.shape[1]))
        scree_fig = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(len(full_var))],
            y=full_var,
            marker_color='#007AFF',
            text=[f"{v*100:.1f}%" for v in full_var],
            textposition='outside',
        ))
        scree_fig.update_layout(
            yaxis_title="Explained Variance Ratio",
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
            font_size=10,
        )
        st.plotly_chart(scree_fig, use_container_width=True)

        st.divider()
        st.caption(f"**Source:** `all-MiniLM-L6-v2` Transformer · 384-D → PCA → 2D · {len(corpus_df)} restaurants")

    with col_viz:
        # Apply geographic filter
        t_display_df = corpus_df[corpus_df['Borough'].isin(t_boro_filter)].copy()

        if len(t_display_df) == 0:
            st.warning("No restaurants match the selected boroughs. Please select at least one borough.")
        else:
            # Run k-means on filtered subset
            t_proj_filtered = t_display_df[['PCA_1', 'PCA_2']].values
            t_labels, t_kmeans = run_kmeans(t_proj_filtered, k=min(t_k, len(t_display_df)))
            t_display_df['Cluster'] = t_labels.astype(str)

            if t_show_clusters:
                st.subheader(f"Transformer PCA — k-means Clusters (k={t_k})")
                fig = px.scatter(
                    t_display_df, x='PCA_1', y='PCA_2', color='Cluster',
                    hover_data=['Restaurant', 'Cuisine', 'Borough'],
                    opacity=0.8, template="plotly_white", height=600,
                    marginal_x="box", marginal_y="violin",
                    title=f"k-means clustering with k={t_k} on {len(t_display_df)} restaurants"
                )
            else:
                st.subheader(f"Transformer PCA — Colored by {t_color}")
                if t_color == "Borough":
                    fig = px.scatter(
                        t_display_df, x='PCA_1', y='PCA_2', color='Borough',
                        color_discrete_map=BORO_COLORS,
                        hover_data=['Restaurant', 'Cuisine', 'Borough'],
                        opacity=0.8, template="plotly_white", height=600,
                        marginal_x="box", marginal_y="violin",
                    )
                else:
                    fig = px.scatter(
                        t_display_df, x='PCA_1', y='PCA_2', color='Cuisine',
                        hover_data=['Restaurant', 'Cuisine', 'Borough'],
                        opacity=0.8, template="plotly_white", height=600,
                        marginal_x="box", marginal_y="violin",
                    )

            fig.update_layout(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                plot_bgcolor="#FAFAFC",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Inter, -apple-system, sans-serif",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(font_size=11),
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            st.plotly_chart(fig, use_container_width=True)

            # ── DENSEST CLUSTER ANALYSIS (TA requirement) ──
            st.subheader("🏆 Densest Cluster Analysis")
            densest_id, cluster_stats = find_densest_cluster(t_proj_filtered, t_labels, min(t_k, len(t_display_df)))

            if densest_id >= 0:
                densest_mask = t_labels == densest_id
                densest_subset = t_display_df[densest_mask]
                char_summary = summarize_cluster(densest_subset, ["Borough", "Cuisine"])

                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.metric("Densest Cluster ID", f"Cluster {densest_id}")
                with col_d2:
                    st.metric("Restaurants in Cluster", f"{cluster_stats[densest_id]['count']}")
                with col_d3:
                    st.metric("Avg Distance to Centroid", f"{cluster_stats[densest_id]['avg_dist']:.3f}")

                st.markdown("**Top characteristics of the densest cluster:**")
                for attr, counts in char_summary.items():
                    items = [f"**{val}** ({cnt})" for val, cnt in counts.items()]
                    st.markdown(f"- **{attr}:** {', '.join(items)}")

                with st.expander("📊 All cluster sizes"):
                    for c_id, stats in sorted(cluster_stats.items()):
                        marker = " ⬅️ densest" if c_id == densest_id else ""
                        st.text(f"  Cluster {c_id}: {stats['count']} restaurants, avg dist = {stats['avg_dist']:.3f}{marker}")

        with st.expander("🔬 Interpretation Guide"):
            st.markdown("""
**What to look for:**
- **Tight clusters of one color** → The model's 384-D embedding space encodes that attribute strongly.
  For example: if all Italian restaurants cluster in the top-left, the Transformer learned "Italian-ness" as a dominant feature.
- **Mixed colors everywhere** → That attribute is NOT a primary axis of variation in the embedding space.
  This is also informative! It means the model distinguishes restaurants along other dimensions (e.g., formality, price).
- **Geographic filtering** lets you compare how Manhattan vs. Brooklyn restaurants are distributed in the latent space.
  If they overlap heavily, the model treats them similarly; if they separate, there are distinct linguistic/cuisine patterns.
- **k-means densest cluster** reveals which subset of restaurants the model considers "most similar."
  Check their cuisines and boroughs to understand what drives this similarity.

**Technical Detail:** PCA finds the directions of maximum variance in the 384-D space.
PC1 captures the most variance, PC2 the second-most. Together they show the "best 2D summary" of all 384 dimensions.
""")


# ────────────────────────────────────────
# TAB 2: AUTOENCODER PCA
# ────────────────────────────────────────
with tab_autoencoder:
    ae_df, ae_emb = compute_autoencoder_embeddings()

    col_ctrl2, col_viz2 = st.columns([1.2, 3.5])

    with col_ctrl2:
        st.subheader("🎛️ Controls")
        ae_color = st.selectbox(
            "Color PCA scatter by:",
            ["Grade", "Borough", "Violations", "Pest_Control_Days", "Training_Hours"],
            key="ae_color",
            help="Validate whether the Autoencoder's learned 32-D intermediate space captures meaningful structure."
        )

        # ── GEOGRAPHIC FILTER (TA requirement) ──
        st.divider()
        st.subheader("🌍 Geographic Filter")
        ae_boro_filter = st.multiselect(
            "Show only selected boroughs:",
            ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            default=["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            key="ae_boro_filter",
            help="Filter to compare restaurant operational profiles across different boroughs."
        )

        # ── K-MEANS CONTROLS (TA requirement) ──
        st.divider()
        st.subheader("🔬 k-means Clustering")
        ae_k = st.slider("Number of clusters (k):", 2, 10, 5, key="ae_k")
        ae_show_clusters = st.toggle("Overlay k-means clusters", value=False, key="ae_show_clusters")

        st.divider()
        st.subheader("📐 PCA Statistics")

        # Run PCA on ALL autoencoder embeddings
        ae_projected, ae_var_ratio, ae_pca = run_pca(ae_emb, n_components=2)
        ae_df['PCA_1'] = ae_projected[:, 0]
        ae_df['PCA_2'] = ae_projected[:, 1]

        st.metric("PC1 Explained Variance", f"{ae_var_ratio[0]*100:.1f}%")
        st.metric("PC2 Explained Variance", f"{ae_var_ratio[1]*100:.1f}%")
        st.metric("Total (2 PCs)", f"{sum(ae_var_ratio)*100:.1f}%")

        st.divider()

        # Scree plot
        st.subheader("📉 Scree Plot")
        ae_full_var = run_full_pca(ae_emb, max_components=min(20, ae_emb.shape[1]))
        ae_scree_fig = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(len(ae_full_var))],
            y=ae_full_var,
            marker_color='#AF52DE',
            text=[f"{v*100:.1f}%" for v in ae_full_var],
            textposition='outside',
        ))
        ae_scree_fig.update_layout(
            yaxis_title="Explained Variance Ratio",
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
            font_size=10,
        )
        st.plotly_chart(ae_scree_fig, use_container_width=True)

        st.divider()
        st.caption(f"**Source:** `RestaurantAutoencoder` intermediate layer · 32-D → PCA → 2D · {len(ae_df)} restaurants")

    with col_viz2:
        # Apply geographic filter
        ae_display_df = ae_df[ae_df['Borough'].isin(ae_boro_filter)].copy()

        if len(ae_display_df) == 0:
            st.warning("No restaurants match the selected boroughs. Please select at least one borough.")
        else:
            # Run k-means on filtered subset
            ae_proj_filtered = ae_display_df[['PCA_1', 'PCA_2']].values
            ae_labels, ae_kmeans = run_kmeans(ae_proj_filtered, k=min(ae_k, len(ae_display_df)))
            ae_display_df['Cluster'] = ae_labels.astype(str)

            if ae_show_clusters:
                st.subheader(f"Autoencoder PCA — k-means Clusters (k={ae_k})")
                fig2 = px.scatter(
                    ae_display_df, x='PCA_1', y='PCA_2', color='Cluster',
                    hover_data=['Grade', 'Borough', 'Violations', 'Pest_Control_Days', 'Training_Hours'],
                    opacity=0.75, template="plotly_white", height=600,
                    marginal_x="box", marginal_y="violin",
                    title=f"k-means clustering with k={ae_k} on {len(ae_display_df)} restaurants"
                )
            else:
                st.subheader(f"Autoencoder PCA — Colored by {ae_color}")
                is_categorical = ae_color in ["Grade", "Borough"]

                if is_categorical:
                    if ae_color == "Grade":
                        color_map = {'A': '#34C759', 'B': '#FFCC00', 'C': '#FF3B30'}
                    else:
                        color_map = BORO_COLORS
                    fig2 = px.scatter(
                        ae_display_df, x='PCA_1', y='PCA_2', color=ae_color,
                        color_discrete_map=color_map,
                        hover_data=['Grade', 'Borough', 'Violations', 'Pest_Control_Days', 'Training_Hours'],
                        opacity=0.75, template="plotly_white", height=600,
                        marginal_x="box", marginal_y="violin",
                    )
                else:
                    c_scale = "Aggrnyl" if ae_color == "Training_Hours" else "OrRd"
                    fig2 = px.scatter(
                        ae_display_df, x='PCA_1', y='PCA_2', color=ae_color,
                        color_continuous_scale=c_scale,
                        hover_data=['Grade', 'Borough', 'Violations', 'Pest_Control_Days', 'Training_Hours'],
                        opacity=0.8, template="plotly_white", height=600,
                        marginal_x="box", marginal_y="violin",
                    )

            fig2.update_layout(
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                plot_bgcolor="#FAFAFC",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Inter, -apple-system, sans-serif",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(font_size=11),
            )
            fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            st.plotly_chart(fig2, use_container_width=True)

            # ── DENSEST CLUSTER ANALYSIS (TA requirement) ──
            st.subheader("🏆 Densest Cluster Analysis")
            ae_densest_id, ae_cluster_stats = find_densest_cluster(
                ae_proj_filtered, ae_labels, min(ae_k, len(ae_display_df))
            )

            if ae_densest_id >= 0:
                ae_densest_mask = ae_labels == ae_densest_id
                ae_densest_subset = ae_display_df[ae_densest_mask]
                ae_char_summary = summarize_cluster(ae_densest_subset, ["Grade", "Borough"])

                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("Densest Cluster ID", f"Cluster {ae_densest_id}")
                with col_a2:
                    st.metric("Restaurants in Cluster", f"{ae_cluster_stats[ae_densest_id]['count']}")
                with col_a3:
                    st.metric("Avg Distance to Centroid", f"{ae_cluster_stats[ae_densest_id]['avg_dist']:.3f}")

                st.markdown("**Top characteristics of the densest cluster:**")
                for attr, counts in ae_char_summary.items():
                    items = [f"**{val}** ({cnt})" for val, cnt in counts.items()]
                    st.markdown(f"- **{attr}:** {', '.join(items)}")

                # Show avg continuous features for dense cluster
                st.markdown("**Average operational features in densest cluster:**")
                avg_viol = ae_densest_subset['Violations'].mean()
                avg_pest = ae_densest_subset['Pest_Control_Days'].mean()
                avg_train = ae_densest_subset['Training_Hours'].mean()
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    st.metric("Avg Violations", f"{avg_viol:.1f}")
                with col_f2:
                    st.metric("Avg Pest Control Days", f"{avg_pest:.1f}")
                with col_f3:
                    st.metric("Avg Training Hours", f"{avg_train:.1f}")

                with st.expander("📊 All cluster sizes"):
                    for c_id, stats in sorted(ae_cluster_stats.items()):
                        marker = " ⬅️ densest" if c_id == ae_densest_id else ""
                        st.text(f"  Cluster {c_id}: {stats['count']} restaurants, avg dist = {stats['avg_dist']:.3f}{marker}")

        with st.expander("🔬 Interpretation Guide"):
            st.markdown("""
**What to look for in the Autoencoder PCA:**
- **Grade clustering (A/B/C)** is the most important validation. If grades form distinct clusters, the
  Autoencoder's intermediate representation successfully captures the *risk profile* that determines health grades.
- **Borough clustering:** If boroughs separate, it suggests geographic location correlates with operational patterns
  (e.g., Manhattan restaurants may have stricter compliance than outer boroughs).
- **Continuous features (Violations, Pest Control, Training):** Smooth color gradients across the PCA space
  confirm that the learned representation encodes these features linearly — a sign of a well-trained model.
- **k-means densest cluster** in this tab typically reveals the "Grade A" archetype — low violations, recent pest
  control, high training hours. This validates that the Autoencoder learned a meaningful health-risk manifold.

**Why 32-D intermediate, not 2-D latent?**
The final 2-D latent space is already the Autoencoder's own dimensionality reduction.
Applying PCA to 2D outputs would be trivial. Instead, we extract the 32-D *intermediate* encoder representation,
which contains richer information before the final bottleneck compression.
PCA on this 32-D space reveals which directions the Autoencoder considers most important.
""")


# ══════════════════════════════════════════════════════════════
# 5. ARCHITECTURE SUMMARY
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
### 🏗️ Architecture Overview
| Source | Original Dim | PCA Target | Clustering | Model |
|---|---|---|---|---|
| Transformer Embedding | 384 | 2 | k-means | `sentence-transformers/all-MiniLM-L6-v2` via `retrieval/vector_search.py` |
| Autoencoder Intermediate | 32 | 2 | k-means | `RestaurantAutoencoder` encoder layer 3 via `models/autoencoder.py` |

**Course Topics Covered:** Week 4 (Embeddings & Similarity), Week 6 (PCA, Autoencoders), Week 7 (k-means Clustering)

*Built by Hollan Yuan — CSCI-UA 473 · NYU · Spring 2026*
""")
