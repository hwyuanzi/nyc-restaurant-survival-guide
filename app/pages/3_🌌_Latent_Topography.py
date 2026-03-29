import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.express as px

from models.autoencoder import RestaurantAutoencoder, train_autoencoder

st.set_page_config(page_title="Restaurant Landscape", page_icon="🗺️", layout="wide")

from app.ui_utils import apply_apple_theme
apply_apple_theme()

st.title("🌌 Autoencoder Latent Topography")

st.markdown("""
### 🎯 What is this?
A **real 2D Latent Space Projection** produced by our PyTorch **Deep Autoencoder** (`models/autoencoder.py`).

### 🧠 How does it work?
We generate a synthetic dataset of restaurants with 6 operational features, then **actually train** the `RestaurantAutoencoder` neural network to compress those 6 dimensions into just 2 latent coordinates (X, Y). The Autoencoder learns to preserve the most important structure — restaurants with similar safety profiles naturally cluster together, *without any human labels*.

### 💡 How to explore?
1. **Change the Color Mapping** to see how latent clusters align with real-world features.
2. **Hover over dots** to inspect individual restaurant profiles.
3. **Analyze the Marginal distributions** (violin + box plots) on the axes.
---
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC FEATURE DATA
# ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training Autoencoder neural network...")
def train_and_project():
    """
    Generates realistic synthetic restaurant features, trains the Autoencoder,
    and projects the data into 2D latent space. All using REAL PyTorch operations.
    """
    np_rng = np.random.default_rng(42)
    n = 1200

    # 6 input features for each restaurant
    # We create 3 archetypes: Clean (A), Average (B), Hazardous (C)
    records = []
    for i in range(n):
        archetype = np_rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15])

        if archetype == 0:  # Grade A — clean
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
        else:  # Grade C — hazardous
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
            'Machine_Grade': grade, 'Borough': boro,
            'Past_Violations': round(viol, 1),
            'Pest_Control_Days': round(pest, 1),
            'Training_Hours': round(train_hrs, 1),
        })

    meta_df = pd.DataFrame(records)

    # Build the 6-feature tensor for the Autoencoder
    feature_cols = ['viol', 'pest', 'train_hrs', 'op_hrs', 'cuisine_idx', 'boro_idx']
    X = torch.tensor(meta_df[feature_cols].values, dtype=torch.float32)

    # Normalize features to [0, 1] range for stable autoencoder training
    X_min = X.min(dim=0).values
    X_max = X.max(dim=0).values
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    # ── ACTUALLY TRAIN THE AUTOENCODER ──
    ae_model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    ae_model = train_autoencoder(ae_model, X_norm, epochs=80, lr=0.003)

    # ── GET REAL LATENT SPACE COORDINATES ──
    latent_coords = ae_model.get_latent_space(X_norm).numpy()

    meta_df['Latent_X'] = latent_coords[:, 0]
    meta_df['Latent_Y'] = latent_coords[:, 1]

    return meta_df


df = train_and_project()

# ──────────────────────────────────────────────────
# 2. INTERACTIVE VISUALIZATION
# ──────────────────────────────────────────────────
col_charts, col_insights = st.columns([3.5, 1.2])

with col_insights:
    st.subheader("🎛️ Diagnostic Controls")
    color_metric = st.selectbox(
        "Color Latent Space By:",
        ("Machine_Grade", "Past_Violations", "Pest_Control_Days", "Training_Hours"),
        help="Watch how the Autoencoder's latent clusters align with real-world features."
    )

    st.divider()

    st.subheader("💡 Cluster Insights")
    st.info("""
    **Cluster Alpha (dense region):**
    Hover over these dots — they are universally low-violation, high-training restaurants. The Autoencoder learned to group them together without labels.
    """)
    st.warning("""
    **Cluster Beta (mid-region):**
    Borderline entities with ~45-day pest control gaps. The neural network placed them in the transition zone between safe and failing.
    """)
    st.error("""
    **Cluster Gamma (isolated outliers):**
    Hover here — catastrophic violation counts and 100+ day pest abandonment. The Autoencoder mathematically isolated these health hazards.
    """)

    st.divider()
    st.caption(f"📐 **Architecture:** `RestaurantAutoencoder(6→64→32→2→32→64→6)` · Trained {80} epochs · {len(df)} data points")

with col_charts:
    if color_metric == "Machine_Grade":
        fig = px.scatter(
            df, x='Latent_X', y='Latent_Y', color='Machine_Grade',
            marginal_y="violin", marginal_x="box",
            hover_data=['Borough', 'Past_Violations', 'Pest_Control_Days', 'Training_Hours'],
            color_discrete_map={'A': '#34C759', 'B': '#FFCC00', 'C': '#FF3B30'},
            opacity=0.75, template="plotly_white", height=700
        )
    else:
        c_scale = "Aggrnyl" if color_metric == "Training_Hours" else "OrRd"
        fig = px.scatter(
            df, x='Latent_X', y='Latent_Y', color=color_metric,
            marginal_y="violin", marginal_x="box",
            hover_data=['Machine_Grade', 'Borough', 'Past_Violations', 'Pest_Control_Days', 'Training_Hours'],
            color_continuous_scale=c_scale,
            opacity=0.85, template="plotly_white", height=700
        )

    fig.update_layout(
        plot_bgcolor="#FAFAFC",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_xaxes(title_text="Autoencoder Latent Dimension 1", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(title_text="Autoencoder Latent Dimension 2", showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')

    st.plotly_chart(fig, use_container_width=True)
