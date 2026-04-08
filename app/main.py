"""
main.py — NYC Restaurant Survival Guide landing page.

Original dashboard framework: Hollan Yuan
Real-data pipeline & user profiles: Rahul Adusumalli
Project integration: Ryan Han (PapTR)
"""

import streamlit as st

st.set_page_config(
    page_title="NYC Restaurant Survival Guide",
    page_icon="🍽️",
    layout="wide",
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.ui_utils import apply_apple_theme
apply_apple_theme()

st.title("🍎 NYC Restaurant Survival Guide")
st.markdown("""
### Welcome to the Ultimate NYC Dining ML Dashboard!
Navigating the New York City restaurant scene can be overwhelming. This multidimensional dashboard leverages seven machine learning tools to help you explore, predict, and discover hidden patterns across NYC restaurants:

1. **🔍 Semantic Vibe Search:** Use our *Transformer-based NLP engine* to find restaurants that match your exact culinary desires via cosine similarity in a 384-dimensional embedding space.
2. **🧪 ML Action Sandbox:** Use our *Custom PyTorch MLP* (trained from scratch via gradient descent) to predict health grades and explore counterfactual "what-if" scenarios in real-time.
3. **🌌 Latent Topography:** Visualize high-dimensional restaurant features compressed into a 2D map via our *Deep Autoencoder*, revealing unsupervised cluster structure.
4. **📍 Geospatial GIS:** Interactively explore geographic distributions with 3D hexagonal density heatmaps and multi-feature scatter maps across NYC's five boroughs.
5. **📊 PCA Embedding Explorer:** Apply *Principal Component Analysis* to Transformer and Autoencoder embeddings, filter by borough, and run *k-means clustering* to identify the densest restaurant groups.
6. **🔎 Live Semantic Search:** Search **real NYC restaurants** enriched with Google Places data — ratings, photos, reviews — with results personalized to your saved profile.
7. **🔮 Cluster Recommendations:** Get personalized restaurant recommendations based on K-Means taste clustering and your saved likes.

👈 **Use the sidebar** to navigate between the seven ML tools!

---

### 👥 Team Credits

| Member | Contribution |
|---|---|
| **Hollan Yuan** | PyTorch MLP classifier, Deep Autoencoder, Apple UI theme, PCA explorer, GIS maps, Semantic Vibe Search |
| **Rahul Adusumalli** | Real-data semantic search (768-D), Google Places enrichment, user profile system, recommendation engine, clustering pipeline |
| **Ryan Han (PapTR)** | Project integration, code merging, data pipeline, coordination |

---
*CSCI-UA 473 · Fundamentals of Machine Learning · New York University · Spring 2026*
""")
