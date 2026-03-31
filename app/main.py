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
Navigating the New York City restaurant scene can be overwhelming. This multidimensional dashboard leverages five machine learning tools to help you explore, predict, and discover hidden patterns across 27,000+ NYC restaurants:

1. **🔍 Semantic Vibe Search:** Use our *Transformer-based NLP engine* to find restaurants that match your exact culinary and atmospheric desires via cosine similarity in a 384-dimensional embedding space.
2. **🧪 ML Action Sandbox:** Use our *Custom PyTorch MLP* (trained from scratch via gradient descent) to predict health grades and explore counterfactual "what-if" scenarios in real-time.
3. **🌌 Latent Topography:** Visualize high-dimensional restaurant features compressed into a 2D map via our *Deep Autoencoder*, revealing unsupervised cluster structure.
4. **📍 Geospatial GIS:** Interactively explore geographic distributions with 3D hexagonal density heatmaps and multi-feature scatter maps across NYC's five boroughs.
5. **📊 PCA Embedding Explorer:** Apply *Principal Component Analysis* to Transformer and Autoencoder embeddings, filter by borough, and run *k-means clustering* to identify the densest restaurant groups and their shared characteristics.

👈 **Use the sidebar** to navigate between the five ML tools!

---
*CSCI-UA 473 · Fundamentals of Machine Learning · New York University · Spring 2026*
*Built by Hollan Yuan*
""")
