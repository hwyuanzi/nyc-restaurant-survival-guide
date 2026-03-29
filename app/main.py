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
Navigating the New York City restaurant scene can be overwhelming. This multidimensional dashboard helps you answer two critical questions:
1. **Find by Vibe:** Use our *Semantic Search* (powered by HuggingFace Transformer embeddings) to find restaurants that match your exact culinary and atmospheric desires.
2. **Health Risk Predictor:** Use our *Custom PyTorch MLP* perfectly trained from scratch to predict whether a restaurant is statistically likely to maintain an 'A' health grade, based on historical DOHMH inspection patterns.
3. **Explore the Latent Landscape:** Visualize high-dimensional restaurant features compressed into a sleek 2D mathematical map via our *Deep Autoencoder*.
4. **Geospatial Intelligence:** Interactively explore real geographical distributions, 3D Hexagon density heatmaps, and neighborhood "Safe Zones" overlaid on the beautiful NYC map.

👈 **Use the sidebar** to navigate between the different ML tools!

---
*Created for CSCI-UA 473 Machine Learning (NYU, Spring 2026).*
""")
