# 🍎 NYC Restaurant Survival Guide

**CSCI-UA 473 — Fundamentals of Machine Learning**
New York University · Spring 2026

An end-to-end interactive Machine Learning dashboard that helps New Yorkers
find restaurants they will actually enjoy and trust.  Real NYC DOHMH
inspection data is fused with Google Places metadata, then put to work in
five connected tools: semantic search, a health-grade classifier, a
restaurant clustering explorer (GIS + 3D PCA), and a personalized
recommendation engine.

Built with **PyTorch**, **Streamlit**, **Plotly**, **pydeck**, and
**HuggingFace Transformers**.

For rubric-facing notes and likely TA questions, see
[PRESENTATION_QA_README.md](PRESENTATION_QA_README.md).

---

## What the App Does

| Page | Purpose |
|---|---|
| 🔍 **Semantic Search** | Type what you want ("cozy Italian pasta spot in Brooklyn") and get ranked restaurant matches from a prepared dataset of **~2,835 real NYC restaurants** enriched with Google Places data. |
| 🧪 **Health Grade Classifier** | Pick a real NYC restaurant, get the predicted DOHMH health grade (A / B / C) with class probabilities. Held-out test-set performance is shown below so the audience can verify the classifier is a real trained MLP, not a rule. |
| 📍 **Restaurant Cluster GIS Map** | Cluster the ~2,835 restaurants with three comparable algorithms (K-Means from scratch, Gaussian Mixture, Ward) in a 22-dim interpretable feature space (price / rating / review volume / health / cuisine / borough / geo-location). Restaurants are colored by cluster on a real NYC map with persona labels + narrative stories that explain *why* each cluster exists. |
| 📊 **PCA Embedding Explorer** | Projects the same clusters into 3-D PCA space with feature-loading bar charts, cluster-distance heatmap, and prototype restaurants nearest each centroid. |
| 🔮 **Personalized Recommendations** | Given your saved profile + liked restaurants, ranks candidates by per-liked cosine KNN fused via **Reciprocal Rank Fusion**, re-ranked with **Maximal Marginal Relevance** for cuisine diversity, and a **cuisine-alignment multiplier** so Chinese-preferring users see Chinese restaurants first. |

---

## Key Models & Algorithms

### Semantic Search
- **Encoder:** HuggingFace `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **Technique:** Transformer embeddings → L2 normalization → cosine similarity
- **Code:** `utils/search.py` + `retrieval/vector_search.py`
- **Course Topics:** Week 3 (Transformers), Week 4 (Similarity Metrics, NN search)

### Health Grade Classifier — `CustomMLP`
- **Architecture:** 3-layer MLP (29 → 128 → 128 → 3) with ReLU + dropout
- **Training:** Adam + CrossEntropyLoss with class weighting, early stopping
  on validation F1, hyperparameter grid search over (hidden_dim, lr, dropout)
- **Data:** Real NYC DOHMH inspection records (~14,000 restaurants, stratified
  80/20 train/test split)
- **From-Scratch Requirement:** ✅ `models/custom_mlp.py` implements the
  model, training loop, evaluation, gradient-based feature importance,
  permutation importance, and counterfactual adversarial search — no
  scikit-learn wrappers.
- **Course Topics:** Week 3 (Optimization), Week 8 (Classification, F1)

### Restaurant Clustering
- **Primary algorithm:** `KMeansScratch` — our own NumPy K-Means with
  k-means++ initialization, small-cluster merging, and stable global label
  reindexing.  Code in `models/kmeans_scratch.py`.
- **Comparison baselines:** `sklearn.mixture.GaussianMixture` (tied
  covariance) and `sklearn.cluster.AgglomerativeClustering` (Ward linkage).
- **Feature space (22-D):** standardized price tier, Google rating, review
  volume (log-scaled), DOHMH health score, lat/lng, + cuisine one-hot
  (top-10 + Other), + borough one-hot.  All features are interpretable —
  no learned embeddings — so cluster personas can be read off directly.
- **Dimensionality reduction:** our own `models/pca_scratch.py` for cluster
  projection + automatic persona-aware axis labeling.
- **Cluster labeling:** persona-based three-slot labels like
  *"Chinese · Reliable"*, *"Pizza · Budget"*, *"Donuts · Under-the-Radar"*
  combining dominant cuisine, borough concentration, and rating-review
  persona (Hidden Gem / Tourist Favorite / Reliable / Under-the-Radar /
  Overhyped) or price tier (Budget / Mid-Range / Upscale / Luxury).
  Clusters that aren't cuisine-driven are labeled *"Mixed Cuisine"* and the
  narrative story explicitly says which signals (price / rating / location)
  bind them together.
- **Course Topics:** Week 7 (K-Means, GMM, hierarchical clustering), Week 6
  (PCA, dimensionality reduction)

### Recommendations — Per-Liked KNN + RRF + MMR + Cuisine Alignment
- **Per-liked retrieval:** for each restaurant the user liked, cosine-KNN
  against the full candidate pool in the 22-dim feature space.
- **Rank fusion:** Reciprocal Rank Fusion (Cormack et al. 2009) aggregates
  the per-liked rankings with a small profile-similarity bias term.
- **Diversity re-ranking:** Maximal Marginal Relevance on the top-50 with
  tunable λ so the final list isn't dominated by one cuisine.
- **Cuisine alignment:** `cuisine_alignment_score()` applies a multiplicative
  boost so explicit `favorite_cuisines` matches get full score and
  mismatches are reduced to 15%; implicit preferences from likes history
  reduce mismatches to 30%.  Keeps price / rating / location as ranking
  signals *within* the preferred cuisine.
- **Code:** `utils/clustering.py` — `build_user_feature_vector`,
  `cuisine_alignment_score`, `recommend_per_liked_knn`, `apply_mmr`.

### Autoencoder (Legacy / Research)
- **Model:** Encoder(6→64→32→2) + Decoder(2→32→64→6), Adam + MSELoss
- **Code:** `models/autoencoder.py`
- Retained for coursework completeness; not wired into the main app UI.

---

## Repository Structure

```
nyc-restaurant-survival-guide/
├── app/                                  # Streamlit frontend
│   ├── main.py                           # Landing page & profile onboarding
│   ├── ui_utils.py                       # Apple-inspired CSS theme
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py
│       ├── 2_🧪_Health_Grade_Classifier.py
│       ├── 3_📍_Restaurant_Cluster_Map.py
│       ├── 4_📊_PCA_Embedding_Explorer.py
│       └── 5_🔮_Recommendations.py
├── models/                               # From-scratch PyTorch / NumPy models
│   ├── custom_mlp.py                     # MLP + training + counterfactuals
│   ├── autoencoder.py                    # Deep autoencoder (legacy)
│   ├── kmeans_scratch.py                 # NumPy K-Means
│   └── pca_scratch.py                    # NumPy PCA
├── retrieval/
│   └── vector_search.py                  # Transformer embedding + cosine search
├── utils/                                # App-layer glue
│   ├── auth.py                           # Lightweight username/password auth
│   ├── clustering.py                     # Feature matrix, clustering pipeline,
│   │                                     #   cluster personas, KNN + MMR + RRF
│   ├── data.py                           # DOHMH API fetcher (with caching)
│   ├── google_places.py                  # Google Places enrichment
│   ├── search.py                         # Embedding cache + semantic ranking
│   ├── search_assets.py                  # Cache-first prepared-data loader
│   └── user_profile.py                   # Profile CRUD + history → features
├── data/                                 # Pipeline outputs + caches
│   ├── download_data.py                  # NYC DOHMH OpenData fetcher
│   ├── preprocess.py                     # Feature engineering, train/test split
│   ├── train.csv / test.csv              # MLP training data (29-D)
│   ├── meta_train.csv / meta_test.csv    # Restaurant metadata for UI
│   ├── feature_config.json               # Scaler params + column order
│   └── cache/
│       ├── prepared_search_v4_3800.pkl   # 2,835 NYC + Google-enriched rows
│       ├── embeddings_prepared_v4_*.npy  # Cached sentence-transformer output
│       └── health_classifier.pt          # Trained MLP checkpoint
├── tests/
│   ├── test_custom_mlp.py                # Forward shape + training convergence
│   ├── test_autoencoder.py               # Forward shape + loss reduction
│   └── test_semantic_search.py           # Embedding shape + L2 + relevance
├── .streamlit/config.toml                # Streamlit theme
├── Pipfile / requirements.txt            # Dependencies
└── README.md                             # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+  (3.11 recommended)
- `pip` and optionally `pipenv`
- **Important:** the project now requires `numpy>=2.0` and `pandas>=2.2`
  for compatibility with the committed pkl cache.

### Option A — pipenv (recommended)
```bash
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide
pipenv install
pipenv run streamlit run app/main.py
```

### Option B — pip
```bash
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide
python -m venv venv
source venv/bin/activate           # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

### No Google API key needed
The repo ships with `data/cache/prepared_search_v4_3800.pkl` (~2,835
restaurants already enriched with Google Places metadata) and the
corresponding embedding cache, so the app runs immediately without any
API key or network call.  The live DOHMH / Google Places path is only used
when you tick "Refresh cached Google Places sample" in the Semantic Search
sidebar.

### Optional — rebuild the dataset from scratch
```bash
# Fetch ~50,000 DOHMH rows
pipenv run python data/download_data.py
# Preprocess into train/test splits
pipenv run python data/preprocess.py
# To rebuild the Google-enriched cache you need a Places API key
# in .streamlit/secrets.toml (copy from .streamlit/secrets.toml.example)
```

---

## Running Tests

```bash
pipenv run pytest tests/ -v
```

Covers the MLP (forward shape + training convergence), Autoencoder
(forward shape + loss reduction + latent output), and Semantic Search
(embedding shape + L2 norm + semantic relevance).

---

## How to Use the Dashboard

1. **Open the app** at `http://localhost:8501`.
2. **Log in or sign up** (profiles persist in `data/user_profiles.json`).
3. **Semantic Search** — type a natural-language craving or occasion,
   filter by borough / grade / rating, pick a minimum match threshold.
   Click ❤️ to save restaurants to your profile for Recommendations.
4. **Health Grade Classifier** — search the held-out test set, click a
   row, see the predicted grade, class probabilities, and the true grade
   for comparison.  The Model Performance section below shows confusion
   matrix and per-class F1 / precision / recall.
5. **Restaurant Cluster GIS Map** — pick an algorithm (K-Means / GMM /
   Ward) and `K` in the sidebar.  Restaurants are drawn as 3D columns
   on a real NYC map, colored by cluster persona.  Scroll down for the
   per-cluster summary cards with cuisine mix, borough mix, average
   rating/price, and a narrative story.
6. **PCA Embedding Explorer** — view the same clusters in 3-D PCA space.
   Inspect feature loadings per principal component, the cluster-distance
   heatmap, and prototype restaurants nearest each centroid.
7. **Recommendations** — set your budget / favorite cuisines / boroughs
   in the sidebar, or add liked restaurants.  The top-15 list respects
   your stated cuisine preferences and shows which liked restaurant
   influenced each recommendation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, pydeck |
| ML Models (from scratch) | PyTorch (MLP, autoencoder), NumPy (K-Means, PCA) |
| Classical ML baselines | scikit-learn (GMM, Ward, sanity checks) |
| NLP | HuggingFace Transformers — `sentence-transformers/all-mpnet-base-v2` |
| Data | Pandas, NumPy, NYC DOHMH OpenData API, Google Places API |
| Testing | Pytest |

---

## Authors

- **Hollan Yuan** — Initial project framework and architecture; Google
  Places API integration and restaurant description generation that made
  semantic search functional.
- **Ryan Han(PapTR / Pap)** — NYC DOHMH Open Data integration and real
  restaurant dataset; data preprocessing pipeline; pre-presentation
  engineering: cache-first loading so the app runs without a Google API
  key, persona-based cluster labels with narrative stories, cuisine-aware
  recommendation ranking, simplified classifier UI, numpy 2 / pandas 3
  compatibility debugging.
- **Rahul Adusumalli** — Health-grade classifier work; cluster and
  semantic-search refinements.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
