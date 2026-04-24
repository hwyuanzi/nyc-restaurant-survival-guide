# NYC Restaurant Survival Guide

**CSCI-UA 473 — Fundamentals of Machine Learning**
New York University · Spring 2026

A full-stack interactive Machine Learning dashboard for exploring, searching, and predicting health outcomes of NYC restaurants. The app combines real NYC Department of Health inspection data, pre-trained Transformer embeddings, and custom-built PyTorch models to let users semantically search restaurants, predict health grades, visualize geospatial clusters, and receive personalized recommendations — all from a single Streamlit interface.

---

## Authors

| Name | GitHub |
|---|---|
| Hollan Yuan | [hwyuanzi](https://github.com/hwyuanzi) |
| TBD | — |
| TBD | — |
| TBD | — |
| TBD | — |

---

## Table of Contents

1. [Features & Algorithms](#features--algorithms)
2. [Datasets](#datasets)
3. [ML Models & Algorithms In Depth](#ml-models--algorithms-in-depth)
4. [Repository Structure](#repository-structure)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [Running Tests](#running-tests)
7. [How to Use the Dashboard](#how-to-use-the-dashboard)
8. [Tech Stack](#tech-stack)
9. [License](#license)

---

## Features & Algorithms

### 1. Semantic Vibe Search

Search a corpus of NYC restaurants by describing the atmosphere or cuisine you want in plain English (e.g., *"cozy romantic Italian with dim lighting"* or *"cheap spicy ramen for late night"*).

| Component | Detail |
|---|---|
| **Model called** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional Transformer embeddings) |
| **Technique** | Encode query → encode restaurant descriptions → L2 normalize → cosine similarity via `torch.mm()` → `torch.topk()` for top-k ranking |
| **Implementation** | `retrieval/vector_search.py` — `SemanticSearchModel` |
| **Enrichment** | Top results are enriched with Google Places ratings, photos, and price tier |

**How it works:** The model maps both the query and every restaurant description into the same 384-D semantic vector space. Because vectors are L2-normalized, the dot product equals cosine similarity, making nearest-neighbor search a single matrix multiply. This means *"upscale sushi"* and *"expensive Japanese omakase"* surface the same restaurants even though they share no keywords.

---

### 2. Health Grade Classifier (ML Action Sandbox)

Select any real NYC restaurant, adjust its operational parameters (violation count, pest control days, training hours, inspection score) with sliders, and watch the neural network re-predict the health grade in real time with a live radar chart and 3-class probability display.

| Component | Detail |
|---|---|
| **Model (from scratch)** | `CustomMLP` — 3-layer Multi-Layer Perceptron implemented in PyTorch |
| **Architecture** | `FC(29 → 128) → ReLU → Dropout(0.3) → FC(128 → 64) → ReLU → Dropout(0.3) → FC(64 → 3)` |
| **Training data** | ~14,000 real NYC DOHMH restaurants, 29 features (7 numerical + 6 borough dummies + 16 cuisine dummies) |
| **Optimizer** | AdamW with class-weighted `CrossEntropyLoss` (compensates for imbalanced A:B:C split ≈ 78:16:6) |
| **Hyperparameter search** | Grid search over `hidden_dim`, `lr`, `dropout`; results cached in `data/cache/hp_search_results.json` |
| **Counterfactual engine** | Gradient descent on the input vector to find the minimal feature perturbation that flips the predicted grade to A |
| **Interpretability** | Gradient-based feature importance + permutation importance |
| **Implementation** | `models/custom_mlp.py` |

**How it works:** The MLP learns a non-linear boundary in 29-dimensional inspection-feature space separating A, B, and C graded restaurants. At inference, a slider-adjusted feature vector is passed through the trained network; the softmax output gives the probability of each grade. The counterfactual engine runs gradient descent *on the input* (not the weights) to find the smallest change in violation score, pest control days, etc. that pushes the prediction across the A threshold — surfaced in the UI as actionable "what-if" recommendations.

---

### 3. Restaurant Cluster GIS Map

Explore real NYC restaurants across all 5 boroughs on a 3D hexagonal density map (PyDeck) and a multi-feature spatial scatter (Plotly Mapbox). Color by health grade, total violations, pest control frequency, or training hours.

| Component | Detail |
|---|---|
| **Clustering algorithm (from scratch)** | `KMeansScratch` — K-Means++ initialization + iterative E/M steps in NumPy |
| **Comparison baselines** | Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`) and Ward Hierarchical Clustering (`sklearn.cluster.AgglomerativeClustering`) |
| **Features clustered** | Price tier, rating, health score, cuisine type (encoded), borough (encoded), latitude, longitude |
| **Cluster quality** | Silhouette score computed automatically; best k chosen by inertia elbow |
| **Visualization** | PyDeck `HexagonLayer` + `ScatterplotLayer`; Plotly Express `scatter_mapbox` |
| **Implementation** | `models/kmeans_scratch.py`, `utils/clustering.py`, `app/pages/3_📍_Restaurant_Cluster_GIS_Map.py` |

**How it works:** K-Means++ selects initial centroids by choosing each successive centroid with probability proportional to its squared distance from the nearest already-chosen centroid. This gives a spread-out initialization that dramatically reduces the risk of poor local minima compared to random seeding. The E-step assigns each restaurant to its nearest centroid (Euclidean distance); the M-step recomputes centroids as cluster means. This iterates until centroid shifts fall below a convergence threshold. Multi-start runs are compared by inertia and the best is retained.

---

### 4. PCA Embedding Explorer

Apply Principal Component Analysis to high-dimensional embeddings from two different model sources and explore whether learned representations naturally separate restaurants by geography, cuisine, or health profile.

| Component | Detail |
|---|---|
| **Embedding sources** | (A) Transformer embeddings: `all-MiniLM-L6-v2` produces 384-D vectors from restaurant descriptions; (B) Autoencoder intermediate layer: 32-D activations from `RestaurantAutoencoder` |
| **Dimensionality reduction** | `sklearn.decomposition.PCA` projects 384-D or 32-D → 2-D for plotting |
| **Clustering on embeddings** | K-Means (`sklearn.cluster.KMeans`) applied after PCA to identify dense regions |
| **Geographic filter** | Filter by borough to see how well embeddings separate Manhattan from Brooklyn, etc. |
| **Implementation** | `app/pages/4_📊_PCA_Embedding_Explorer.py` |

**How PCA works here:** PCA finds the directions of maximum variance in the embedding space and projects each restaurant's vector onto the top-2 principal components. If the model has captured meaningful structure, restaurants with similar cuisine/vibe will cluster together even in 2-D. The PCA Explorer lets you switch between Transformer embeddings (capture semantic/language structure) and Autoencoder embeddings (capture operational/inspection structure) to compare what each model encodes.

**How the Autoencoder works:** `RestaurantAutoencoder` (`models/autoencoder.py`) compresses a 6-feature operational profile (health score, violations, rating, price, training hours, pest control) through the encoder path `6 → 64 → 32 → 2` using ReLU activations, and then reconstructs it through the decoder `2 → 32 → 64 → 6`. The network is trained with `Adam` optimizer and `MSELoss` for 80 epochs on ~1,200 restaurant profiles. The 32-D intermediate layer activations (not the 2-D bottleneck) are used as feature vectors for PCA, giving richer structural information than the compressed bottleneck alone.

---

### 5. Personalized Recommendations

After completing a short preference survey and liking restaurants during browsing, the app generates a ranked list of personalized restaurant recommendations.

| Component | Detail |
|---|---|
| **Scoring** | Weighted blend: preference score (from survey answers) + quality score (health grade + inspection score + Google rating + popularity) + lexical keyword match |
| **User profile** | Persisted in `data/user_profiles.json`; includes survey answers, liked restaurants, and cluster affinity |
| **Cluster affinity** | Compares user's liked restaurants to K-Means cluster centroids to infer preferred neighborhood / cuisine cluster |
| **Implementation** | `utils/recommendation_engine.py`, `utils/user_profile.py`, `app/pages/5_🔮_Recommendations.py` |

---

## Datasets

### 1. NYC DOHMH Restaurant Inspection Results (Primary)

| Property | Value |
|---|---|
| **Source** | NYC OpenData — NYC Department of Health and Mental Hygiene |
| **API Endpoint** | `https://data.cityofnewyork.us/resource/43nn-pn8j.csv` (SODA API, public) |
| **Default download** | 50,000 rows; configurable up to 200,000+ |
| **Granularity** | One row per inspection visit per violation code |
| **Key fields** | `camis` (restaurant ID), `dba` (name), `boro`, `cuisine_description`, `inspection_date`, `action`, `violation_code`, `critical_flag`, `score`, `grade`, `latitude`, `longitude` |
| **Preprocessing** | `data/preprocess.py` aggregates multiple inspection rows per restaurant into a single feature vector |

**Feature engineering steps in `data/preprocess.py`:**
- Aggregate by `camis`: compute `latest_score`, `avg_score`, `max_score`, `num_inspections`, `num_violations`, `critical_ratio`, `violations_per_inspection`
- One-hot encode `boro` (6 categories) and top-15 `cuisine_description` types (16 features including "Other")
- Apply `StandardScaler` to all 7 numerical features
- Stratified 80/20 train/test split
- Save to `data/train.csv`, `data/test.csv`, and `data/feature_config.json`

**Final feature vector:** 29 dimensions — 7 numerical (inspection statistics) + 6 borough dummies + 16 cuisine dummies.

### 2. Google Places Enrichment (Runtime)

| Property | Value |
|---|---|
| **Source** | Google Places API (Text Search + Place Details endpoints) |
| **Fields fetched** | `rating`, `user_ratings_total`, `price_level`, `editorial_summary`, `photos`, `url` |
| **Usage** | Enriches top search results and recommendation cards with real-time ratings and photos |
| **Caching** | Results cached in `data/cache/` to minimize API calls |
| **Requires** | `GOOGLE_API_KEY` environment variable (see Setup below) |

### 3. Synthetic / Curated Descriptions (NLP Corpus)

| Property | Value |
|---|---|
| **Size** | 220+ restaurant descriptions |
| **Usage** | Pre-embedded corpus for semantic search; used as fallback when DOHMH data is not downloaded |
| **Format** | Text descriptions with vibe tags embedded at app startup into 384-D vectors |

---

## ML Models & Algorithms In Depth

### Models Called (Pre-trained External)

| Model | Source | Purpose |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace Hub | 384-D semantic embeddings for restaurant descriptions and search queries |

### Models Built From Scratch (PyTorch / NumPy)

#### CustomMLP (`models/custom_mlp.py`)

```
Input (29) → FC → ReLU → Dropout(0.3) → FC(128) → ReLU → Dropout(0.3) → FC(3) → Softmax
```

- Full training loop: mini-batch SGD with AdamW, class-weighted cross-entropy, validation F1 early stopping (patience=10)
- Hyperparameter grid search: `hidden_dim ∈ {64, 128, 256}`, `lr ∈ {1e-3, 5e-4}`, `dropout ∈ {0.2, 0.3, 0.4}`
- Counterfactual search: gradient descent on the input vector (weights frozen) to minimize L2 perturbation while crossing the grade-A decision boundary
- Gradient-based feature importance + permutation importance for interpretability

#### RestaurantAutoencoder (`models/autoencoder.py`)

```
Encoder: 6 → FC(64) → ReLU → FC(32) → ReLU → FC(2)
Decoder: 2 → FC(32) → ReLU → FC(64) → ReLU → FC(6)
```

- Trained with `Adam` optimizer, `MSELoss`, 80 epochs
- `get_latent_space()` extracts the 32-D intermediate layer (not the 2-D bottleneck) for PCA input

#### KMeansScratch (`models/kmeans_scratch.py`)

- K-Means++ initialization: first centroid chosen randomly; each subsequent centroid chosen with probability proportional to squared distance from the nearest existing centroid
- E-step: vectorized Euclidean distance computation; each point assigned to nearest centroid
- M-step: recompute centroids as cluster mean
- Convergence: stop when max centroid shift < `tol` (default 1e-4) or `max_iter` reached
- Multi-start: run `n_init` times and retain run with lowest inertia
- Silhouette score computed post-fit for cluster quality assessment

### Algorithms Used (scikit-learn Wrappers, for Comparison or Utility)

| Algorithm | Library | Purpose |
|---|---|---|
| PCA | `sklearn.decomposition.PCA` | Project 384-D / 32-D embeddings to 2-D for plotting |
| KMeans | `sklearn.cluster.KMeans` | Density clustering in PCA Embedding Explorer |
| GaussianMixture | `sklearn.mixture.GaussianMixture` | Soft probabilistic clustering (comparison baseline in GIS map) |
| AgglomerativeClustering | `sklearn.cluster.AgglomerativeClustering` | Ward hierarchical clustering (comparison baseline) |
| t-SNE | `sklearn.manifold.TSNE` | Optional 2-D/3-D projection for embedding visualization |
| StandardScaler | `sklearn.preprocessing.StandardScaler` | Normalize inspection features before MLP training |

---

## Repository Structure

```
nyc-restaurant-survival-guide/
│
├── app/                                   # Streamlit frontend application
│   ├── main.py                            # Landing page, auth, home search
│   ├── ui_utils.py                        # Global Apple-inspired CSS theme
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py        # NLP vibe search (Transformers + cosine)
│       ├── 2_🧪_Health_Grade_Classifier.py # MLP sandbox with counterfactuals
│       ├── 3_📍_Restaurant_Cluster_GIS_Map.py # K-Means GIS + PyDeck 3D map
│       ├── 4_📊_PCA_Embedding_Explorer.py # Multi-model PCA + k-means density
│       └── 5_🔮_Recommendations.py        # Personalized restaurant recommendations
│
├── models/                                # ML model implementations (from scratch)
│   ├── __init__.py
│   ├── custom_mlp.py                      # 3-layer MLP: training, eval, counterfactuals
│   ├── autoencoder.py                     # Deep Autoencoder: 6D → 2D compression
│   ├── kmeans_scratch.py                  # K-Means++ in NumPy (no sklearn wrapper)
│   └── pca_scratch.py                     # PCA utilities
│
├── retrieval/                             # NLP retrieval system
│   ├── __init__.py
│   └── vector_search.py                   # HuggingFace embeddings + cosine similarity
│
├── utils/                                 # Helper modules
│   ├── __init__.py
│   ├── auth.py                            # User authentication
│   ├── clustering.py                      # Clustering pipeline orchestration
│   ├── data.py                            # Data loading utilities
│   ├── google_places.py                   # Google Places API integration
│   ├── recommendation_engine.py           # Personalized scoring logic
│   ├── search.py                          # Semantic search coordination
│   ├── search_assets.py                   # Runtime asset preparation
│   └── user_profile.py                    # User profile management & persistence
│
├── data/                                  # Data pipeline & storage
│   ├── __init__.py
│   ├── download_data.py                   # NYC DOHMH OpenData API fetcher
│   ├── preprocess.py                      # Feature engineering & train/test splits
│   ├── feature_config.json                # Feature schema & StandardScaler params
│   ├── user_profiles.json                 # Persisted user survey answers & likes
│   └── cache/                             # Runtime caches (embeddings, API responses)
│       ├── health_classifier_history.json
│       └── hp_search_results.json
│
├── tests/                                 # Pytest unit test suite
│   ├── __init__.py
│   ├── test_custom_mlp.py                 # MLP forward shape + training convergence
│   ├── test_autoencoder.py                # AE shape + loss reduction + latent output
│   └── test_semantic_search.py            # Embedding dimension + L2 norm
│
├── .streamlit/
│   └── config.toml                        # Streamlit theme configuration
├── .gitignore
├── Pipfile                                # Pipenv dependency specification
├── requirements.txt                       # pip dependency list
├── LICENSE                                # MIT License
└── README.md                              # This file
```

---

## Step-by-Step Setup

### Prerequisites

- Python 3.10 or higher
- `pip` (comes with Python) — or `pipenv` for the recommended workflow
- A terminal / command prompt

### Step 1 — Clone the repository

```bash
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide
```

### Step 2 — Install dependencies

**Option A: Pipenv (recommended)**

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install all project dependencies into an isolated virtual environment
pipenv install
```

**Option B: pip + venv**

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3 — (Optional) Get a Google Places API key

The app works without a Google Places API key — restaurant cards will show DOHMH data and descriptions only. To enable ratings, review counts, photos, and price tiers from Google:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project (or select an existing one)
3. Navigate to **APIs & Services → Library** and enable **Places API**
4. Navigate to **APIs & Services → Credentials → Create Credentials → API Key**
5. Copy your API key

Store it in one of two ways:

**Environment variable (works for both Pipenv and pip):**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

**Streamlit secrets file (persists across sessions):**
```bash
# Create the secrets file
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "your_api_key_here"' >> .streamlit/secrets.toml
```

### Step 4 — (Optional) Download real NYC restaurant inspection data

The app ships with a synthetic fallback corpus and works out of the box. To use the full real dataset from the NYC Department of Health:

```bash
# Download ~50,000 inspection records from NYC OpenData (no API key required — public data)
pipenv run python data/download_data.py

# Engineer features and create train/test splits
pipenv run python data/preprocess.py
```

This generates `data/train.csv`, `data/test.csv`, and updates `data/feature_config.json`. The Health Grade Classifier will automatically retrain on the real data on next launch.

> The NYC OpenData API is free and requires no account or API key. The download typically takes 15–30 seconds on a standard connection.

### Step 5 — Launch the app

```bash
# With Pipenv
pipenv run streamlit run app/main.py

# With pip venv (after activating)
streamlit run app/main.py
```

Open your browser to `http://localhost:8501`. The app will load, pre-compute embeddings, and be ready to use within a few seconds.

---

## Running Tests

```bash
pipenv run pytest tests/ -v
```

Expected: **10 tests, all passing** — covering the MLP (2 tests), Autoencoder (4 tests), and Semantic Search (4 tests).

---

## How to Use the Dashboard

1. **Home / Landing page** — Enter a natural language search query directly from the home screen. Create an account or log in to enable personalized recommendations and saved likes.

2. **Semantic Search (Page 1)** — Type any description of what you're looking for (cuisine, vibe, neighborhood, dietary preference). Results are ranked by cosine similarity of Transformer embeddings. Click a result card to see Google Places details.

3. **Health Grade Classifier (Page 2)** — Select a real NYC restaurant from the dropdown. Adjust the operational sliders (violation score, number of violations, pest control days, training hours). Watch the 3-class probability bar chart and radar chart update live. Click **"How to reach grade A"** to run the counterfactual engine.

4. **Restaurant Cluster GIS Map (Page 3)** — Use the sidebar to choose the number of clusters (k) and the clustering algorithm. The map automatically colors restaurants by cluster assignment. Hover for restaurant details. Use the color metric dropdown to visualize health grade, violations, or inspection score.

5. **PCA Embedding Explorer (Page 4)** — Choose between Transformer embeddings and Autoencoder embeddings. Apply a borough filter to see whether the embeddings separate geographic regions. Click **"Run k-means"** to overlay cluster boundaries on the PCA scatter plot.

6. **Recommendations (Page 5)** — Complete the short preference survey (cuisine type, budget, health priority, neighborhood). Like restaurants as you browse other pages. Return here to get a ranked personalized list with explanations of why each restaurant was recommended.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, PyDeck |
| **ML — from scratch** | PyTorch (CustomMLP, RestaurantAutoencoder, KMeansScratch) |
| **ML — pre-trained** | HuggingFace Transformers (`sentence-transformers/all-MiniLM-L6-v2`) |
| **ML — utilities** | scikit-learn (PCA, GMM, Ward, StandardScaler, t-SNE) |
| **Data** | Pandas, NumPy, NYC DOHMH OpenData API (SODA), Google Places API |
| **Testing** | Pytest |
| **Language** | Python 3.10+ |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
