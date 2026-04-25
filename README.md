# NYC Restaurant Survival Guide

**CSCI-UA 473 — Fundamentals of Machine Learning**
New York University · Spring 2026

An end-to-end interactive Machine Learning dashboard that helps New Yorkers find restaurants they will actually enjoy and trust. Real NYC DOHMH inspection data is fused with Google Places metadata, then put to work in five connected tools: semantic search, a health-grade classifier, a restaurant clustering explorer (GIS + 3D PCA), and a personalized recommendation engine.

Built with **PyTorch**, **Streamlit**, **Plotly**, **pydeck**, and **HuggingFace Transformers**.

---

## Authors

| Name | GitHub |
|---|---|
| Hollan Yuan | [hwyuanzi](https://github.com/hwyuanzi) |
| Ryan Han (PapTR) | [PapTR](https://github.com/PapTR) |
| Rahul Adusumalli | — |
| TBD | — |
| TBD | — |

---

## Table of Contents

1. [What the App Does](#what-the-app-does)
2. [Datasets](#datasets)
3. [ML Models & Algorithms In Depth](#ml-models--algorithms-in-depth)
4. [Repository Structure](#repository-structure)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [Running Tests](#running-tests)
7. [How to Use the Dashboard](#how-to-use-the-dashboard)
8. [Tech Stack](#tech-stack)
9. [License](#license)

---

## What the App Does

| Page | Purpose |
|---|---|
| 🔍 **Semantic Search** | Type what you want ("cozy Italian pasta spot in Brooklyn") and get ranked restaurant matches from a prepared dataset of ~2,835 real NYC restaurants enriched with Google Places data. |
| 🧪 **Health Grade Risk Classifier** | Pick a held-out NYC restaurant and estimate whether its inspection profile looks more like Grade A, B, or C restaurants. The page shows probabilities as risk signals, compares the MLP to an always-A baseline, and explains why score-derived leakage features were removed. |
| 📍 **Restaurant Cluster GIS Map** | Cluster the ~2,835 restaurants with three comparable algorithms (K-Means from scratch, Gaussian Mixture, Ward) in an 18-dim interpretable feature space (price / rating / review volume / health / cuisine group / borough / geo-location). Restaurants are colored by cluster on a real NYC map with persona labels + narrative stories. |
| 📊 **PCA Embedding Explorer** | Projects the same clusters into 3-D PCA space with feature-loading bar charts, cluster-distance heatmap, and prototype restaurants nearest each centroid. |
| 🔮 **Personalized Recommendations** | Given your saved profile and liked restaurants, ranks candidates by per-liked cosine KNN fused via Reciprocal Rank Fusion, re-ranked with Maximal Marginal Relevance for cuisine diversity, and a cuisine-alignment multiplier so users see their preferred cuisines first. |

---

## Datasets

### 1. NYC DOHMH Restaurant Inspection Results (Primary)

| Property | Value |
|---|---|
| **Source** | NYC OpenData — NYC Department of Health and Mental Hygiene |
| **API Endpoint** | `https://data.cityofnewyork.us/resource/43nn-pn8j.csv` (SODA API, free public access, no key required) |
| **Default download** | 50,000 rows; configurable up to 200,000+ |
| **Granularity** | One row per inspection visit per violation code |
| **Key fields** | `camis` (restaurant ID), `dba` (name), `boro`, `cuisine_description`, `inspection_date`, `action`, `violation_code`, `critical_flag`, `score`, `grade`, `latitude`, `longitude` |

**Feature engineering steps (in `data/preprocess.py`):**
- Aggregate by `camis`: compute `latest_score`, `avg_score`, `max_score`, `num_inspections`, `num_violations`, `critical_ratio`, `violations_per_inspection`
- One-hot encode `boro` (6 categories) and top-15 `cuisine_description` types (16 features including "Other")
- Apply `StandardScaler` to all 7 numerical features
- Stratified 80/20 train/test split → `data/train.csv`, `data/test.csv`, `data/meta_train.csv`, `data/meta_test.csv`
- Save scaler parameters and column order to `data/feature_config.json`

**Final MLP feature vector:** 29 dimensions — 7 numerical inspection statistics + 6 borough dummies + 16 cuisine dummies.

### 2. Google Places Enrichment

| Property | Value |
|---|---|
| **Source** | Google Places API (Text Search + Place Details endpoints) |
| **Fields fetched** | `rating`, `user_ratings_total`, `price_level`, `editorial_summary`, `photos`, `url` |
| **Pre-built cache** | `data/cache/prepared_search_v4_3800.pkl` ships with the repo — ~2,835 restaurants already enriched |
| **Usage** | Powers semantic search ranking, restaurant cards, and the clustering feature matrix |
| **Requires live key** | Only needed when refreshing the cache; normal usage reads the committed pickle |

### 3. Sentence Embeddings Cache

| Property | Value |
|---|---|
| **File** | `data/cache/embeddings_prepared_v4_*.npy` |
| **Model** | `sentence-transformers/all-mpnet-base-v2` |
| **Dimensions** | 768-D per restaurant |
| **Usage** | Pre-computed at first run; reused on every subsequent launch |

---

## ML Models & Algorithms In Depth

### Models Called (Pre-trained External)

| Model | Source | Dimensions | Purpose |
|---|---|---|---|
| `sentence-transformers/all-mpnet-base-v2` | HuggingFace Hub | 768-D | Semantic embeddings for restaurant descriptions and search queries |

**How it works:** The model maps both the user's query and each restaurant description into the same 768-dimensional semantic vector space using a Transformer encoder with mean pooling. After L2 normalization, cosine similarity equals the dot product, so ranking is a single matrix multiply (`torch.mm()`) followed by `torch.topk()`. This means queries like *"upscale sushi"* and *"expensive Japanese omakase"* surface the same restaurants without any shared keywords.

---

### Models Built From Scratch

#### CustomMLP — Health Grade Classifier (`models/custom_mlp.py`)

**Architecture:**
```
Input (25-D)
  → FC(25 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)
  → Softmax (3 classes: A, B, C)
```

**Training details:**
- Dataset: ~14,000 real DOHMH restaurants, 25-feature vectors, stratified 80/20 split
- Inputs: violation history counts/rates, borough one-hot features, and top-cuisine one-hot features
- Leakage control: score-derived fields such as `latest_score`, `avg_score`, `max_score`, and `critical_ratio` are excluded because DOHMH letter grades are derived from inspection-score thresholds
- Optimizer: AdamW
- Loss: Class-weighted `CrossEntropyLoss` — compensates for heavily imbalanced label distribution (A ≈ 78%, B ≈ 16%, C ≈ 6%)
- Early stopping: monitors validation F1 (patience = 10 epochs)
- Hyperparameter grid search over `hidden_dim ∈ {64, 128, 256}`, `lr ∈ {1e-3, 5e-4}`, `dropout ∈ {0.2, 0.3, 0.4}`; results cached in `data/cache/hp_search_results.json`
- Trained checkpoint persisted at `data/cache/health_classifier.pt`

**From-scratch requirement:** `models/custom_mlp.py` implements the model class, training loop (mini-batch gradient descent, forward/backward, loss computation), evaluation (confusion matrix, per-class F1/precision/recall), gradient-based feature importance, permutation importance, and a counterfactual adversarial search engine — no scikit-learn wrappers used.

**Classifier explainability UI:** Given a selected held-out restaurant, the Streamlit page exposes integer sliders for total violations and inspection count, derives violations-per-inspection from those counts, provides context selectors for borough/cuisine, compares before-vs-after class probabilities, ranks local feature sensitivity, runs a constrained "Path to A" search, and shows a PCA context map using the project's NumPy PCA implementation. Improvement advice is limited to actionable violation-pattern features; PCA is used only to explain feature-space geometry.

---

#### KMeansScratch — Restaurant Clustering (`models/kmeans_scratch.py`)

**Algorithm step by step:**

1. **K-Means++ initialization:** First centroid chosen uniformly at random. Each successive centroid is chosen from remaining points with probability proportional to its squared distance from the nearest already-chosen centroid. This spread-out initialization dramatically reduces the chance of poor local minima compared to random seeding.

2. **E-step (Assignment):** For each restaurant, compute Euclidean distance to all centroids and assign it to the nearest one. Fully vectorized with NumPy broadcasting.

3. **M-step (Update):** Recompute each centroid as the mean of all points assigned to it. Micro-clusters below 1.5% of the dataset are merged into their nearest larger neighbor to avoid degenerate singletons while preserving meaningful smaller groups.

4. **Convergence:** Iterate until the maximum centroid shift falls below `tol = 1e-4` or `max_iter` is reached.

5. **Multi-start:** Run `n_init` times with different seeds; retain the run with lowest inertia (sum of squared distances to assigned centroids).

6. **Quality / K selection:** The app sweeps `K=4..15` and records silhouette, inertia, and largest-cluster share. It chooses a conservative K from the silhouette knee and K-Means inertia elbow while skipping catch-all candidates where one cluster exceeds 35% of the dataset.

**Feature space (18-D):** standardized price tier, Google rating, log(review volume), DOHMH health score, latitude, longitude + cuisine-group one-hot (American, Asian, Latin, Cafe, Italian, European, Other = 7 features) + borough one-hot (5 features). Raw cuisine strings are mapped into these broader groups before clustering, which reduces the old "Other" catch-all from roughly 43% to about 7% and prevents one giant mixed cluster from absorbing the dataset.

**Cluster labeling:** K-Means only assigns numeric `cluster_id`s. The app names each cluster afterward from summary statistics in the same grouped feature space: a dominant cuisine group leads the label when present; otherwise borough concentration, price tier, and rating-review persona explain what binds the cluster. This is why labels such as `American`, `Asian`, `Cafe`, `European · Mid-Range`, `Staten Island · Mid-Range`, and `Mid-Range · Highly Rated` are interpretations of the learned clusters, not manual assignments.

**Current default result:** On the prepared ~2,835-restaurant sample, the K-selection sweep chooses requested `K = 9`; one micro-cluster is merged by the 1.5% stability rule, leaving 8 displayed clusters. The largest cluster is about 27% of restaurants, below the 35% catch-all threshold.

---

#### RestaurantAutoencoder — Dimensionality Reduction (`models/autoencoder.py`)

**Architecture:**
```
Encoder: FC(6 → 64) → ReLU → FC(64 → 32) → ReLU → FC(32 → 2)
Decoder: FC(2 → 32) → ReLU → FC(32 → 64) → ReLU → FC(64 → 6)
```

- Trained with Adam optimizer and MSELoss for 80 epochs on ~1,200 restaurant operational profiles (6 features: health score, violations, rating, price, training hours, pest control days)
- `get_latent_space()` extracts 32-D intermediate layer activations (not the 2-D bottleneck) for use as richer feature vectors in PCA

> **Note:** The Autoencoder is retained for coursework completeness and the PCA Explorer page. It is not part of the main search or recommendation pipeline.

---

#### PCA — Cluster Projection (`models/pca_scratch.py`)

Applied in the PCA Embedding Explorer to project the 18-D clustering feature space into 3 principal components for visualization. Feature loading bar charts show which original features (price, rating, health score, cuisine group, borough, etc.) explain the most variance per principal component, giving interpretable axes to the scatter plot.

---

### Comparison Baselines (scikit-learn)

| Algorithm | Library | Purpose |
|---|---|---|
| Gaussian Mixture Model | `sklearn.mixture.GaussianMixture` | Soft probabilistic clustering; allows ellipsoidal cluster shapes |
| Ward Hierarchical Clustering | `sklearn.cluster.AgglomerativeClustering` | Merges clusters by minimizing within-cluster variance |
| StandardScaler | `sklearn.preprocessing.StandardScaler` | Normalizes 29-D inspection features before MLP training |
| Silhouette Score | `sklearn.metrics.silhouette_score` | Cluster quality evaluation for K-Means and baselines |

---

### Recommendations — KNN + RRF + MMR + Cuisine Alignment (`utils/clustering.py`)

**How it works:**

1. **Per-liked KNN retrieval:** For each restaurant the user has liked, compute cosine similarity to all candidates in the 18-D feature space. This produces one ranked list per liked restaurant.

2. **Reciprocal Rank Fusion (RRF):** The per-liked ranked lists are merged using RRF (Cormack et al. 2009): each candidate's final score is the sum of `1 / (k + rank_i)` across all lists. A small profile-similarity bias term derived from the user's survey answers is added.

3. **Maximal Marginal Relevance (MMR) re-ranking:** The top-50 candidates from RRF are re-ranked to maximize relevance while penalizing similarity to already-selected items, with tunable λ. This prevents the final list from being dominated by one cuisine type.

4. **Cuisine alignment multiplier:** `cuisine_alignment_score()` applies a multiplicative boost: explicit `favorite_cuisines` matches get full weight; mismatches are reduced to 15% (explicit) or 30% (inferred from likes history). Price, rating, and location remain as tiebreakers within the preferred cuisine.

---

## Repository Structure

```
nyc-restaurant-survival-guide/
│
├── app/                                   # Streamlit frontend application
│   ├── main.py                            # Landing page, auth, home search
│   ├── ui_utils.py                        # Global Apple-inspired CSS theme
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py        # NLP search (Transformers + cosine)
│       ├── 2_🧪_Health_Grade_Classifier.py # MLP classifier + what-if explainer
│       ├── 3_📍_Restaurant_Cluster_Map.py  # K-Means GIS + PyDeck 3D map
│       ├── 4_📊_PCA_Embedding_Explorer.py  # 3-D PCA + cluster loadings
│       └── 5_🔮_Recommendations.py        # KNN + RRF + MMR recommendations
│
├── models/                                # ML model implementations
│   ├── __init__.py
│   ├── custom_mlp.py                      # MLP: training, eval, counterfactuals (from scratch)
│   ├── autoencoder.py                     # Deep Autoencoder: 6D → 2D (from scratch)
│   ├── kmeans_scratch.py                  # K-Means++ in NumPy (from scratch)
│   └── pca_scratch.py                     # PCA utilities (from scratch)
│
├── retrieval/                             # NLP retrieval system
│   ├── __init__.py
│   └── vector_search.py                   # HuggingFace embeddings + cosine similarity
│
├── utils/                                 # App-layer helper modules
│   ├── __init__.py
│   ├── auth.py                            # Lightweight username/password auth
│   ├── clustering.py                      # Feature matrix, clustering pipeline,
│   │                                      #   cluster personas, KNN + MMR + RRF
│   ├── data.py                            # DOHMH API fetcher (with caching)
│   ├── google_places.py                   # Google Places API integration
│   ├── recommendation_engine.py           # Personalized scoring logic
│   ├── search.py                          # Embedding cache + semantic ranking
│   ├── search_assets.py                   # Cache-first prepared-data loader
│   └── user_profile.py                    # User profile CRUD + history → features
│
├── data/                                  # Data pipeline & storage
│   ├── __init__.py
│   ├── download_data.py                   # NYC DOHMH OpenData API fetcher
│   ├── preprocess.py                      # Feature engineering, train/test splits
│   ├── train.csv / test.csv               # MLP training data (29-D features)
│   ├── meta_train.csv / meta_test.csv     # Restaurant metadata for UI
│   ├── feature_config.json                # Scaler params + column order
│   ├── user_profiles.json                 # Persisted user survey answers & likes
│   └── cache/
│       ├── prepared_search_v4_3800.pkl    # 2,835 Google-enriched restaurants
│       ├── embeddings_prepared_v4_*.npy   # Pre-computed 768-D embeddings
│       ├── health_classifier.pt           # Trained MLP checkpoint
│       ├── health_classifier_history.json # Training loss/F1 curves
│       └── hp_search_results.json         # Hyperparameter grid search log
│
├── tests/                                 # Pytest unit test suite
│   ├── __init__.py
│   ├── test_custom_mlp.py                 # MLP forward shape + training convergence
│   ├── test_autoencoder.py                # AE shape + loss reduction + latent output
│   └── test_semantic_search.py            # Embedding dimension + L2 norm + relevance
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

- Python 3.10 or higher (3.11 recommended)
- `pip` — or `pipenv` for the recommended workflow

### Step 1 — Clone the repository

### Option A — pipenv (recommended)
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

> **Compatibility note:** The project requires `numpy >= 2.0` and `pandas >= 2.2` for compatibility with the committed pickle cache. Both are pinned in `requirements.txt`.

### Step 3 — (Optional) Get a Google Places API key

The app works out of the box without any API key — the repo ships with `data/cache/prepared_search_v4_3800.pkl` containing ~2,835 restaurants already enriched with Google Places data, plus pre-computed embeddings. You only need an API key if you want to refresh the cache with new restaurant data.

**To get a Google Places API key:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Navigate to **APIs & Services → Library** and enable **Places API**
4. Navigate to **APIs & Services → Credentials → Create Credentials → API Key**
5. Copy your new API key

**Store it in one of two ways:**

### Optional — rebuild the dataset from scratch
```bash
# Environment variable (temporary — works for the current shell session)
export GOOGLE_API_KEY="your_api_key_here"
```

```toml
# .streamlit/secrets.toml (persistent across sessions — recommended)
GOOGLE_API_KEY = "your_api_key_here"
```

### Step 4 — (Optional) Download fresh NYC inspection data

The app runs immediately with the committed cache. To rebuild from the live NYC OpenData API:

```bash
# Download ~50,000 inspection records (no API key required — public data)
pipenv run python data/download_data.py

# Engineer features and create train/test splits
pipenv run python data/preprocess.py
# To rebuild the Google-enriched cache you need a Places API key
# in .streamlit/secrets.toml (copy from .streamlit/secrets.toml.example)
```

This generates `data/train.csv`, `data/test.csv`, `data/meta_train.csv`, `data/meta_test.csv`, and updates `data/feature_config.json`. The Health Grade Classifier will automatically retrain on the new data the next time it is loaded.

> The NYC OpenData SODA API is free, requires no account or API key, and the download typically takes 15–30 seconds.

### Step 5 — Launch the app

```bash
# With Pipenv
pipenv run streamlit run app/main.py

# With pip venv (after activating)
streamlit run app/main.py
```

Open your browser to `http://localhost:8501`. The app pre-computes embeddings on first run (about 10–20 seconds) and caches them for every subsequent launch.

---

## Running Tests

```bash
pipenv run pytest tests/ -v
```

Expected: **10 tests, all passing** — covering the MLP (2 tests), Autoencoder (4 tests), and Semantic Search (4 tests).

---

## How to Use the Dashboard

1. **Landing page** — Log in or sign up. Profiles persist in `data/user_profiles.json` and enable personalized recommendations and saved likes.

2. **Semantic Search (Page 1)** — Type a natural-language craving or occasion (cuisine, vibe, neighborhood, dietary preference). Results are ranked by cosine similarity of Transformer embeddings. Filter by borough, health grade, and minimum Google rating in the sidebar. Click the heart icon to save restaurants to your profile.

3. **Health Grade Risk Classifier (Page 2)** — Search the held-out test set and click a restaurant row to see its estimated A/B/C risk category, 3-class probability bar, and the true DOHMH grade for comparison. The Model Performance panel shows a majority-class baseline, full confusion matrix, and per-class F1, precision, and recall.

4. **Restaurant Cluster GIS Map (Page 3)** — Choose a clustering algorithm (K-Means / GMM / Ward) and number of clusters in the sidebar. Restaurants appear as 3D columns on a real NYC map, colored by cluster persona. Scroll down for per-cluster summary cards showing cuisine mix, borough distribution, average rating and price, and a narrative explaining why the cluster exists.

5. **PCA Embedding Explorer (Page 4)** — View the same clusters projected into 3-D PCA space. Inspect the feature-loading bar chart per principal component to understand what each axis represents (e.g., PC1 = price/rating axis, PC2 = geographic axis). The cluster-distance heatmap and prototype restaurants nearest each centroid help identify cluster relationships.

6. **Recommendations (Page 5)** — Set your budget, favorite cuisines, and preferred boroughs in the sidebar, or simply like restaurants while browsing. The top-15 list respects your cuisine preferences (via the alignment multiplier) and shows which liked restaurant influenced each recommendation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, pydeck |
| **ML — from scratch** | PyTorch (CustomMLP, RestaurantAutoencoder), NumPy (KMeansScratch, PCA) |
| **ML — pre-trained** | HuggingFace Transformers — `sentence-transformers/all-mpnet-base-v2` |
| **ML — baselines / utilities** | scikit-learn (GMM, Ward, StandardScaler, silhouette score) |
| **Data** | Pandas, NumPy, NYC DOHMH OpenData API (SODA), Google Places API |
| **Testing** | Pytest |
| **Language** | Python 3.10+ |

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
