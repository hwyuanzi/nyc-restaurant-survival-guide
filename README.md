# NYC Restaurant Survival Guide

**CSCI-UA 473 - Fundamentals of Machine Learning**

New York University, Spring 2026

NYC Restaurant Survival Guide is a Streamlit machine-learning app for exploring New York City restaurants. It combines real NYC Department of Health and Mental Hygiene inspection data, cached Google Places metadata, semantic search, a health-grade risk classifier, K-Means clustering from scratch, PCA-based cluster visualization, and personalized recommendations from saved liked restaurants.

The repository is set up to run locally without live downloads. The prepared restaurant table, embedding matrix, classifier checkpoint, and clustering caches are committed so the demo can start from the submitted files. Restaurant photos use the Google Places Photo API and only display when you provide a local `GOOGLE_API_KEY`.

---

## Authors

| Name | GitHub |
|---|---|
| Hollan Yuan | [hwyuanzi](https://github.com/hwyuanzi) |
| Ryan Han | [PapTR](https://github.com/PapTR) |
| Rahul Adusumalli | [Rahuman-Noodles](https://github.com/Rahuman-Noodles) |
| Muqiao Tao | [taomuqiao](https://github.com/taomuqiao) |
| Jaiden Xu | [jbx202](https://github.com/jbx202) |

---

## Project Checklist

| Requirement area | How this repository addresses it |
|---|---|
| Working app | `app/Main.py` is the Streamlit entry point. The app has five navigable pages and loads from committed caches by default. |
| Real dataset and meaningful task | The project uses NYC DOHMH inspection data plus Google Places metadata to answer user-facing questions about restaurant discovery, health-risk signals, restaurant segments, and liked-history recommendations. |
| Course algorithm implementation | `models/kmeans_scratch.py` implements K-Means++ directly in NumPy, including initialization, assignment, centroid updates, empty-cluster handling, convergence, and multi-start model selection. |
| ML coherence | Semantic retrieval, health-grade classification, clustering, PCA visualization, K selection, and recommendation reranking all operate on documented data representations. |
| Usability | Pages include labeled controls, constrained filters, cached runtime assets, error messages for missing data, and a login/profile flow for saving likes. |
| Repository hygiene | Active code is under `app/`, `models/`, `utils/`, `data/`, and `tests/`. Obsolete checkpoint-only modules and old cache files have been removed. |

---

## Install And Run

### 1. Clone The Repository

```bash
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide
```

### 2. Create A Python Environment

Python 3.11+ is recommended. The project has also been smoke-tested in the local Python 3.14 environment used during development.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start The App

```bash
streamlit run app/Main.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

### 4. Optional: Enable Restaurant Photos

Search results still work without a Google key, but restaurant photos require the Google Places Photo API. Add a key in one of these two ways:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

Or set the key as an environment variable before starting Streamlit:

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
streamlit run app/Main.py
```

The real `.streamlit/secrets.toml` file is intentionally ignored by Git.

### 5. Optional Pipenv Workflow

```bash
pip install pipenv
pipenv install
pipenv run streamlit run app/Main.py
```

---

## Step-By-Step App Use

1. **Log in or create an account.** User accounts are stored locally in `data/user_profiles.local.json`; no external authentication service is used. This runtime file is ignored by Git so personal profiles, liked restaurants, password hashes, and salts are not committed.
2. **Start on the Home page.** Try one of the suggested restaurant queries, review the result cards, and click "Like this restaurant" on places you would actually want.
3. **Open Semantic Search.** Use natural language such as `cozy Italian pasta in Brooklyn`, `late night ramen Manhattan`, or `cheap Caribbean food Bronx`. The page uses cached sentence embeddings when available and falls back gracefully if the embedding model cannot load.
4. **Open Health Grade Risk Classifier.** Select a held-out restaurant, inspect predicted A/B/C risk probabilities, change inspection-pattern inputs, and review feature importance plus the constrained "Path to A" analysis.
5. **Open Restaurant Cluster GIS Map.** View restaurants colored by learned cluster on an NYC map. The default clustering path uses the NumPy K-Means++ implementation; GMM and Ward are included as comparison baselines.
6. **Open PCA Embedding Explorer.** Inspect the same clusters in 3D PCA, centroid-distance view, or t-SNE; use the feature loading and prototype panels to explain what separates clusters.
7. **Open Personalized Recommendations.** Add or remove liked restaurants in the sidebar. Recommendations are generated from liked-history nearest neighbors, RRF fusion, and MMR diversity reranking. The cluster visualization on this page is explanatory context, not a "you belong to this cluster" rule.

---

## App Pages

| Page | What it does | Main ML concept |
|---|---|---|
| Home / Landing Search | Login/signup, cached restaurant search, like/unlike restaurants. | Embedding search + profile persistence |
| Semantic Search | Natural-language restaurant search with cuisine, location, price, and quality guardrails. | Transformer embeddings + cosine similarity |
| Health Grade Risk Classifier | Held-out DOHMH restaurant classification, feature editing, model diagnostics, and path-to-A search. | Custom PyTorch MLP + class-weighted cross entropy |
| Restaurant Cluster GIS Map | Cluster restaurants on an 18-D feature space and view clusters on an NYC map. | K-Means++ from scratch; GMM/Ward baselines |
| PCA Embedding Explorer | 3D PCA, centroid-distance space, t-SNE, feature loadings, distances, summaries, and prototypes. | PCA visualization + cluster interpretation |
| Recommendations | Personalized restaurant picks from explicit likes. | Per-liked KNN + Reciprocal Rank Fusion + MMR |

---

## Data And Cache Files

### DOHMH Classifier Data

The classifier data comes from NYC OpenData's DOHMH restaurant inspection dataset:

```text
https://data.cityofnewyork.us/resource/43nn-pn8j.csv
```

`data/preprocess.py` aggregates raw inspection-violation rows into one row per restaurant, keeps the latest grade as the label, engineers features, and writes:

| File | Shape | Purpose |
|---|---:|---|
| `data/train.csv` | `11,401 x 26` | Classifier training rows and target |
| `data/test.csv` | `2,851 x 26` | Held-out classifier test rows and target |
| `data/meta_train.csv` | `11,401 x 8` | Restaurant metadata for train rows |
| `data/meta_test.csv` | `2,851 x 8` | Restaurant metadata for held-out UI selection |
| `data/feature_config.json` | config | Feature names, label mapping, scaler statistics |

The health classifier uses 25 input features:

- `num_inspections`, `num_violations`, `violations_per_inspection`
- borough one-hot features
- top cuisine one-hot features plus `cuisine_Other`

Score-derived columns such as `latest_score`, `avg_score`, `max_score`, and `critical_ratio` are intentionally excluded because DOHMH grades are derived from inspection score thresholds. Keeping them would leak the label.

### Prepared Search And Demo Cache

These files are intentionally committed for a reliable local demo:

| File | Shape / size | Purpose |
|---|---:|---|
| `data/cache/prepared_search_v4_3800.pkl` | `2,835 x 24` | Main Google-enriched restaurant table used by search, clustering, PCA, and recommendations |
| `data/cache/embeddings_prepared_v4_3800_2835.npy` | `2,835 x 768` | Cached `all-mpnet-base-v2` restaurant embeddings |
| `data/cache/enriched_restaurants_3800.pkl` | `3,401 x 22` | Intermediate Google Places enrichment cache |
| `data/cache/health_classifier.pt` | checkpoint | Trained PyTorch classifier weights |
| `data/cache/health_classifier_history.json` | history | Training and validation loss/F1 history |
| `data/cache/health_classifier_importance.json` | importance | Cached permutation-importance output |
| `data/cluster_cache.parquet` | `2,835 x 48` | K-Means clustered restaurant table |
| `data/kmeans_model.joblib` | model cache | K-Means model, scaler, and PCA artifacts |
| `data/cluster_cache_gmm.parquet` | `2,835 x 48` | GMM baseline clustered table |
| `data/cluster_model_gmm.joblib` | model cache | GMM baseline artifacts |
| `data/cluster_cache_agglo.parquet` | `2,835 x 48` | Ward/agglomerative baseline clustered table |
| `data/cluster_model_agglo.joblib` | model cache | Ward/agglomerative baseline artifacts |

Earlier experimental caches, including old v3 prepared search files, partial v4 embedding files, and hyperparameter-search JSON output, are not needed for the final app and have been removed from the active repository state.

### Dataset Size Choice

The prepared search sample starts from `3,800` candidate restaurants and keeps `2,835` restaurants after Google enrichment and validity filters. A larger prepared dataset could improve search and recommendation coverage, especially for rare cuisines and neighborhoods. For the submitted project, the cache size is deliberately moderate so the repository stays lightweight, starts quickly on a local laptop, and still lets users rebuild a larger local cache from NYC DOHMH plus Google Places if they want more coverage.

### Local Profile Storage

Runtime accounts are written to `data/user_profiles.local.json`, which is ignored by Git. The committed `data/user_profiles.json` and `data/user_profiles.example.json` are empty placeholders only. New profiles store only account metadata, password hash/salt, and `likes`. The recommendation page intentionally ranks from liked restaurants only, so cuisine, borough, budget, spice, and vibe preference fields are not created by default.

---

## Models And Algorithms

### Semantic Search

Implemented in `utils/search.py` and used by `app/Main.py` plus `app/pages/1_🔍_Semantic_Search.py`.

- Restaurant descriptions combine name, cuisine, borough/neighborhood, address, Google summary, rating, price tier, and health-grade information.
- Cached embeddings use `sentence-transformers/all-mpnet-base-v2`.
- Query and restaurant vectors are L2-normalized, so cosine similarity is computed as a dot product.
- Structured guardrails for cuisine, borough/neighborhood, price, and quality keep explicit user intent from being overwhelmed by generic semantic matches.
- If the embedding model is unavailable, the search code falls back to lexical and structured scoring instead of crashing.

### Health Grade Risk Classifier

Implemented in `models/custom_mlp.py` and `app/pages/2_🧪_Health_Grade_Classifier.py`.

```text
Input(25)
  -> Linear(25, 128) -> ReLU -> Dropout(0.3)
  -> Linear(128, 128) -> ReLU -> Dropout(0.3)
  -> Linear(128, 3)
```

Training details:

- PyTorch model and training loop implemented directly with `torch.nn`, `DataLoader`, AdamW, and class-weighted cross entropy.
- Stratified validation split is taken from the training data.
- Early stopping monitors validation weighted F1.
- The page reports held-out accuracy/F1, majority baseline, confusion matrix, per-class metrics, permutation importance, local sensitivity, and path-to-A feature edits.

Current held-out metrics from the committed checkpoint:

| Metric | Value |
|---|---:|
| Accuracy | `70.4%` |
| Weighted F1 | `0.708` |
| Macro F1 | `0.394` |

The classifier is presented as an inspection-risk signal, not an official future-grade forecast.

### K-Means++ From Scratch

Implemented in `models/kmeans_scratch.py` and used as the default clustering algorithm in `utils/clustering.py`.

This is the primary non-wrapper course algorithm implementation. It is not scikit-learn K-Means.

The implementation includes:

1. K-Means++ initialization.
2. Vectorized Euclidean assignment.
3. Centroid update by cluster means.
4. Empty-cluster reinitialization.
5. Convergence by unchanged labels or centroid shift tolerance.
6. Multi-start `n_init` with lowest-inertia model selection.

The clustering feature matrix has 18 interpretable dimensions:

- price tier, Google rating, log review count, inverted DOHMH score, latitude, longitude
- cuisine group one-hot features: American, Asian, Latin, Cafe, Italian, European, Other
- borough one-hot features: Manhattan, Brooklyn, Queens, Bronx, Staten Island

Features are standardized before K-Means, so Euclidean distance is more meaningful across mixed scales. GMM and Ward/agglomerative clustering are included only as comparison baselines on the same feature matrix.

### PCA And Cluster Interpretation

Implemented in `utils/clustering.py` and `app/pages/4_📊_PCA_Embedding_Explorer.py`.

PCA is used for visualization and explanation; the scratch-implementation requirement is satisfied by `models/kmeans_scratch.py`. The app shows:

- 3D PCA projection of the standardized clustering feature matrix
- centroid-distance PCA view for cleaner cluster separation
- optional t-SNE visualization
- component loadings, explained variance, cluster distance matrix, cluster summaries, and prototype restaurants

### Personalized Recommendations

Implemented in `app/pages/5_🔮_Recommendations.py` and helper functions in `utils/clustering.py`.

Recommendation uses explicit liked restaurants only:

1. Each saved like is one positive example.
2. The app retrieves cosine nearest neighbors for each liked restaurant in the 18-D restaurant feature space.
3. Exact cuisine signals from liked history are applied before final ranking. For example, if most saved likes are Korean, Korean candidates are boosted above nearby Asian-family cuisines such as Chinese, Thai, or Japanese.
4. Per-liked ranked lists are combined with Reciprocal Rank Fusion.
5. Maximal Marginal Relevance reranks the candidates to balance relevance and diversity after the liked-cuisine signal has been applied.

The recommendation algorithm is independent of cluster labels. The cluster view on the Recommendation page explains where liked restaurants and top picks sit in restaurant feature space; it does not assign the user to a cluster.

---

## Repository Structure

```text
nyc-restaurant-survival-guide/
├── app/
│   ├── Main.py
│   ├── ui_utils.py
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py
│       ├── 2_🧪_Health_Grade_Classifier.py
│       ├── 3_📍_Restaurant_Cluster_Map.py
│       ├── 4_📊_PCA_Embedding_Explorer.py
│       └── 5_🔮_Recommendations.py
├── data/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.csv / test.csv
│   ├── meta_train.csv / meta_test.csv
│   ├── feature_config.json
│   ├── cluster caches and model caches
│   └── cache/
│       ├── prepared_search_v4_3800.pkl
│       ├── embeddings_prepared_v4_3800_2835.npy
│       ├── enriched_restaurants_3800.pkl
│       └── health classifier artifacts
├── models/
│   ├── custom_mlp.py
│   └── kmeans_scratch.py
├── tests/
│   ├── test_custom_mlp.py
│   ├── test_kmeans_scratch.py
│   └── test_semantic_search.py
├── utils/
│   ├── auth.py
│   ├── clustering.py
│   ├── data.py
│   ├── google_places.py
│   ├── search.py
│   ├── search_assets.py
│   └── user_profile.py
├── requirements.txt
├── Pipfile
└── README.md
```

---

## Rebuild Or Expand The Data

The submitted app does not expose a Streamlit "refresh data" button. Rebuilding is a terminal workflow.

### Rebuild DOHMH Classifier Splits

```bash
source .venv/bin/activate
python data/download_data.py 50000
python data/preprocess.py
```

This rewrites `data/train.csv`, `data/test.csv`, `data/meta_train.csv`, `data/meta_test.csv`, and `data/feature_config.json`. If the classifier checkpoint is deleted, the Health Grade Risk Classifier page can retrain and save a new `data/cache/health_classifier.pt` from those files.

### Rebuild Google-Enriched Search Cache

Set a Google Places key either in the environment or in `.streamlit/secrets.toml`:

```bash
export GOOGLE_API_KEY="your_key_here"
```

For `.streamlit/secrets.toml`, copy the example file and fill in your key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
GOOGLE_API_KEY = "your_key_here"
```

Then run:

```bash
python -c "from utils.search_assets import load_prepared_search_assets, DEFAULT_SEARCH_SAMPLE_SIZE; load_prepared_search_assets(DEFAULT_SEARCH_SAMPLE_SIZE, force_refresh=True)"
```

To try a larger local prepared cache, edit `DEFAULT_SEARCH_SAMPLE_SIZE` in `utils/search_assets.py` or call `load_prepared_search_assets(sample_size=YOUR_SIZE, force_refresh=True)`. Larger samples improve coverage but require more Google API calls, more embedding time, and larger cache files.

---

## Tests And Smoke Checks

```bash
pipenv run pytest tests/ -v
python3 -m py_compile app/Main.py app/pages/*.py utils/*.py models/*.py
```

Current test coverage checks:

- K-Means++ fit, predict, and distance-transform behavior
- Custom MLP forward pass and training loop behavior
- Semantic search fallback and borough-filter behavior without downloading a model

During repository review, the Streamlit pages were also smoke-tested with `streamlit.testing.v1.AppTest`; all app entry points loaded with zero page exceptions.

---

## Common Questions

**Which part is implemented from scratch?**

The K-Means++ algorithm in `models/kmeans_scratch.py` is implemented directly in NumPy and is the default clustering engine shown in the app.

**Why use Euclidean distance for K-Means?**

The restaurant feature matrix is standardized before clustering, so price, rating, review count, health score, cuisine indicators, borough indicators, and location features are put on comparable scales.

**Are cluster labels learned directly?**

No. K-Means learns numeric cluster IDs. Human-readable labels are generated afterward from cluster summary statistics such as cuisine group, borough concentration, price, rating, and review volume.

**Is recommendation based on the user's cluster?**

No. Recommendations come from nearest neighbors around explicit liked restaurants, exact cuisine alignment learned from those likes, then RRF and MMR. Cluster visualizations are explanatory context only.

**Is the health classifier predicting future grades?**

No. It classifies held-out restaurant inspection profiles into A/B/C risk categories. A true future-grade model would require time-sliced historical inspection data.

**Why is the prepared restaurant dataset not larger?**

A larger prepared dataset would likely improve search and recommendation coverage. The committed dataset is intentionally moderate so the repo stays lightweight, runs quickly, and remains reproducible without forcing every local setup to fetch thousands of Google Places records.

---

## License

MIT License. See [LICENSE](LICENSE).
