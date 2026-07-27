# NYC Restaurant Survival Guide

**Advisor:** Professor [Kyunghyun Cho](https://kyunghyuncho.me)

NYC Restaurant Survival Guide is a Streamlit machine-learning app for exploring New York City restaurants. It combines real NYC Department of Health and Mental Hygiene inspection data, cached Google Places metadata, semantic search, a health-grade risk classifier, K-Means clustering from scratch, PCA-based cluster visualization, and personalized recommendations from saved liked restaurants.

The repository is set up to run locally without live downloads. The prepared restaurant table, embedding matrix, classifier checkpoint, and clustering caches are committed so the demo can start from the submitted files. Restaurant photos are optional and use Places API (New) at runtime when you provide a local `GOOGLE_API_KEY`.

## Project Checklist

| Requirement area | How this repository addresses it |
|---|---|
| Working app | `app/Main.py` is the Streamlit entry point. The app has five navigable pages and loads from committed caches by default. |
| Real dataset and meaningful task | The project uses NYC DOHMH inspection data plus Google Places metadata to answer user-facing questions about restaurant discovery, health-risk signals, restaurant segments, and liked-history recommendations. |
| Course algorithm implementation | `models/kmeans_scratch.py` implements K-Means++ directly in NumPy, including initialization, assignment, centroid updates, empty-cluster handling, convergence, and multi-start model selection. |
| ML coherence | Semantic retrieval, health-grade classification, clustering, PCA visualization, K selection, and recommendation reranking all operate on documented data representations. |
| Usability | Pages include labeled controls, constrained filters, cached runtime assets, error messages for missing data, and a login/profile flow for saving likes. |
| Deployment | `Dockerfile` and `render.yaml` provide a reproducible Python 3.12 deployment, CPU-only PyTorch, HTTP health checks, secret injection, and persistent profile storage. |
| Repository hygiene | Active code is under `app/`, `models/`, `utils/`, `data/`, and `tests/`. Obsolete checkpoint-only modules and old cache files have been removed. The largest orchestration functions (`semantic_search`, `_build_cluster_profiles`) are factored into small, single-responsibility stages for easier review. |
| Evaluation integrity | Health-classifier preprocessing is fit on the training split only, with an explicit `fit`/`transform` preprocessor, a persisted fitted object, and a regression test guarding against leakage. |

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
.venv/bin/streamlit run app/Main.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

### 4. Optional: Enable Restaurant Photos

Search results still work without a Google key, but restaurant photos are loaded at runtime through **Places API (New)** using each restaurant's committed `g_place_id`. On Google Cloud, enable **Places API (New)** for the project that owns your key. The UI does not use the legacy Place Photo URL.

Add a key in one of these two ways:

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
.venv/bin/streamlit run app/Main.py
```

The real `.streamlit/secrets.toml` file is intentionally ignored by Git and must never be committed. If photos do not appear after adding a key, confirm that Places API (New) is enabled, billing is active, and the key's application/API restrictions allow local development.

### 5. Optional Pipenv Workflow

```bash
pip install pipenv
pipenv install
pipenv run streamlit run app/Main.py
```

---

## Step-By-Step App Use

1. **Log in or create an account.** User accounts are stored in `data/user_profiles.local.json` locally, or at `USER_PROFILES_PATH` when configured for deployment. The runtime file is ignored by Git so personal profiles, liked restaurants, password hashes, and salts are not committed.
2. **Start on the Home page.** Try one of the suggested restaurant queries, review the result cards, and click "Like this restaurant" on places you would actually want.
3. **Open Semantic Search.** Use natural language such as `cozy Italian pasta in Brooklyn`, `late night ramen Manhattan`, or `cheap Caribbean food Bronx`. The page uses cached sentence embeddings when available and falls back gracefully if the embedding model cannot load.
4. **Open Health Grade Risk Classifier.** Select a held-out restaurant, inspect predicted A/B/C risk probabilities, change inspection-pattern inputs, and review feature importance plus the constrained "Path to A" analysis.
5. **Open Restaurant Cluster GIS Map.** View restaurants colored by learned cluster on an NYC map. The default clustering path uses the NumPy K-Means++ implementation; GMM and Ward are included as comparison baselines.
6. **Open PCA Embedding Explorer.** Inspect the same clusters in 3D PCA, centroid-distance view, or t-SNE; use the feature loading and prototype panels to explain what separates clusters.
7. **Open Personalized Recommendations.** Review the current profile's liked restaurants in the sidebar and remove any saved like if needed. New likes are added from Home or Semantic Search result cards. Recommendations are generated from liked-history nearest neighbors, RRF fusion, and MMR diversity reranking. The cluster visualization on this page is explanatory context, not a "you belong to this cluster" rule.

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

`data/preprocess.py` aggregates raw inspection-violation rows into one row per restaurant, keeps the latest grade as the label, splits the rows into train/test **before** any statistic is learned, fits all preprocessing on the training split only, and writes:

| File | Shape | Purpose |
|---|---:|---|
| `data/train.csv` | `19,366 x 26` | Classifier training rows and target |
| `data/test.csv` | `4,842 x 26` | Held-out classifier test rows and target |
| `data/meta_train.csv` | `19,366 x 8` | Restaurant metadata for train rows |
| `data/meta_test.csv` | `4,842 x 8` | Restaurant metadata for held-out UI selection |
| `data/feature_config.json` | config | Feature names, label mapping, train-only imputation/scaler statistics |
| `data/cache/health_preprocessor.joblib` | object | Fitted, train-only `HealthFeaturePreprocessor` (replayed at test/inference time) |

(Exact row counts depend on how many raw rows you download; the committed splits are built from 200,000 raw inspection rows, which aggregate to 24,208 unique graded restaurants.)

The health classifier uses 25 input features:

- `num_inspections`, `num_violations`, `violations_per_inspection`
- borough one-hot features
- top-15 cuisine one-hot features plus `cuisine_Other`

Score-derived columns such as `latest_score`, `avg_score`, `max_score`, and `critical_ratio` are intentionally excluded because DOHMH grades are derived from inspection score thresholds. Keeping them would leak the label.

### Leakage-Safe Preprocessing (Train-Only Fit/Transform)

Every preprocessing decision that *learns* something from data is treated as part of the model and is fit on the training split only:

1. **Split first.** `preprocess_dohmh()` aggregates to one row per restaurant, then calls `train_test_split` on the *raw* aggregated rows (stratified on grade, `random_state=42`).
2. **Fit on train only.** `HealthFeaturePreprocessor.fit(train_rows)` learns the median imputation values, the `StandardScaler` mean/scale, the top-15 cuisine vocabulary, and the one-hot column schema — all from the training rows alone.
3. **Replay on test/inference.** `transform()` applies those fixed statistics to the held-out test rows (and to single live rows in the app) without ever recomputing them. Cuisines unseen in training collapse into `cuisine_Other`; boroughs unseen in training produce an all-zero borough sub-vector.
4. **Persist the fitted object.** The preprocessor is saved to `data/cache/health_preprocessor.joblib`, and its learned statistics are mirrored into `feature_config.json` (`numerical_medians`, `scaler_mean`, `scaler_scale`, `boro_categories`, `cuisine_categories`, and `fit_on: "train_split_only"`).

This prevents the earlier behavior where imputation medians, scaling statistics, and the cuisine vocabulary were computed on the full dataset before splitting, which let the held-out test distribution leak into the training-time representation. The regression test `tests/test_preprocess_leakage.py` asserts these statistics come from the training rows only.

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
| `data/cache/health_preprocessor.joblib` | object | Fitted train-only preprocessor (imputation, scaler, cuisine vocab, schema) |
| `data/cluster_cache.parquet` | `2,835 x 48` | K-Means clustered restaurant table |
| `data/kmeans_model.joblib` | model cache | K-Means model, scaler, and PCA artifacts |
| `data/cluster_cache_gmm.parquet` | `2,835 x 48` | GMM baseline clustered table |
| `data/cluster_model_gmm.joblib` | model cache | GMM baseline artifacts |
| `data/cluster_cache_agglo.parquet` | `2,835 x 48` | Ward/agglomerative baseline clustered table |
| `data/cluster_model_agglo.joblib` | model cache | Ward/agglomerative baseline artifacts |

Earlier experimental caches, including old v3 prepared search files, partial v4 embedding files, and hyperparameter-search JSON output, are not needed for the final app and have been removed from the active repository state.

### Cluster Cache Behavior

The cluster-related pages all use the same cache-aware clustering helper:

- `Restaurant Cluster GIS Map`
- `PCA Embedding Explorer`
- `Personalized Recommendations`

You do not need to open the GIS Map page first. Any of these pages can load the committed cluster cache directly. If a matching cache is missing, stale, corrupt, or no longer matches the selected algorithm/K/data signature, that page recomputes the clusters and writes a fresh cache. User liked history does not invalidate the cluster cache because clustering is learned from restaurant features only; liked history is recomputed as per-session context for affinity scores, recommendation lines, and the liked-history marker.

### Dataset Size Choice

The prepared search sample starts from `3,800` candidate restaurants and keeps `2,835` restaurants after Google enrichment and validity filters. A larger prepared dataset could improve search and recommendation coverage, especially for rare cuisines and neighborhoods. For the submitted project, the cache size is deliberately moderate so the repository stays lightweight, starts quickly on a local laptop, and still lets users rebuild a larger local cache from NYC DOHMH plus Google Places if they want more coverage.

### Profile Storage

Runtime accounts are written to `data/user_profiles.local.json` by default, or to the path in the `USER_PROFILES_PATH` environment variable. The default file is ignored by Git; the committed `data/user_profiles.json` and `data/user_profiles.example.json` are empty placeholders only. Writes use an atomic file replacement and a process-wide lock so simultaneous Streamlit sessions cannot read partially written JSON. The Render deployment mounts `/app/storage` and sets `USER_PROFILES_PATH=/app/storage/user_profiles.json`, so accounts and liked history survive restarts and redeploys.

New passwords are stored with salted PBKDF2-HMAC-SHA256 (600,000 iterations). Profiles created by an earlier version with the legacy salted SHA-256 format still work and are upgraded after their next successful login. This JSON store is intended for a single app instance; use a managed database before scaling the service to multiple instances.

---

## Models And Algorithms

### Semantic Search

Implemented in `utils/search.py` and used by `app/Main.py` plus `app/pages/1_🔍_Semantic_Search.py`.

- Restaurant descriptions combine name, cuisine, borough/neighborhood, address, Google summary, rating, price tier, and health-grade information.
- Cached embeddings use `sentence-transformers/all-mpnet-base-v2`.
- Query and restaurant vectors are L2-normalized, so cosine similarity is computed as a dot product.
- Structured guardrails for cuisine, borough/neighborhood, price, and quality keep explicit user intent from being overwhelmed by generic semantic matches.
- If the embedding model is unavailable, the search code falls back to lexical and structured scoring instead of crashing.
- `semantic_search` is organized into reviewable stages — query expansion, semantic similarity, signal components, hard filters, signal blending, the price-intent cascade, relevance gating, and final ranking — each in its own helper function.

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
- All preprocessing statistics are fit on the training split only (see "Leakage-Safe Preprocessing" above), so the held-out metrics reflect a clean train/test protocol.
- Stratified validation split is taken from the training data.
- Early stopping monitors validation weighted F1.
- The page reports held-out accuracy/F1, majority baseline, confusion matrix, per-class metrics, permutation importance, local sensitivity, and path-to-A feature edits.
- The committed checkpoint can be regenerated deterministically from a terminal with `python -m models.train_health_classifier`, which mirrors the exact recipe the Streamlit page uses and prints the held-out metrics below.

Current held-out metrics from the committed checkpoint (train-only preprocessing, `4,842` held-out restaurants):

| Metric | Value |
|---|---:|
| Accuracy | `76.4%` |
| Weighted F1 | `0.804` |
| Macro F1 | `0.508` |

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

The Recommendation sidebar is intentionally limited to the active profile's saved likes because the recommender should expose the exact positive signals used for ranking. Likes are added from restaurant discovery surfaces, while the Recommendation page focuses on reviewing that saved preference signal and removing a like when needed.

---

## Repository Structure

```text
nyc-restaurant-survival-guide/
├── .github/workflows/ci.yml
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── app/
│   ├── __init__.py
│   ├── Main.py
│   ├── ui_utils.py
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py
│       ├── 2_🧪_Health_Grade_Classifier.py
│       ├── 3_📍_Restaurant_Cluster_Map.py
│       ├── 4_📊_PCA_Embedding_Explorer.py
│       └── 5_🔮_Recommendations.py
├── data/
│   ├── __init__.py
│   ├── download_data.py
│   ├── preprocess.py
│   ├── feature_config.json
│   ├── train.csv / test.csv
│   ├── meta_train.csv / meta_test.csv
│   ├── user_profiles.json / user_profiles.example.json
│   ├── cluster_cache.parquet
│   ├── cluster_cache_agglo.parquet / cluster_cache_gmm.parquet
│   ├── kmeans_model.joblib
│   ├── cluster_model_agglo.joblib / cluster_model_gmm.joblib
│   └── cache/
│       ├── prepared_search_v4_3800.pkl
│       ├── embeddings_prepared_v4_3800_2835.npy
│       ├── enriched_restaurants_3800.pkl
│       ├── health_classifier.pt
│       ├── health_classifier_history.json
│       ├── health_classifier_importance.json
│       └── health_preprocessor.joblib
├── models/
│   ├── __init__.py
│   ├── custom_mlp.py
│   ├── kmeans_scratch.py
│   └── train_health_classifier.py
├── scripts/
│   └── smoke_test_app.py
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_custom_mlp.py
│   ├── test_kmeans_scratch.py
│   ├── test_preprocess_leakage.py
│   ├── test_recommendations.py
│   └── test_semantic_search.py
├── utils/
│   ├── __init__.py
│   ├── auth.py
│   ├── clustering.py
│   ├── data.py
│   ├── google_places.py
│   ├── place_photos.py
│   ├── search.py
│   ├── search_assets.py
│   └── user_profile.py
├── .gitignore
├── .dockerignore
├── Dockerfile
├── LICENSE
├── requirements.txt
├── Pipfile
├── render.yaml
└── README.md
```

---

## Rebuild Or Expand The Data

The submitted app does not expose a Streamlit "refresh data" button. Rebuilding is a terminal workflow.

### Rebuild DOHMH Classifier Splits

```bash
source .venv/bin/activate
python data/download_data.py 200000
python data/preprocess.py
```

This rewrites `data/train.csv`, `data/test.csv`, `data/meta_train.csv`, `data/meta_test.csv`, `data/feature_config.json`, and the fitted `data/cache/health_preprocessor.joblib`. Preprocessing splits the rows first and then fits all imputation/scaling/vocabulary statistics on the training split only, so the regenerated test rows stay genuinely held out. You can pass a smaller number (e.g. `50000`) for a lighter rebuild; the documented committed splits use `200000` raw rows.

To regenerate the committed classifier checkpoint and print fresh held-out metrics, run:

```bash
python -m models.train_health_classifier
```

This rewrites `data/cache/health_classifier.pt` and `data/cache/health_classifier_history.json`, and removes the stale `data/cache/health_classifier_importance.json` so the app recomputes it lazily. If the checkpoint is simply deleted, the Health Grade Risk Classifier page will also retrain and save a new one from the prepared files on next load.

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

The committed cache is enough for the submitted demo. Rebuilding the Google-enriched cache may require enabling the legacy Places Text Search/Details endpoints because the enrichment pipeline was originally built against those endpoints. Runtime photo display is separate and uses Places API (New) through `utils/place_photos.py`.

To try a larger local prepared cache, edit `DEFAULT_SEARCH_SAMPLE_SIZE` in `utils/search_assets.py` or call `load_prepared_search_assets(sample_size=YOUR_SIZE, force_refresh=True)`. Larger samples improve coverage but require more Google API calls, more embedding time, and larger cache files.

---

## Tests And Smoke Checks

```bash
# Inside the activated .venv (pytest is in requirements.txt):
python -m pytest tests/ -q
python -m py_compile app/Main.py app/pages/*.py utils/*.py models/*.py data/*.py
python scripts/smoke_test_app.py
```

The documented `.venv` (built from `requirements.txt`) includes `pytest`, so the suite runs cleanly without extra setup. The full suite is 20 tests. GitHub Actions repeats compilation, the test suite, and the six-entry-point page smoke test on every push to `main` and on pull requests using Python 3.12 and the CPU-only PyTorch wheel.

Current test coverage checks:

- K-Means++ fit, predict, and distance-transform behavior
- Custom MLP forward pass and training loop behavior
- Semantic search fallback and borough-filter behavior without downloading a model
- Recommendation reranking behavior
- Account registration/authentication, PBKDF2 password storage, legacy-hash migration, and atomic profile writes
- **Preprocessing leakage guard** (`tests/test_preprocess_leakage.py`): asserts the scaler mean/scale, imputation medians, and cuisine vocabulary are learned from the training rows only, that transforming held-out rows never mutates those statistics, and that unseen test categories fall back safely.

During repository review, the Streamlit pages were also smoke-tested with `streamlit.testing.v1.AppTest`; all five pages plus the landing page loaded with zero page exceptions using the committed runtime data and checkpoint.

---

## Deployment And Security Notes

The repository is ready for a container-based Render deployment. The included Blueprint selects a Standard web service because the app loads PyTorch, transformer search components, Plotly, and committed ML caches; it also adds a 1 GB persistent disk for accounts and likes. Render services and disks are billed resources, so review the selected plan before confirming the Blueprint.

### Deploy On Render

1. In Render, choose **New → Blueprint** and connect `hwyuanzi/nyc-restaurant-survival-guide`. Render detects `render.yaml` from `main` and builds the included `Dockerfile`.
2. Enter `GOOGLE_API_KEY` when prompted, or leave photos disabled and add the secret later. Never put the real value in `render.yaml`.
3. Wait for `/_stcore/health` to pass, then open the generated `*.onrender.com` URL. Future pushes to `main` trigger a new deployment.
4. In the service's **Settings → Custom Domains**, add `foodguide.hyuan.io`.
5. At the DNS provider for `hyuan.io`, create the CNAME record Render shows (normally host `foodguide` pointing to the service's `*.onrender.com` hostname), then return to Render and verify it. Render provisions and renews TLS automatically and redirects HTTP to HTTPS. Follow Render's provider-specific instructions if Cloudflare proxying or existing `AAAA` records are enabled.

The container binds Streamlit to the platform's `PORT` on `0.0.0.0`, runs as a non-root user, installs the official CPU-only PyTorch wheel, excludes local secrets/raw data from its build context, and exposes both Docker and Render health checks. The persistent disk is mounted at `/app/storage` and is the only place where deployed profile writes are retained.

### Streamlit Community Cloud Alternative

For a quick free demo, create an app from this GitHub repository, choose branch `main`, entry point `app/Main.py`, and Python 3.12, then add `GOOGLE_API_KEY` in the platform's Secrets settings. Community Cloud provides a customizable `*.streamlit.app` subdomain, but its app filesystem is ephemeral; account persistence and the direct `foodguide.hyuan.io` setup above are why Render is the recommended path for this project.

### Security Checklist

- Do not commit `.streamlit/secrets.toml` or real API keys.
- Keep `data/user_profiles.local.json` untracked because it can contain local usernames, password hashes, salts, preferences, and likes.
- Restrict the Google key to Places API (New), set an appropriate quota, and add application restrictions supported by the deployment setup.
- Treat generated model/data caches as reproducible artifacts; rebuild instructions are documented above.
- Rotate any Google key that has been pasted into chat, screenshots, public logs, or committed history.

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
