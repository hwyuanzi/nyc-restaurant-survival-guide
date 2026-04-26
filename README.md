# NYC Restaurant Survival Guide

**CSCI-UA 473 — Fundamentals of Machine Learning**<br>
New York University · Spring 2026

An end-to-end Streamlit machine-learning dashboard for exploring NYC restaurants. The app combines real NYC Department of Health inspection data, Google Places metadata, semantic search embeddings, an inspection-risk classifier, interpretable clustering, PCA visualization, and liked-history recommendations.

The submitted repo is designed to run as a working demo without live data downloads: the required processed CSVs, Google-enriched restaurant cache, embedding cache, model checkpoint, and clustering cache are included.

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

## Quick Start

```bash
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
streamlit run app/Main.py
```

Open `http://localhost:8501`, sign up or log in, then use the sidebar navigation.

Optional Pipenv workflow:

```bash
pip install pipenv
pipenv install
pipenv run streamlit run app/Main.py
```

The app can run without a Google Places API key because `data/cache/prepared_search_v4_3800.pkl` and `data/cache/embeddings_prepared_v4_3800_2835.npy` are committed. A Google key is only needed if you intentionally rebuild the enrichment cache.

---

## Rubric Readiness

| Rubric item | Project status |
|---|---|
| **Working Demo — 10%** | The app launches from `app/Main.py`, uses committed caches by default, handles missing external APIs with cache fallback, and has five navigable Streamlit pages. |
| **Algorithm Implementation — 10%** | `models/kmeans_scratch.py` implements K-Means++ from scratch in NumPy and is the default algorithm used by the Restaurant Cluster GIS Map. `models/pca_scratch.py` is also implemented from scratch and used in the Health Grade Classifier context plot. |
| **Data/task meaningfulness — 3 pts** | Uses real NYC DOHMH inspection data and Google Places metadata. User-facing tasks are plausible: search restaurants, inspect health-risk signals, discover restaurant segments, and get recommendations from saved likes. |
| **UI/UX clarity — 3 pts** | Pages are labeled, sidebar controls are constrained, bad inputs show `st.info`/`st.warning`/`st.error`, and expensive data/model work is cached. |
| **Technical correctness — 4 pts** | K-Means, MLP classification, PCA projection, semantic cosine retrieval, K selection, permutation importance, and MMR/RRF recommendation are used coherently and exposed in the UI. |
| **Code quality deductions** | Main active code is under `app/`, `models/`, `utils/`, `data/`, and `tests/`. Obsolete root-level Streamlit pages were removed to avoid demo confusion. Tests pass. |

Current verification command:

```bash
pipenv run pytest tests/test_semantic_search.py tests/test_custom_mlp.py tests/test_autoencoder.py
```

Expected result: **10 tests passing**.

---

## App Pages

| Page | What it does | Main ML concept |
|---|---|---|
| **Home / Landing Search** | Login/signup, cached restaurant search, like/unlike restaurants. | Embedding search + profile persistence |
| **Semantic Search** | Natural-language restaurant search with a clean query-first interface. | Transformer embeddings + cosine similarity |
| **Health Grade Risk Classifier** | Select a held-out DOHMH restaurant, view A/B/C risk probabilities, edit integer inspection-pattern features, inspect permutation importance, and search for a realistic path toward Grade A. | PyTorch MLP classifier + class-weighted cross entropy |
| **Restaurant Cluster GIS Map** | Run K-Means/GMM/Ward on an 18-D interpretable feature space and view clusters on a real NYC map. | K-Means from scratch + clustering baselines |
| **PCA Embedding Explorer** | View the same clusters in 3D PCA, cleaner centroid-distance view, or t-SNE visualization; inspect feature loadings, cluster distances, prototypes, and summary stats. | PCA projection + cluster interpretation |
| **Recommendations** | Add/remove liked restaurants and get recommendations from liked history only. | Per-liked KNN + Reciprocal Rank Fusion + MMR |

---

## Data

### NYC DOHMH Inspection Data

| Property | Value |
|---|---|
| Source | NYC OpenData, Department of Health and Mental Hygiene |
| Endpoint | `https://data.cityofnewyork.us/resource/43nn-pn8j.csv` |
| Raw granularity | One row per inspection violation |
| Processed classifier rows | `11,401` train restaurants + `2,851` held-out test restaurants |
| Class labels | `A`, `B`, `C` mapped to `0`, `1`, `2` |
| Current train distribution | A: `9,235`, B: `1,396`, C: `770` |
| Current test distribution | A: `2,310`, B: `349`, C: `192` |

`data/preprocess.py` aggregates raw violation rows to one restaurant profile per `camis`, keeps the latest DOHMH grade as the target, and writes:

- `data/train.csv`
- `data/test.csv`
- `data/meta_train.csv`
- `data/meta_test.csv`
- `data/feature_config.json`

The classifier currently uses **25 features**:

- 3 standardized numeric inspection-history features:
  `num_inspections`, `num_violations`, `violations_per_inspection`
- 6 borough one-hot features:
  `boro_0`, `boro_Bronx`, `boro_Brooklyn`, `boro_Manhattan`, `boro_Queens`, `boro_Staten Island`
- 16 cuisine one-hot features:
  top-15 cuisine categories plus `cuisine_Other`

Score-derived columns such as `latest_score`, `avg_score`, `max_score`, and `critical_ratio` are intentionally excluded from the final classifier feature set because DOHMH grades are derived from inspection score thresholds. Keeping them produced label leakage.

### Google-Enriched Restaurant Cache

| File | Purpose |
|---|---|
| `data/cache/prepared_search_v4_3800.pkl` | Cached Google-enriched restaurant table, currently `2,835` rows |
| `data/cache/embeddings_prepared_v4_3800_2835.npy` | Cached 768-D sentence-transformer embeddings for the prepared restaurants |
| `data/cluster_cache.parquet` | Cached clustered restaurant table |
| `data/kmeans_model.joblib` | Cached K-Means model, scaler, and PCA artifacts |

At runtime, `utils/search_assets.py` first tries the committed cache, then falls back to live DOHMH/Google rebuild only if requested.

---

## Models And Algorithms

### 1. Semantic Search

Implemented in `utils/search.py`.

- Text descriptions combine restaurant name, cuisine, borough/neighborhood, Google summary, rating, price, health grade, and address.
- Embeddings use `sentence-transformers/all-mpnet-base-v2` when available.
- Query and restaurant vectors are L2-normalized, so cosine ranking is a dot product.
- Cuisine and neighborhood query hints add guardrails for terms like `pho`, `Vietnamese`, `romantic French bistro`, and borough/neighborhood phrases.
- The UI no longer shows misleading match percentages; users see ranked restaurant cards instead.

### 2. Health Grade Risk Classifier

Implemented in `models/custom_mlp.py` and `app/pages/2_🧪_Health_Grade_Classifier.py`.

Architecture:

```text
Input(25)
  -> Linear(25, 128) -> ReLU -> Dropout(0.3)
  -> Linear(128, 128) -> ReLU -> Dropout(0.3)
  -> Linear(128, 3)
```

Training:

- Optimizer: AdamW
- Loss: class-weighted `CrossEntropyLoss`
- Validation split: 20% of training data, stratified
- Early stopping: validation weighted F1, patience 12
- Checkpoint: `data/cache/health_classifier.pt`
- History: `data/cache/health_classifier_history.json`

Current held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | `70.4%` |
| Weighted F1 | `0.708` |
| Macro F1 | `0.394` |

Per-class behavior is intentionally shown in the app because the DOHMH labels are imbalanced. Grade A dominates, while B/C are rarer and harder. The page frames predictions as **risk signals**, not official future-grade forecasts.

Explainability:

- Permutation importance shows which input groups affect weighted F1.
- What-if controls use integer inspection counts, not fractional counts.
- `violations_per_inspection` is derived from total violations and inspection count.
- The constrained "Path to A" search lowers actionable violation count while holding inspection history fixed.
- A 2D PCA context map uses `models/pca_scratch.PCAScratch`.

### 3. K-Means Clustering From Scratch

Implemented in `models/kmeans_scratch.py`; used by default in `utils/clustering.py` and the Restaurant Cluster GIS Map.

Algorithm details:

1. K-Means++ centroid initialization.
2. Vectorized Euclidean assignment step.
3. Centroid update by cluster means.
4. Empty cluster reinitialization to a random data point.
5. Convergence by unchanged labels or centroid shift below `tol`.
6. Multi-start `n_init`; keep the lowest inertia run.

This directly satisfies the course requirement for a non-wrapper algorithm implementation. The app also exposes GMM and Ward clustering as comparison baselines, but the default demo path uses the NumPy K-Means implementation.

Clustering feature space:

- 6 numeric features:
  price tier, Google rating, log review count, inverted DOHMH score, latitude, longitude
- 7 cuisine group one-hot features:
  American, Asian, Latin, Cafe, Italian, European, Other
- 5 borough one-hot features:
  Manhattan, Brooklyn, Queens, Bronx, Staten Island

Total: **18 interpretable dimensions**.

Current default K selection:

- The app sweeps K=4..15.
- It tracks silhouette score, inertia, and largest-cluster share.
- It chooses a conservative K from the silhouette knee and K-Means elbow logic.
- Current prepared cache result: **K = 9**, displayed as 9 clusters.
- Small niche clusters are kept, so selected K and displayed cluster count match.

Cluster labels:

K-Means only outputs numeric `cluster_id`s. Human-readable labels are generated afterward from cluster summaries:

- dominant cuisine group when one exists,
- otherwise borough concentration,
- otherwise price tier and rating/review persona.

Therefore labels such as `American`, `Asian`, `Cafe`, `European · Mid-Range`, and `Mid-Range · Highly Rated` are interpretations of learned clusters, not manual assignments.

### 4. PCA And Embedding Explorer

Implemented in `app/pages/4_📊_PCA_Embedding_Explorer.py`.

The clustering model is trained in higher-dimensional standardized feature space. The 3D plots are visualizations:

- **Principal Components**: direct 3D PCA projection of the scaled 18-D feature matrix.
- **Cleaner Cluster View**: PCA projection of each restaurant's distances to all centroids, useful for seeing separation.
- **t-SNE**: optional visualization only; not used for cluster assignment.

The page includes:

- feature loading charts,
- explained variance,
- cluster distance matrix,
- cluster evidence panel,
- prototype restaurants nearest the centroid,
- price tier formatting as `1..4`, where `1=$` and `4=$$$$`.

### 5. Personalized Recommendations

Implemented in `app/pages/5_🔮_Recommendations.py` and `utils/clustering.py`.

Recommendation is now based only on restaurants the user explicitly likes:

1. User adds/removes liked restaurants in the Recommendation sidebar or likes restaurants from search cards.
2. Each liked restaurant is treated as one equal positive example.
3. For each liked restaurant, the app retrieves cosine nearest neighbors in the same 18-D restaurant feature space.
4. Per-liked ranked lists are fused with Reciprocal Rank Fusion (RRF).
5. The top candidates are reranked with Maximal Marginal Relevance (MMR) for diversity.

Cluster membership is shown only as context. The recommender is not "recommend everything from my cluster"; it recommends nearby restaurants around the user's liked examples and can return candidates from any cluster.

---

## Repository Structure

```text
nyc-restaurant-survival-guide/
├── app/
│   ├── main.py
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
│   ├── cluster_cache.parquet
│   ├── kmeans_model.joblib
│   └── cache/
│       ├── prepared_search_v4_3800.pkl
│       ├── embeddings_prepared_v4_3800_2835.npy
│       ├── health_classifier.pt
│       ├── health_classifier_history.json
│       └── hp_search_results.json
├── models/
│   ├── autoencoder.py
│   ├── custom_mlp.py
│   ├── kmeans_scratch.py
│   └── pca_scratch.py
├── retrieval/
│   └── vector_search.py
├── tests/
│   ├── test_autoencoder.py
│   ├── test_custom_mlp.py
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

Primary Streamlit entry point: `app/Main.py`.

---

## Rebuilding Data

The submitted cache is enough for the live demo. To rebuild from raw DOHMH data:

```bash
pipenv run python data/download_data.py 50000
pipenv run python data/preprocess.py
```

To rebuild Google Places enrichment, add an API key:

```toml
# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_key_here"
```

Then use the refresh checkbox in the app or call the cache loaders directly.

---

## Tests And Smoke Checks

```bash
pipenv run pytest tests/ -v
python -m py_compile app/Main.py app/pages/*.py utils/*.py models/*.py
```

Current test coverage:

- MLP forward/training behavior,
- Autoencoder shape/loss/latent output,
- Semantic search embedding shape, normalization, and relevance behavior.

---

## Presentation Notes

Useful answers for common grading questions:

- **Why K-Means?** It is a course algorithm, implemented from scratch, easy to explain, and appropriate after standardizing the interpretable feature space.
- **What does K mean?** K is the requested number of clusters. The current app keeps K displayed clusters rather than merging small groups.
- **Are cluster labels learned directly?** No. The algorithm learns numeric cluster IDs; labels are generated afterward from cluster summary statistics.
- **Why are many prices mid-range?** Google price tier is 1..4, and this sample is concentrated between 1 and 2. That is a data distribution fact, not a labeling bug.
- **Is Recommendation based on cluster?** No. It uses per-liked nearest neighbors in the 18-D feature space, then RRF and MMR. Cluster is context only.
- **Is the health classifier predicting the future?** No. It classifies current/held-out restaurant inspection profiles into A/B/C risk categories. A true future-grade model would need time-sliced inspection histories.

---

## License

MIT License. See [LICENSE](LICENSE).
