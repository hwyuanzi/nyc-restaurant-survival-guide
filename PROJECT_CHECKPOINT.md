# Project Checkpoint — NYC Restaurant Survival Guide
**CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026**

> Historical checkpoint note: this document records the early proposal/design checkpoint. It is kept for course-process context, but it is not the source of truth for the final demo. See [README.md](README.md) for the current implementation, algorithms, data files, and rubric-facing explanation.

## Part 1: Written Proposal

**Problem Description**
New York City's restaurant ecosystem is uniquely complex: over 27,000 active food establishments are regulated by the Department of Health and Mental Hygiene (DOHMH), which publishes granular inspection records—including violation codes, critical flags, and letter grades—through the NYC OpenData API. For diners and restaurant owners, navigating this landscape is overwhelming. Our project, the NYC Restaurant Survival Guide, utilizes this comprehensive dataset to address three concrete user-facing questions: (1) "Given a natural-language description of my desired dining experience, which NYC restaurants best match?" (2) "Given a restaurant's operational profile, what is its predicted health grade, and what minimal changes would flip it to an A?" and (3) "When projecting restaurant embeddings into a latent space, how do they naturally group based on geographic location (e.g., Manhattan vs. Brooklyn), and what underlying characteristics define the most dense clusters?" These questions are interesting because they combine subjective text queries (vibe/atmosphere), objective regulatory compliance forecasting, and high-dimensional geospatial data exploration, addressing real-world needs through a unified representation learning system.

**Methods**
To solve these problems, our system employs several core algorithms from the course syllabus. For semantic retrieval, we use **Transformers for discrete-to-continuous conversion (Week 3)** to generate 384-dimensional embeddings of restaurant descriptions, enabling nearest neighbor search using cosine similarity **(Week 4)**. For operational forecasting, we train a **Multi-Layer Perceptron (MLP) for Multi-class Classification (Week 8)** to predict health grades (A/B/C) based on inspection features. To discover latent patterns, we use **Deep Autoencoders for Dimensionality Reduction (Week 6)** to compress numerical operational features into a 32-dimensional latent space. Crucially, to answer our third question regarding visual clustering, we apply **Principal Component Analysis (PCA) (Week 6)** to simultaneously reduce both the Transformer embeddings (text) and Autoencoder embeddings (operations) into 2D/3D visualizations. This multi-model approach enables users to apply geographic filters (e.g., Manhattan only) and dynamically compare the topological distributions. Finally, we apply **k-means clustering (Week 7)** on the PCA-reduced embeddings to automatically identify and isolate the "most dense" clusters, allowing the system to extract and display the shared characteristics (e.g., cuisine, grade prevalence, or price point) of the clustered restaurants. These methods are highly appropriate because they leverage both linear (PCA) and non-linear (Autoencoders/Transformers) dimensionality reduction, blending retrieval with unsupervised cluster discovery.

## Part 2: Design Document

**1. Repository Structure**
```text
nyc-restaurant-survival-guide/
├── app/                        # Streamlit frontend application directory
│   ├── main.py                 # Landing page, navigation routing, and global app config
│   ├── ui_utils.py             # Global CSS theme injection and UI helper functions
│   └── pages/                  # Streamlit multipage application views
│       ├── 1_Semantic_Search.py         # NLP vibe search via Transformer embeddings and cosine retrieval
│       ├── 2_ML_Action_Sandbox.py       # Interactive MLP classifier for health grade prediction
│       ├── 3_Latent_Topography.py       # Autoencoder latent space visualization
│       ├── 4_Dynamic_GIS_Map.py         # PyDeck 3D hexbin & Mapbox geospatial explorer
│       └── 5_PCA_Embedding_Explorer.py  # Multi-model PCA visualization, k-means density clustering, and geographic filtering
├── models/                     # PyTorch model definitions and training logic
│   ├── custom_mlp.py           # 3-layer MLP classifier implementation for grade prediction
│   └── autoencoder.py          # Deep Autoencoder architecture and embedding extractor
├── retrieval/                  # NLP vector retrieval sub-system
│   └── vector_search.py        # HuggingFace Transformer embedding generator and similarity ranking
├── data/                       # Data pipeline and preprocessing module
│   ├── download_data.py        # Automated NYC DOHMH OpenData API data collection script
│   └── preprocess.py           # Cleaning, feature engineering, and reproducible train/test split generation
├── tests/                      # Pytest unit and integration test suite
│   ├── test_custom_mlp.py      # Validates MLP forward passes and output shapes
│   ├── test_autoencoder.py     # Validates AE intermediate embedding extraction
│   └── test_semantic_search.py # Validates embedding dimension constraints and L2 norms
├── .gitignore                  # Git ignore rules for caches and data artifacts
├── .streamlit/config.toml      # Streamlit framework layout and global theme configuration
├── Pipfile                     # Dependency lock file for deterministic virtual environments
├── requirements.txt            # Package dependency list for alternative environment setup
└── README.md                   # Project overview, setup tutorial, and runtime instructions
```

**2. Division of Labor**
*   **Member A (NLP Engineer):** Owns `retrieval/vector_search.py`, `app/pages/1_Semantic_Search.py`, and `tests/test_semantic_search.py`. Responsible for the Transformer embedding pipeline and nearest neighbor search view.
*   **Member B (ML Classifier Lead):** Owns `models/custom_mlp.py`, `app/pages/2_ML_Action_Sandbox.py`, and `tests/test_custom_mlp.py`. Responsible for the multi-label health grade classification engine.
*   **Member C (Autoencoder Architect):** Owns `models/autoencoder.py`, `app/pages/3_Latent_Topography.py`, and `tests/test_autoencoder.py`. Responsible for unsupervised dimensionality reduction of operational parameters.
*   **Member D (Visualization & Geo-Data Lead):** Owns `app/pages/4_Dynamic_GIS_Map.py` and `app/pages/5_PCA_Embedding_Explorer.py`. Responsible for implementing PCA on multi-model embeddings, geographic filtering (e.g., Borough selection), and k-means dense-cluster characteristic analysis.
*   **Member E (Data & DevOps Engineer):** Owns `data/download_data.py`, `data/preprocess.py`, `app/main.py`, and `app/ui_utils.py`. Responsible for defining reproducible data splits, API extraction, and global UI layout.

**3. Stub Code in Public GitHub Repo**
*   **Public Repository:** The project repository exists and is accessible at https://github.com/hwyuanzi/nyc-restaurant-survival-guide
*   **README & Environments:** The root directory contains a comprehensive `README.md`, alongside a `requirements.txt` and `Pipfile` specifying identical environments (Python, PyTorch, Streamlit, scikit-learn). A working virtual environment can be instantly generated via `pipenv install` or `pip install -r requirements.txt`.
*   **Stub Directories & Files:** All directories (`app/`, `models/`, `retrieval/`, `data/`, `tests/`) have been successfully established. Empty placeholder (stub) files exist exactly as mapped in the repository structure above to support parallel, conflict-free development across the 5 team members.
