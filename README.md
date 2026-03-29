# 🍎 NYC Restaurant Survival Guide

**CSCI-UA 473 — Fundamentals of Machine Learning**
New York University · Spring 2026

A full-stack interactive Machine Learning dashboard for exploring, searching, and predicting health outcomes of NYC restaurants. Built with **PyTorch**, **Streamlit**, **Plotly**, and **HuggingFace Transformers**.

---

## Features & Algorithms

### 🔍 Semantic Vibe Search
Search a corpus of **220 synthetic NYC restaurants** by describing the atmosphere you want in natural language (e.g., *"cozy romantic Italian with dim lighting"*).

| Component | Detail |
|---|---|
| **Model** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Technique** | Transformer embedding → Mean Pooling → L2 normalization → Cosine similarity via `torch.mm()` |
| **Course Topics** | Week 4 (Similarity Metrics, Nearest Neighbor Search), Week 3 (Transformers) |
| **Implementation** | `retrieval/vector_search.py` — `SemanticSearchModel` class with `embed_texts()` and `search()` |

### 🧪 ML Action Sandbox (Health Grade Predictor)
Select any restaurant, adjust its operational parameters (violations, pest control, training hours, operating hours), and watch the neural network re-predict the health grade in real-time with a **live Radar Chart** and **3-class probability bar chart**.

| Component | Detail |
|---|---|
| **Model** | `CustomMLP` — 3-layer Multi-Layer Perceptron (from scratch in PyTorch) |
| **Training** | Trained via **real gradient descent** (`Adam`, `CrossEntropyLoss`) on 2,000 synthetic labeled samples for 80 epochs |
| **Course Topics** | Week 8 (Classification, Multi-label, F1), Week 3 (Optimization) |
| **From-Scratch Requirement** | ✅ `models/custom_mlp.py` implements forward pass, training loop, evaluation, and a counterfactual adversarial engine — no scikit-learn wrappers |

### 🌌 Autoencoder Latent Topography
Visualize 1,200 restaurant operational profiles compressed from **6 dimensions → 2D** via a trained Deep Autoencoder. Explore clusters with interactive color mapping and marginal distributions.

| Component | Detail |
|---|---|
| **Model** | `RestaurantAutoencoder` — Encoder(6→64→32→2) + Decoder(2→32→64→6) |
| **Training** | Trained via **real gradient descent** (`Adam`, `MSELoss`) for 80 epochs on normalized feature data |
| **Course Topics** | Week 6 (Dimensionality Reduction, Deep Autoencoders) |
| **Implementation** | `models/autoencoder.py` — `train_autoencoder()` and `get_latent_space()` produce real neural network outputs |

### 📍 Dynamic Geospatial GIS
Explore **2,500 simulated restaurants** across NYC's 5 boroughs on a 3D hexagonal density map (PyDeck) and a multi-feature spatial scatter map (Plotly Mapbox). Color by Grade, Violations, Pest Control Days, or Training Hours.

| Component | Detail |
|---|---|
| **Libraries** | PyDeck (`HexagonLayer` + `ScatterplotLayer`), Plotly Express (`scatter_mapbox`) |
| **Course Topics** | Data visualization, feature engineering |

---

## Repository Structure

```
nyc-restaurant-survival-guide/
├── app/                        # Streamlit frontend
│   ├── main.py                 # Landing page & navigation
│   ├── ui_utils.py             # Global Apple-inspired CSS theme
│   └── pages/
│       ├── 1_🔍_Semantic_Search.py     # NLP vibe search
│       ├── 2_🧪_ML_Action_Sandbox.py   # Interactive health prediction
│       ├── 3_🌌_Latent_Topography.py   # Autoencoder visualization
│       └── 4_📍_Dynamic_GIS_Map.py     # Geospatial intelligence
├── models/                     # PyTorch model implementations
│   ├── custom_mlp.py           # From-scratch MLP + counterfactual engine
│   └── autoencoder.py          # Deep Autoencoder for dimensionality reduction
├── retrieval/                  # NLP retrieval system
│   └── vector_search.py        # Transformer embedding + cosine similarity search
├── data/                       # Data pipeline
│   ├── download_data.py        # NYC DOHMH OpenData API fetcher
│   └── preprocess.py           # Feature engineering & train/test splits
├── tests/                      # Unit test suite
│   ├── test_custom_mlp.py      # MLP forward shape + training convergence
│   ├── test_autoencoder.py     # AE forward shape + loss reduction + latent output
│   └── test_semantic_search.py # Embedding shape + L2 norm + semantic relevance
├── .streamlit/config.toml      # Streamlit theme configuration
├── Pipfile                     # Pipenv dependency lock
├── requirements.txt            # Pip dependency list
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- `pip` and `pipenv` (or just `pip`)

### Option A: Using Pipenv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide

# 2. Install dependencies
pipenv install

# 3. Run the dashboard
pipenv run streamlit run app/main.py
```

### Option B: Using pip

```bash
# 1. Clone the repository
git clone https://github.com/hwyuanzi/nyc-restaurant-survival-guide.git
cd nyc-restaurant-survival-guide

# 2. Create a virtual environment and install
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app/main.py
```

### Optional: Download Real DOHMH Data

```bash
# Fetch 50,000 rows from the NYC OpenData API
pipenv run python data/download_data.py

# Preprocess into train/test splits
pipenv run python data/preprocess.py
```

The app runs with synthetic data by default — downloading real data is optional for further experimentation.

---

## Running Tests

```bash
pipenv run pytest tests/ -v
```

Expected output: **9 tests, all passing** — covering the MLP, Autoencoder, and Semantic Search model.

---

## How to Use the Dashboard

1. **Open the app** in your browser (default: `http://localhost:8501`).
2. **Use the sidebar** to navigate between the four ML tools.
3. **Semantic Search:** Type a natural-language description → get ranked restaurant matches.
4. **ML Sandbox:** Select a restaurant → adjust sliders → watch the neural network prediction and radar chart update live.
5. **Latent Topography:** Change the color mapping dropdown → hover over clusters to inspect feature distributions.
6. **GIS Map:** Rotate the 3D map with `Shift + Drag` → zoom into specific neighborhoods → switch color metrics.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, PyDeck |
| ML Models | PyTorch (from scratch) |
| NLP | HuggingFace Transformers (`all-MiniLM-L6-v2`) |
| Data | Pandas, NumPy, NYC DOHMH OpenData API |
| Testing | Pytest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.