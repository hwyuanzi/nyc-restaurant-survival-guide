import warnings

import numpy as np
import pandas as pd
import torch

from models.custom_mlp import CustomMLP
from models.kmeans_scratch import KMeansScratch
from utils import clustering as clustering_utils
from utils import search as search_utils


def test_kmeans_pp_identical_points_avoids_runtime_warning():
    X = np.ones((8, 3), dtype=np.float64)
    model = KMeansScratch(n_clusters=4, random_state=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = model.fit_predict(X)

    assert len(labels) == len(X)
    assert model.inertia_ == 0.0
    assert np.isfinite(model.cluster_centers_).all()
    assert not any("invalid value encountered in divide" in str(w.message) for w in caught)


def test_get_clustered_data_supports_k2_without_projection_crash(tmp_path, monkeypatch):
    df = pd.DataFrame(
        [
            {
                "restaurant_id": "r1",
                "name": "North Noodles",
                "lat": 40.71,
                "lng": -73.99,
                "cuisine_type": "Chinese",
                "price_tier": 1,
                "avg_rating": 4.3,
                "review_count": 120,
                "boro": "Brooklyn",
                "score": 8,
                "grade": "A",
                "tags": "noodles dumplings",
                "g_summary": "casual noodle shop",
                "description": "brooklyn noodles dumplings",
            },
            {
                "restaurant_id": "r2",
                "name": "South Sushi",
                "lat": 40.72,
                "lng": -73.98,
                "cuisine_type": "Japanese",
                "price_tier": 3,
                "avg_rating": 4.6,
                "review_count": 240,
                "boro": "Manhattan",
                "score": 10,
                "grade": "A",
                "tags": "sushi omakase",
                "g_summary": "high end sushi counter",
                "description": "manhattan sushi omakase",
            },
            {
                "restaurant_id": "r3",
                "name": "East Pizza",
                "lat": 40.73,
                "lng": -73.97,
                "cuisine_type": "Italian",
                "price_tier": 2,
                "avg_rating": 4.1,
                "review_count": 90,
                "boro": "Queens",
                "score": 12,
                "grade": "B",
                "tags": "pizza pasta",
                "g_summary": "family pizza place",
                "description": "queens pizza pasta",
            },
            {
                "restaurant_id": "r4",
                "name": "West Tacos",
                "lat": 40.74,
                "lng": -73.96,
                "cuisine_type": "Mexican",
                "price_tier": 1,
                "avg_rating": 4.0,
                "review_count": 75,
                "boro": "Bronx",
                "score": 15,
                "grade": "B",
                "tags": "tacos street food",
                "g_summary": "late night taco spot",
                "description": "bronx tacos street food",
            },
        ]
    )

    class DummyKMeans:
        def __init__(self, n_clusters, **kwargs):
            self.n_clusters = n_clusters
            self.labels_ = None
            self.cluster_centers_ = None

        def fit_predict(self, X):
            labels = np.arange(len(X)) % self.n_clusters
            self.labels_ = labels.astype(int)
            self.cluster_centers_ = np.vstack(
                [X[self.labels_ == cluster_id].mean(axis=0) for cluster_id in range(self.n_clusters)]
            )
            return self.labels_

        def fit(self, X):
            self.fit_predict(X)
            return self

        def transform(self, X):
            return np.stack(
                [np.linalg.norm(X - center, axis=1) for center in self.cluster_centers_],
                axis=1,
            )

    monkeypatch.setattr(clustering_utils, "CACHE_PATH", str(tmp_path / "cluster_cache.parquet"))
    monkeypatch.setattr(clustering_utils, "MODEL_PATH", str(tmp_path / "kmeans_model.joblib"))
    monkeypatch.setattr(clustering_utils, "KMeansScratch", DummyKMeans)
    monkeypatch.setattr(clustering_utils, "save_cache", lambda *args, **kwargs: None)

    clustered_df, _, _, pca = clustering_utils.get_clustered_data(df, user_history={}, k=2, force=True)

    for column in ["cluster_view_x", "cluster_view_y", "cluster_view_z", "pca_x", "pca_y", "pca_z"]:
        assert column in clustered_df.columns
        assert np.isfinite(clustered_df[column]).all()
    assert len(pca.axis_labels_) == 3
    assert len(pca.component_summaries_) == 3


def test_semantic_search_uses_cuisine_type_for_location_and_cuisine_queries(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "dba": "Pasta Corner",
                "description": "Fresh pasta dishes in Brooklyn with handmade noodles",
                "cuisine_type": "Italian",
                "boro": "Brooklyn",
                "neighborhood": "Williamsburg",
                "zipcode": "11211",
                "g_summary": "Neighborhood pasta favorite",
                "g_rating": 4.5,
                "score": 8,
                "grade": "A",
            }
        ]
    )

    monkeypatch.setattr(search_utils, "load_model", lambda: None)
    monkeypatch.setattr(
        search_utils,
        "score_restaurants_for_user",
        lambda frame, profile: pd.DataFrame({"preference_score": np.zeros(len(frame))}, index=frame.index),
    )

    results = search_utils.semantic_search(
        query="Italian pasta Brooklyn",
        df=df,
        embeddings=None,
        top_k=5,
        boro_filter="All",
        grade_filter="All",
        min_rating=0,
        profile={},
    )

    assert len(results) == 1
    assert results.iloc[0]["dba"] == "Pasta Corner"


def test_mlp_forward_sanitizes_nan_inputs():
    model = CustomMLP(input_dim=4, hidden_dim=8, output_dim=3)
    X = torch.tensor([[1.0, float("nan"), 2.0, 3.0]], dtype=torch.float32)

    outputs = model(X)

    assert torch.isfinite(outputs).all()
