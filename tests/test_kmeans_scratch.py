import numpy as np

from models.kmeans_scratch import KMeansScratch


def test_kmeans_scratch_separates_two_obvious_clusters():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ]
    )

    model = KMeansScratch(n_clusters=2, n_init=5, max_iter=100, random_state=7)
    labels = model.fit_predict(X)

    assert labels.shape == (6,)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert model.cluster_centers_.shape == (2, 2)
    assert model.inertia_ >= 0


def test_kmeans_predict_and_transform_shapes():
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [4.8, 5.0],
            [5.2, 5.1],
        ]
    )
    model = KMeansScratch(n_clusters=2, n_init=4, random_state=11)
    model.fit_predict(X)

    new_points = np.array([[0.1, 0.2], [5.1, 4.9]])
    predicted = model.predict(new_points)
    distances = model.transform(new_points)

    assert predicted.shape == (2,)
    assert distances.shape == (2, 2)
    assert np.all(distances >= 0)


def test_kmeans_handles_duplicate_points_without_nan_initialization():
    X = np.ones((5, 3))
    model = KMeansScratch(n_clusters=2, n_init=2, random_state=13)
    labels = model.fit_predict(X)

    assert labels.shape == (5,)
    assert model.cluster_centers_.shape == (2, 3)
    assert np.isfinite(model.cluster_centers_).all()
    assert np.isfinite(model.inertia_)
