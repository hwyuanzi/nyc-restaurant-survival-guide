import numpy as np
import pandas as pd

from utils.clustering import liked_history_cuisine_boost


def test_liked_history_cuisine_boost_prefers_dominant_exact_cuisine():
    cuisines = pd.Series(["Korean", "Chinese", "Thai", "Japanese", "Korean"])
    liked_metadata = [
        {"cuisine": "Korean"},
        {"cuisine": "Korean"},
        {"cuisine": "Korean"},
    ]

    boost = liked_history_cuisine_boost(cuisines, liked_metadata)

    assert np.all(boost[cuisines == "Korean"] == 3.0)
    assert np.all(boost[cuisines != "Korean"] == 0.25)


def test_liked_history_cuisine_boost_blends_when_no_dominant_cuisine():
    cuisines = pd.Series(["Korean", "Chinese", "Thai", "Japanese"])
    liked_metadata = [
        {"cuisine": "Korean"},
        {"cuisine": "Chinese"},
    ]

    boost = liked_history_cuisine_boost(cuisines, liked_metadata, dominant_threshold=0.75)

    assert boost[0] > 1.0
    assert boost[1] > 1.0
    assert boost[2] == 1.0
    assert boost[3] == 1.0
