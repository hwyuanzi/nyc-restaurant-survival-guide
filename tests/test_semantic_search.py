import numpy as np
import pandas as pd

from utils import search


def _toy_restaurants():
    return pd.DataFrame(
        [
            {
                "dba": "Sakura Sushi",
                "description": "Japanese sushi omakase restaurant in Manhattan with fresh fish.",
                "cuisine": "Japanese",
                "boro": "Manhattan",
                "neighborhood": "Midtown East",
                "zipcode": "10017",
                "g_summary": "Sushi counter",
                "g_rating": 4.7,
                "g_price": 3,
                "grade": "A",
                "score": 8,
            },
            {
                "dba": "Sunset Tacos",
                "description": "Mexican tacos and burritos in Brooklyn.",
                "cuisine": "Mexican",
                "boro": "Brooklyn",
                "neighborhood": "Park Slope",
                "zipcode": "11215",
                "g_summary": "Casual taco shop",
                "g_rating": 4.4,
                "g_price": 1,
                "grade": "A",
                "score": 10,
            },
            {
                "dba": "Neighborhood Coffee",
                "description": "Coffee, pastries, and quick breakfast in Queens.",
                "cuisine": "Coffee/Tea",
                "boro": "Queens",
                "neighborhood": "Astoria",
                "zipcode": "11103",
                "g_summary": "Cafe",
                "g_rating": 4.1,
                "g_price": 1,
                "grade": "B",
                "score": 18,
            },
        ]
    )


def test_lexical_score_counts_meaningful_tokens():
    assert search.lexical_score("cozy sushi dinner", "fresh sushi dinner counter") > 0
    assert search.lexical_score("cozy sushi dinner", "coffee and pastries") == 0


def test_semantic_search_structured_fallback_without_model(monkeypatch):
    monkeypatch.setattr(search, "load_model", lambda: None)
    results = search.semantic_search(
        "sushi in Manhattan",
        _toy_restaurants(),
        embeddings=None,
        top_k=2,
        boro_filter="All",
        grade_filter="All",
        min_rating=0.0,
        profile={},
        min_match=0.1,
    )

    assert not results.empty
    assert results.iloc[0]["dba"] == "Sakura Sushi"
    assert (results["boro"] == "Manhattan").all()


def test_semantic_search_respects_borough_filter(monkeypatch):
    monkeypatch.setattr(search, "load_model", lambda: None)
    results = search.semantic_search(
        "tacos",
        _toy_restaurants(),
        embeddings=np.empty((0, 0)),
        top_k=3,
        boro_filter="Brooklyn",
        grade_filter="All",
        min_rating=0.0,
        profile={},
        min_match=0.1,
    )

    assert len(results) == 1
    assert results.iloc[0]["dba"] == "Sunset Tacos"
