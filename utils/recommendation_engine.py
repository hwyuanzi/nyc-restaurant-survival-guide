import numpy as np
import pandas as pd

from utils.user_profile import score_restaurants_for_user


def token_overlap_score(query, text):
    query_tokens = {token for token in str(query).lower().split() if len(token) > 2}
    text_tokens = {token for token in str(text).lower().split() if len(token) > 2}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def quality_score(df):
    if "g_rating" in df:
        rating_source = df["g_rating"]
    else:
        rating_source = pd.Series([3.4] * len(df), index=df.index)

    if "g_reviews" in df:
        reviews_source = df["g_reviews"]
    else:
        reviews_source = pd.Series([0] * len(df), index=df.index)

    grade_score = df["grade"].map({"A": 1.0, "B": 0.78, "C": 0.58}).fillna(0.6)
    inspection_score = 1 - df["score"].fillna(21).clip(0, 42) / 42
    rating_score = pd.to_numeric(rating_source, errors="coerce").fillna(3.4) / 5
    reviews = pd.to_numeric(reviews_source, errors="coerce").fillna(0)
    popularity = np.clip(np.log1p(reviews) / np.log1p(4000), 0, 1)
    return 0.35 * grade_score + 0.25 * inspection_score + 0.3 * rating_score + 0.1 * popularity


def personalized_recommendations(df, profile, query="", count=8):
    if df.empty:
        return df.copy()

    scored = score_restaurants_for_user(df, profile)
    lexical = scored["description"].fillna("").apply(lambda text: token_overlap_score(query, text)) if "description" in scored else 0
    quality = quality_score(scored)

    if isinstance(lexical, int):
        lexical = pd.Series(np.zeros(len(scored)), index=scored.index)

    scored["final_score_raw"] = (
        0.72 * (scored["preference_score"] / 10)
        + 0.16 * lexical
        + 0.12 * quality
    )
    scored["recommendation_score"] = np.clip(np.round(1 + 9 * scored["final_score_raw"], 1), 1, 10)
    return (
        scored.sort_values(["recommendation_score", "preference_score", "dba"], ascending=[False, False, True])
        .drop_duplicates(subset=["dba", "address"])
        .head(count)
        .reset_index(drop=True)
    )
