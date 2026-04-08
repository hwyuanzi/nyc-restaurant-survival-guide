"""
utils/search.py — Semantic search engine for NYC restaurants.

Combines sentence-transformer embeddings, lexical keyword overlap, quality
scoring, and user profile scoring into a blended ranking function.

Original author: Rahul Adusumalli
Integrated by: Ryan Han (PapTR)
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from utils.user_profile import build_profile_prompt, score_restaurants_for_user

CUISINE_KEYWORDS = {
    "American": "burgers steaks BBQ comfort food fries classic American diner",
    "Italian": "pasta pizza risotto tiramisu antipasto gelato wine",
    "Chinese": "dim sum dumplings noodles stir fry Peking duck Cantonese Sichuan",
    "Japanese": "sushi ramen sashimi tempura udon miso izakaya omakase",
    "Mexican": "tacos burritos enchiladas guacamole margaritas salsa tamales",
    "French": "croissants baguette coq au vin escargot crepes bistro wine",
    "Indian": "curry tandoori naan biryani masala daal spices South Asian",
    "Thai": "pad thai green curry satay lemongrass coconut milk noodles",
    "Korean": "bibimbap kimchi bulgogi Korean BBQ tofu banchan galbi",
    "Vietnamese": "pho banh mi spring rolls lemongrass rice noodle soup bun",
    "Mediterranean": "hummus falafel gyros shawarma olive oil mezze halloumi",
    "Latin": "empanadas rice beans plantains ceviche Latin fusion churros",
    "Cafe": "coffee espresso brunch pastries sandwiches light fare",
    "Pizza": "New York slice thin crust mozzarella tomato sauce calzone",
    "Seafood": "lobster oysters shrimp clams crab fresh fish grilled seafood",
    "Bakery": "bread pastries croissants cakes artisan baked goods sourdough",
    "Steak": "steakhouse prime rib tenderloin ribeye porterhouse dry-aged beef",
    "Vegetarian": "vegan plant-based tofu salads grain bowls organic healthy",
    "Middle Eastern": "kebab hummus falafel tahini shawarma pita manakeesh",
    "Spanish": "tapas paella sangria chorizo jamon iberico pintxos",
    "Juice Bar": "smoothies cold-pressed juice acai bowls healthy drinks",
    "Ice Cream": "gelato soft serve sundaes sorbet frozen desserts waffle cone",
    "Sandwiches": "subs hoagies paninis wraps deli club sandwich hero",
    "Chicken": "fried chicken wings rotisserie grilled chicken nuggets",
    "Hamburgers": "burgers smash burger cheeseburger double patty gourmet",
    "Caribbean": "jerk chicken rice peas plantains oxtail roti callaloo",
    "Ethiopian": "injera berbere lentils stew communal eating East African",
    "Greek": "souvlaki moussaka spanakopita tzatziki dolmades feta",
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMBEDDING_CACHE_DIR = DATA_DIR / "cache"


def stars(rating):
    if not rating:
        return ""
    full = int(float(rating))
    half = 1 if (float(rating) - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def price_label(tier):
    return {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(tier, "")


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception:
        return None


def build_description(row):
    name = row.get("dba", "")
    cuisine = row.get("cuisine", "")
    boro = row.get("boro", "")
    grade = row.get("grade", "")
    score = row.get("score", 0)
    address = row.get("address", "")
    extras = CUISINE_KEYWORDS.get(cuisine, cuisine)
    summary = row.get("g_summary", "")
    rating = row.get("g_rating")
    price = row.get("g_price")
    reviews = row.get("g_reviews", 0)
    hygiene = {"A": "excellent hygiene", "B": "good hygiene", "C": "acceptable hygiene"}.get(grade, "")
    rating_str = f"Google rating {rating}/5 based on {int(reviews):,} reviews." if rating and reviews else ""
    price_str = f"Price level: {price_label(price)}." if price else ""

    return (
        f"{name} is a {cuisine} restaurant in {boro}, New York City. "
        f"It serves {extras}. "
        f"{summary} "
        f"{rating_str} {price_str} "
        f"Health inspection grade: {grade} ({hygiene}, score {score}). "
        f"Address: {address}."
    ).strip()


def _ensure_cache_dir():
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(ttl=86400, show_spinner=False)
def compute_embeddings(df):
    model = load_model()
    if model is None:
        return None
    return model.encode(df["description"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def get_embeddings(df, cache_key):
    _ensure_cache_dir()
    cache_file = EMBEDDING_CACHE_DIR / f"embeddings_{cache_key}.npy"
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except Exception:
            pass

    embeddings = compute_embeddings(df)
    if embeddings is not None:
        np.save(cache_file, embeddings)
    return embeddings


def lexical_score(query, text):
    query_tokens = {token for token in str(query).lower().split() if len(token) > 2}
    text_tokens = {token for token in str(text).lower().split() if len(token) > 2}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def semantic_search(query, df, embeddings, top_k, boro_filter, grade_filter, min_rating, profile=None):
    profile = profile or {}
    profile_text = build_profile_prompt(profile)
    expanded_query = f"{query}. {profile_text}".strip()

    semantic_scores = np.zeros(len(df))
    model = load_model()
    if embeddings is not None and model is not None:
        try:
            query_embedding = model.encode([expanded_query], normalize_embeddings=True)
            semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
        except Exception:
            semantic_scores = np.zeros(len(df))

    semantic_norm = np.clip((semantic_scores + 1) / 2, 0, 1)
    lexical_scores = df["description"].fillna("").apply(lambda text: lexical_score(query, text)).to_numpy()
    quality_scores = (
        0.5 * pd.to_numeric(df.get("g_rating", 3.4), errors="coerce").fillna(3.4).to_numpy() / 5
        + 0.3 * (1 - pd.to_numeric(df.get("score", 21), errors="coerce").fillna(21).clip(0, 42).to_numpy() / 42)
        + 0.2 * df["grade"].map({"A": 1.0, "B": 0.78, "C": 0.58}).fillna(0.6).to_numpy()
    )
    profile_scores = score_restaurants_for_user(df, profile)["preference_score"].to_numpy() / 10

    mask = np.ones(len(df), dtype=bool)
    if boro_filter != "All":
        mask &= df["boro"].str.lower() == boro_filter.lower()
    if grade_filter != "All":
        mask &= df["grade"] == grade_filter
    if min_rating > 0:
        mask &= pd.to_numeric(df.get("g_rating", 0), errors="coerce").fillna(0) >= min_rating

    filtered_scores = np.where(
        mask,
        0.5 * semantic_norm + 0.2 * lexical_scores + 0.15 * quality_scores + 0.15 * profile_scores,
        -1.0,
    )
    top_indices = np.argsort(filtered_scores)[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["similarity"] = filtered_scores[top_indices]
    results["match_percent"] = np.round(results["similarity"] * 100).astype(int)
    return results[results["similarity"] > 0].reset_index(drop=True)
