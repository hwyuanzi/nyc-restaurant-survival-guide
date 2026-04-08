"""
6_🔎_Live_Semantic_Search.py — Real-data semantic search with Google Places enrichment.

Unlike the synthetic corpus in page 1, this page searches across real NYC
restaurants enriched with Google Places data (ratings, photos, reviews).
Results are personalized using the user's saved profile and likes.

Original search pipeline: Rahul Adusumalli
Embedding model: sentence-transformers/all-mpnet-base-v2 (768-D)
Integrated by: Ryan Han (PapTR)

Course topics: Week 3 (Transformers), Week 4 (Similarity, Nearest Neighbor)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from utils.google_places import build_photo_url, get_google_api_key
from utils.search import price_label, semantic_search, stars
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import add_liked_restaurant, init_session_state, render_profile_sidebar

st.set_page_config(page_title="Live Semantic Search", page_icon="🔎", layout="wide")

from app.ui_utils import apply_apple_theme
apply_apple_theme()

init_session_state()

EXAMPLES = [
    "cozy Italian pasta spot in Brooklyn",
    "late night ramen and dumplings Manhattan",
    "healthy vegan grain bowls",
    "romantic French bistro with good wine",
    "authentic Mexican street tacos Queens",
    "spicy Korean BBQ grill",
    "fresh sushi omakase experience",
    "Sunday brunch with mimosas",
    "cheap and delicious Caribbean food Bronx",
    "upscale seafood and oyster bar",
]


def render_card(row, api_key, profile_name, rank):
    pct = int(row.get("match_percent", round(row["similarity"] * 100)))
    grade = row.get("grade", "N/A")
    rating = row.get("g_rating")
    reviews = row.get("g_reviews")
    price = row.get("g_price")
    maps_url = row.get("g_maps_url", "")
    photo_ref = row.get("g_photo_ref", "")
    cuisine = row.get("cuisine", "")
    boro = row.get("boro", "")
    address = row.get("address", "")
    name = row.get("dba", "Unknown")

    col_img, col_info = st.columns([1, 3])
    with col_img:
        if photo_ref and api_key:
            st.image(build_photo_url(photo_ref, api_key), use_container_width=True)
        else:
            st.markdown("🍽️")

    with col_info:
        st.markdown(f"### {name}")
        st.caption(f"{cuisine} · {boro} · Grade {grade}")
        st.caption(f"📍 {address}")

        if rating:
            review_text = f"({int(reviews):,} reviews)" if reviews else ""
            st.markdown(f"{stars(rating)} **{float(rating):.1f}/5** {review_text}")

        tags = [tag for tag in [cuisine, boro, price_label(price)] if tag]
        if tags:
            st.markdown(" · ".join(f"`{tag}`" for tag in tags))

        description = row.get("description", "")
        if len(description) > 320:
            description = description[:317] + "..."
        st.write(description)

        metric_col, action_col = st.columns([1, 2])
        with metric_col:
            st.metric("Match", f"{pct}%")
        with action_col:
            if maps_url:
                st.markdown(f"[📍 Open in Google Maps]({maps_url})")
            if st.button("Like this restaurant", key=f"search_like_{rank}_{row.get('camis', name)}"):
                if add_liked_restaurant(profile_name, row, source="live_search"):
                    st.success("Saved to your profile.")
                else:
                    st.info("Already saved in your profile.")
    st.divider()


with st.sidebar:
    profile = render_profile_sidebar()
    st.markdown("---")
    st.title("🔎 Search Settings")
    refresh_cache = st.checkbox("Refresh cached Google Places sample", value=False)
    boro_filter = st.selectbox("Borough", ["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    grade_filter = st.selectbox("Health Grade", ["All", "A", "B", "C"])
    min_rating = st.slider("Min Google Rating", 0.0, 5.0, 3.5, 0.5)
    top_k = st.slider("Results to show", 3, 20, 8)

st.title("🔎 Live Semantic Restaurant Search")
st.markdown("""
Search **real NYC restaurants** enriched with Google Places data.
Results are personalized using your saved profile, likes, and a blended ranking of
semantic similarity (50%), lexical overlap (20%), quality (15%), and profile match (15%).
""")

api_key = get_google_api_key()
with st.spinner("Loading prepared search data..."):
    enriched_df, embeddings, runtime_df, cache_info = load_runtime_assets(
        sample_size=DEFAULT_SEARCH_SAMPLE_SIZE,
        force_refresh=refresh_cache,
    )
    st.session_state["raw_df"] = runtime_df

if enriched_df.empty:
    st.error("No restaurants returned valid Google Places data. Check your API key and quota.")
    st.stop()

if enriched_df.empty or "description" not in enriched_df.columns:
    st.warning("No restaurants available. Try rebuilding the cache.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Prepared Restaurants", f"{cache_info.get('rows', len(enriched_df)):,}")
col2.metric("Embedding Model", "all-mpnet-base-v2" if embeddings is not None else "Lexical only")
col3.metric("Saved Likes", len(profile.get("likes", [])))

col_q, col_examples = st.columns([3, 2])
with col_q:
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. date night sushi in Manhattan with great reviews",
    ).strip()
with col_examples:
    example_query = st.selectbox("Or try an example:", [""] + EXAMPLES, format_func=lambda value: value or "Select an example")

if example_query:
    query = example_query

if not query:
    st.info("Enter a search query to see results.")
    st.stop()

results = semantic_search(
    query, enriched_df, embeddings,
    top_k=top_k, boro_filter=boro_filter, grade_filter=grade_filter,
    min_rating=min_rating, profile=profile,
)

if results.empty:
    st.info("No results found. Try adjusting your filters or rephrasing your query.")
else:
    st.success(f"Found {len(results)} matches for *{query}*")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_card(row.to_dict(), api_key, profile["name"], rank)
