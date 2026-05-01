import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

from app.ui_utils import apply_apple_theme

from utils.google_places import build_photo_url, get_google_api_key
from utils.search import price_label, semantic_search, stars
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import add_liked_restaurant, get_active_profile, init_session_state

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

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
            st.image(build_photo_url(photo_ref, api_key), width="stretch")
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

        if maps_url:
            st.markdown(f"[📍 Open in Google Maps]({maps_url})")

        from utils.user_profile import is_restaurant_liked, remove_liked_restaurant
        if is_restaurant_liked(profile_name, row):
            if st.button("❤️ Unlike this restaurant", key=f"search_unlike_{rank}_{row.get('camis', name)}"):
                if remove_liked_restaurant(profile_name, row):
                    st.success("Removed from your profile.")
                    st.rerun()
        else:
            if st.button("🤍 Like this restaurant", key=f"search_like_{rank}_{row.get('camis', name)}"):
                if add_liked_restaurant(profile_name, row, source="semantic_search"):
                    st.success("Saved to your profile.")
                    st.rerun()
    st.divider()


profile = get_active_profile()
refresh_cache = False
boro_filter = "All"
grade_filter = "All"
min_rating = 0.0
min_match = 0.45
top_k = 8

st.title("🔎 Semantic Restaurant Search")
st.markdown("Describe a craving, vibe, or occasion. Results are personalized using your saved profile and likes.")

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

clustered_df = st.session_state.get("clustered_df")
if clustered_df is not None and not clustered_df.empty and "cluster_label" in clustered_df.columns:
    st.session_state["selected_cluster_label"] = "All Clusters"

if enriched_df.empty or "description" not in enriched_df.columns:
    st.warning("No restaurants are available for the current filters yet. Try clearing the cluster filter or rebuilding the cache.")
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
    query,
    enriched_df,
    embeddings,
    top_k=top_k,
    boro_filter=boro_filter,
    grade_filter=grade_filter,
    min_rating=min_rating,
    profile=profile,
    min_match=min_match,
)

if results.empty:
    st.info("No strong matches found. Try rephrasing your query.")
else:
    st.success(f"Found {len(results)} matches for *{query}*")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_card(row.to_dict(), api_key, profile["name"], rank)
