import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.ui_utils import apply_apple_theme

from utils.google_places import build_photo_url, get_google_api_key
from utils.search import price_label, semantic_search, stars
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import add_liked_restaurant, init_session_state, render_profile_sidebar

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

st.set_page_config(
    page_title="NYC Restaurant Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_apple_theme()
init_session_state()


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
            if st.button("Like this restaurant", key=f"home_like_{rank}_{row.get('camis', name)}"):
                if add_liked_restaurant(profile_name, row, source="home_search"):
                    st.success("Saved to your profile.")
                else:
                    st.info("Already saved in your profile.")
    st.divider()


with st.sidebar:
    profile = render_profile_sidebar()
    st.markdown("---")
    st.title("🔎 Search Filters")
    boro_filter = st.selectbox("Borough", ["All", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    grade_filter = st.selectbox("Health Grade", ["All", "A", "B", "C"])
    min_rating = st.slider("Min Google Rating", 0.0, 5.0, 3.5, 0.5)
    top_k = st.slider("Results to show", 3, 20, 8)
    force_refresh = st.checkbox("Rebuild prepared search cache", value=False)

st.title("🍽️ NYC Restaurant Recommender")
st.markdown("Find restaurants in New York City using personalized recommendations, semantic search, and a 3D Manhattan preference map.")

hero_left, hero_right = st.columns([3, 2])
with hero_left:
    st.subheader("What This App Does")
    st.markdown(
        """
        This app combines NYC inspection data, Google Places enrichment, and a persistent user profile to help each user discover restaurants they are more likely to enjoy.

        It includes:
        - Personalized recommendations scored from `1-10`
        - A semantic search page that understands cravings and occasions
        - A 3D Manhattan tower map showing clusters of strong matches
        - A profile survey plus saved likes so recommendations improve over time
        """
    )
with hero_right:
    st.info(
        """
        **Best flow**

        1. Fill out your profile survey in the sidebar on any page
        2. Visit **Recommendations** for your baseline picks
        3. Use the search box below for specific cravings
        4. Explore **3D Manhattan Map** for where your best-fit options cluster
        """
    )

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("🔮 Recommendations")
    st.write("See a predicted enjoyment score for each restaurant based on your survey answers, likes, boroughs, cuisines, budget, and quality signals.")
with col2:
    st.subheader("🔎 Semantic Search")
    st.write("Describe what you want in plain English and get search results ranked using embeddings, keyword overlap, restaurant quality, and your user profile.")
with col3:
    st.subheader("📍 3D Manhattan Map")
    st.write("View tower heights across Manhattan, where taller columns represent more restaurants above your personal `7/10` threshold.")

st.markdown("---")
st.subheader("Performance Improvements")
st.write(
    "Search data is now prepared into local disk artifacts in `data/cache`, including Google-enriched restaurants, descriptions, and embeddings, so later app launches can open directly into a ready-to-search state."
)

api_key = get_google_api_key()
with st.spinner("Preparing search data for this app session..."):
    prepared_df, embeddings, runtime_df, cache_info = load_runtime_assets(
        sample_size=DEFAULT_SEARCH_SAMPLE_SIZE,
        force_refresh=force_refresh,
    )
    st.session_state["raw_df"] = runtime_df

if prepared_df.empty:
    st.error("Search data could not be prepared. Check your network connection or Google Places quota.")
    st.stop()

cluster_filter_label = "All Clusters"
clustered_df = st.session_state.get("clustered_df")
if clustered_df is not None and not clustered_df.empty and "cluster_label" in clustered_df.columns:
    cluster_options = ["All Clusters"] + sorted(clustered_df["cluster_label"].dropna().unique().tolist())
    preselected_label = st.session_state.get("selected_cluster_label", "All Clusters")
    selected_index = cluster_options.index(preselected_label) if preselected_label in cluster_options else 0
    with st.sidebar:
        st.markdown("---")
        cluster_filter_label = st.selectbox("Taste Cluster", cluster_options, index=selected_index)
    st.session_state["selected_cluster_label"] = cluster_filter_label
    if cluster_filter_label != "All Clusters":
        valid_ids = clustered_df.loc[clustered_df["cluster_label"] == cluster_filter_label, "restaurant_id"].astype(str)
        prepared_df = prepared_df[prepared_df["camis"].astype(str).isin(valid_ids)].reset_index(drop=True)
        runtime_df = runtime_df[runtime_df["restaurant_id"].isin(valid_ids)].reset_index(drop=True)
        st.session_state["raw_df"] = runtime_df

if prepared_df.empty or "description" not in prepared_df.columns:
    st.warning("No restaurants are available for the current filters yet. Try clearing the cluster filter or rebuilding the search cache.")
    st.stop()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Restaurants Loaded", f"{cache_info.get('base_rows', len(prepared_df)):,}")
status_col2.metric("Google-Enriched Rows", f"{cache_info.get('enriched_rows', 0):,}")
status_col3.metric("Embeddings", "Ready" if embeddings is not None else "Lexical fallback")

st.caption("Use the page navigation in the sidebar to jump into Recommendations, Semantic Search, PCA Explorer, or the 3D Manhattan Map.")

st.markdown("---")
st.subheader("Search From Home")

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
    st.info("Enter a search query above to search immediately from the landing page.")
    st.stop()

results = semantic_search(
    query,
    prepared_df,
    embeddings,
    top_k=top_k,
    boro_filter=boro_filter,
    grade_filter=grade_filter,
    min_rating=min_rating,
    profile=profile,
)

if results.empty:
    st.info("No results found. Try adjusting your filters or rephrasing your query.")
else:
    st.success(f"Found {len(results)} matches for *{query}*")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_card(row.to_dict(), api_key, profile["name"], rank)
