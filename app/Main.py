import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.ui_utils import apply_apple_theme

from utils.google_places import build_photo_url, get_google_api_key
from utils.search import price_label, semantic_search, stars
from utils.search_assets import DEFAULT_SEARCH_SAMPLE_SIZE, load_runtime_assets
from utils.user_profile import add_liked_restaurant, get_active_profile, init_session_state

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

from utils.auth import authenticate_user, register_user

if "authenticated_profile_id" not in st.session_state:
    st.session_state["authenticated_profile_id"] = None

if not st.session_state["authenticated_profile_id"]:
    st.title("🍽️ NYC Restaurant Survival Guide")
    st.markdown("Please log in or create an account to start searching and saving your favorite restaurants.")

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", width="stretch"):
            success, profile_id = authenticate_user(login_username, login_password)
            if success:
                st.session_state["authenticated_profile_id"] = profile_id
                st.rerun()
            else:
                st.error("Invalid username or password. (If you have an old account without a password, try logging in with an empty password.)")

    with tab_signup:
        signup_username = st.text_input("Choose a Username", key="signup_username")
        signup_password = st.text_input("Choose a Password", type="password", key="signup_password")
        if st.button("Sign Up", width="stretch"):
            success, result = register_user(signup_username, signup_password)
            if success:
                st.success("Account created! Logging you in...")
                st.session_state["authenticated_profile_id"] = result
                st.rerun()
            else:
                st.error(result)
    st.stop()


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
            if st.button("❤️ Unlike this restaurant", key=f"home_unlike_{rank}_{row.get('camis', name)}"):
                if remove_liked_restaurant(profile_name, row):
                    st.success("Removed from your profile.")
                    st.rerun()
        else:
            if st.button("🤍 Like this restaurant", key=f"home_like_{rank}_{row.get('camis', name)}"):
                if add_liked_restaurant(profile_name, row, source="home_search"):
                    st.success("Saved to your profile.")
                    st.rerun()
    st.divider()


with st.sidebar:
    profile = get_active_profile()
    st.title(f"👤 Welcome, {profile.get('name', 'Guest')}")
    if st.button("Logout", width="stretch", key="main_logout"):
        st.session_state["authenticated_profile_id"] = None
        st.rerun()

boro_filter = "All"
grade_filter = "All"
min_rating = 0.0
top_k = 8
force_refresh = False

st.title("🍽️ NYC Restaurant Survival Guide")
st.caption("CSCI-UA 473 · Fundamentals of Machine Learning · NYU Spring 2026")
st.markdown(
    "An end-to-end ML dashboard combining NYC Department of Health inspection data, "
    "Google Places metadata, semantic search, a health-risk classifier, interpretable clustering, "
    "PCA visualization, and liked-history recommendations."
)

hero_left, hero_right = st.columns([3, 2])
with hero_left:
    st.subheader("What's Inside")
    st.markdown(
        """
        **Data** — 14,252 NYC DOHMH inspection records (train + test) enriched with 2,835 Google Places rows including ratings, reviews, price tier, and photos.

        **Course ML implementation:**
        - K-Means++ (`models/kmeans_scratch.py`) — default clustering engine
        - Custom MLP (`models/custom_mlp.py`) — 3-layer PyTorch health grade classifier
        - PCA visualizations — cluster interpretation and feature loading analysis

        **Retrieval & recommendations:**
        - Semantic search via `sentence-transformers/all-mpnet-base-v2` + cosine similarity
        - Personalized recommendations with per-liked KNN → Reciprocal Rank Fusion → MMR reranking
        """
    )
with hero_right:
    st.info(
        """
        **Suggested flow**

        1. Like restaurants you enjoy from search results or the Semantic Search page
        2. Visit **Recommendations** for liked-history picks
        3. Open **Health Grade Classifier** to explore inspection risk signals
        4. Browse **Restaurant Cluster GIS Map** to see learned NYC segments
        5. Dive into **PCA Embedding Explorer** for cluster interpretation
        """
    )

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("**🔍 Semantic Search**")
    st.caption("Natural-language restaurant search using transformer embeddings and cosine similarity, with cuisine and neighborhood query hints.")
with col2:
    st.markdown("**🧪 Health Classifier**")
    st.caption("Select a held-out DOHMH restaurant, view A/B/C risk probabilities, edit inspection features, and find a path toward Grade A.")
with col3:
    st.markdown("**📍 Cluster Map**")
    st.caption("K-Means from scratch on an 18-D interpretable feature space (price, rating, cuisine, borough, location). K=9 learned clusters on a real NYC map.")
with col4:
    st.markdown("**📊 PCA Explorer**")
    st.caption("3D PCA, centroid-distance view, and t-SNE; feature loadings, explained variance, cluster distance matrix, and prototype restaurants.")
with col5:
    st.markdown("**🔮 Recommendations**")
    st.caption("Like restaurants from any page, then get personalized picks via per-liked KNN + RRF + MMR diversity reranking.")

st.markdown("---")
st.subheader("Runtime Data")
st.write(
    "Committed caches in `data/cache/` — Google-enriched restaurant table, 768-D embeddings, "
    "K-Means model, and health classifier checkpoint — let the app run offline without rebuilding anything. "
    "The status panel below shows what's loaded for this session."
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
    st.session_state["selected_cluster_label"] = "All Clusters"

if prepared_df.empty or "description" not in prepared_df.columns:
    st.warning("No restaurants are available for the current filters yet. Try reloading the search cache.")
    st.stop()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Restaurants Loaded", f"{cache_info.get('base_rows', len(prepared_df)):,}")
status_col2.metric("Google-Enriched Rows", f"{cache_info.get('enriched_rows', 0):,}")
status_col3.metric("Embeddings", "Ready" if embeddings is not None else "Lexical fallback")

st.caption("Use the page navigation in the sidebar to jump into Recommendations, Semantic Search, PCA Explorer, or the Restaurant Cluster GIS Map.")

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
    st.info("No results found. Try rephrasing your query.")
else:
    st.success(f"Found {len(results)} matches for *{query}*")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        render_card(row.to_dict(), api_key, profile["name"], rank)
