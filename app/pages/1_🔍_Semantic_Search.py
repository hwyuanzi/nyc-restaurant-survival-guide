import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.ui_utils import apply_apple_theme

st.set_page_config(page_title="Semantic Vibe Search", page_icon="🔍", layout="wide")
apply_apple_theme()

st.title("🔍 Semantic Vibe Search")
st.markdown("""
### 🎯 What is this?
A **natural language search engine** for NYC restaurants powered by **HuggingFace Transformers**.

### 💡 How to use it?
Describe the *vibe* you want — e.g., *"cozy romantic Italian dim lighting"* or *"cheap loud street food late night"*.
The PyTorch NLP model embeds your sentence into a **384-dimensional vector** and returns restaurants ranked by **Cosine Similarity** against a corpus of **200+ NYC restaurants**. No keywords — pure math!
---
""")

# ─────────────────────────────────────────────
# 1. GENERATE A LARGE, DIVERSE SYNTHETIC CORPUS
# ─────────────────────────────────────────────
@st.cache_data
def build_large_corpus(n=220, seed=42):
    """Programmatically generates a diverse 200+ NYC restaurant corpus."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Templates by cuisine/vibe — mix-and-match generates unique sentences
    templates = {
        "Italian": [
            "{adj} Italian trattoria in {boro}. Known for hand-made {pasta}, {atm}. {extra}",
            "Classic red-sauce Italian joint in {boro}. {adj}, {atm}, with a great wine list. {extra}",
            "Modern Italian osteria in {boro} focusing on {pasta} and seasonal ingredients. {adj} and {atm}. {extra}",
        ],
        "Japanese": [
            "{adj} Japanese restaurant in {boro} famous for {jp_dish}. {atm}. {extra}",
            "Cozy ramen shop in {boro}. Slow-simmered broth, perfect noodles, {atm}. {extra}",
            "Omakase sushi counter in {boro}. {adj} experience with {jp_dish}. {atm}. {extra}",
        ],
        "Mexican": [
            "Authentic Mexican taqueria in {boro}. {adj} tacos, {atm}. {extra}",
            "Vibrant {boro} Mexican spot with handmade tortillas and mezcal cocktails. {adj} and lively. {extra}",
            "Street-food style Mexican in {boro}. {adj}, {atm}, with bold flavors. {extra}",
        ],
        "Chinese": [
            "{adj} Chinese restaurant in {boro} featuring {cn_dish}. {atm}. {extra}",
            "Dim sum hall in {boro}. Bustling, loud, and packed on weekends. {adj} food. {extra}",
            "Sichuan specialist in {boro} known for numbing spicy flavors and {cn_dish}. {atm}. {extra}",
        ],
        "American": [
            "{adj} American diner in {boro}. Classic burgers, fries, and milkshakes. {atm}. {extra}",
            "Farm-to-table American bistro in {boro}. Seasonal menu, {adj} atmosphere. {extra}",
            "Old-school American steakhouse in {boro}. Dry-aged beef, {adj} and {atm}. {extra}",
        ],
        "French": [
            "Charming French bistro in {boro}. {adj} ambiance, steak frites, and crème brûlée. {extra}",
            "{adj} French brasserie in {boro}. Elegant, {atm}, with classic Gallic cooking. {extra}",
            "Intimate French fine dining in {boro}. {adj}, hushed, with an extraordinary tasting menu. {extra}",
        ],
        "Indian": [
            "Aromatic Indian restaurant in {boro} with rich curries and tandoor breads. {adj} and {atm}. {extra}",
            "Modern Indian cuisine in {boro}. Inventive spice blends, {adj} decor. {extra}",
            "Casual Indian street food stall in {boro}. Cheap, flavorful, and {atm}. {extra}",
        ],
        "Thai": [
            "Lively Thai restaurant in {boro}. Fragrant curries, pad thai, {adj} atmosphere. {extra}",
            "Authentic Thai in {boro} with {adj} flavors and a great vegetarian menu. {extra}",
        ],
        "Korean": [
            "Korean BBQ in {boro}. Tabletop grills, marinated meats, {adj} and {atm}. {extra}",
            "Modern Korean in {boro} with {adj} fusion twists and rich kimchi-based sides. {extra}",
        ],
        "Mediterranean": [
            "{adj} Mediterranean mezze restaurant in {boro}. Hummus, grilled meats, {atm}. {extra}",
        ],
        "Steakhouse": [
            "Classic New York steakhouse in {boro}. Massive portions, {adj}, {atm}, legendary cuts. {extra}",
        ],
        "Pizza": [
            "New York pizza slice joint in {boro}. {adj}, {atm}, cash only, loyal regulars. {extra}",
            "Artisan wood-fired pizza in {boro}. Creative toppings, {adj}, great natural wine. {extra}",
        ],
        "Bakery": [
            "Neighborhood bakery in {boro}. {adj} pastries, fresh bread, excellent espresso. {atm}. {extra}",
        ],
        "Cafe": [
            "Specialty coffee shop in {boro}. {adj} atmosphere, single-origin pour-overs, {atm}. {extra}",
            "Cozy neighborhood cafe in {boro}, perfect for remote work, {adj} and quiet. {extra}",
        ],
        "Street Food": [
            "Street cart in {boro}. Cheap, quick, {adj}, and always busy. {atm}. {extra}",
        ],
    }

    adj_pool = ["casual", "upscale", "vibrant", "quiet", "intimate", "bustling", "elegant", "rustic", "trendy", "no-frills", "cozy", "minimalist", "loud", "bright", "dimly lit", "romantic", "family-friendly", "hipster", "old-school"]
    atm_pool = ["great for a date night", "perfect for groups", "ideal for solo dining", "wonderful for business lunches", "packed on weekends", "always a long wait", "reservations essential", "walk-ins welcome", "cash only", "BYOB friendly", "outdoor seating available"]
    extra_pool = ["Extensive wine list.", "Cash only.", "Open late.", "Dog friendly.", "Live music on weekends.", "Vegan options available.", "BYOB welcome.", "Michelin recommended.", "Zagat-rated.", "James Beard award-winning chef.", "Known for very long lines.", "Reservations required months ahead.", "No reservations taken.", "Best value in the neighborhood."]
    pasta_pool = ["pappardelle", "tagliatelle", "cacio e pepe", "carbonara", "orecchiette", "gnocchi", "risotto", "lasagna"]
    jp_pool = ["wagyu sashimi", "tonkotsu ramen", "omakase nigiri", "yakitori skewers", "tempura", "chirashi bowls"]
    cn_pool = ["Peking duck", "xiao long bao", "mapo tofu", "hand-pulled noodles", "pork belly buns", "crispy duck"]
    boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    boro_weights = [0.38, 0.28, 0.20, 0.10, 0.04]

    # Name fragments for generating plausible restaurant names
    prefixes = ["Casa", "Bistro", "Café", "Osteria", "Trattoria", "Kitchen", "Bar", "The", "Little", "Big", "Old", "New", "Corner", "Village", "Market", "Garden"]
    suffixes = ["& Co.", "NYC", "House", "Room", "Table", "Place", "Corner", "Spot", "Lane", "Street", "Ave", "Court"]
    middles = ["Bella", "Luna", "Verde", "Rosso", "Primo", "Oro", "Mare", "Terra", "Cielo", "Vino", "Fuego", "Azul", "Rouge", "Blanc", "Spice", "Smoke", "Salt", "Coal", "Oak", "Ember"]

    cuisines = list(templates.keys())
    cuisine_weights = [0.12, 0.10, 0.09, 0.10, 0.12, 0.08, 0.08, 0.07, 0.07, 0.05, 0.04, 0.04, 0.02, 0.01, 0.01]

    rows = []
    used_names = set()
    for i in range(n):
        cuisine = rng.choices(cuisines, weights=cuisine_weights, k=1)[0]
        boro = rng.choices(boros, weights=boro_weights, k=1)[0]

        # Generate unique name
        for _ in range(20):
            name = f"{rng.choice(prefixes)} {rng.choice(middles)}"
            if rng.random() > 0.5:
                name += f" {rng.choice(suffixes)}"
            if name not in used_names:
                used_names.add(name)
                break

        adj = rng.choice(adj_pool)
        atm = rng.choice(atm_pool)
        extra = rng.choice(extra_pool)

        tmpl = rng.choice(templates[cuisine])
        desc = tmpl.format(
            adj=adj, atm=atm, extra=extra, boro=boro,
            pasta=rng.choice(pasta_pool),
            jp_dish=rng.choice(jp_pool),
            cn_dish=rng.choice(cn_pool),
        )

        rows.append({"Restaurant": name, "Cuisine": cuisine, "Borough": boro, "Desc": desc})

    return pd.DataFrame(rows)


corpus_df = build_large_corpus(n=220)

# ─────────────────────────────────────────────
# 2. LOAD MODEL & PRE-EMBED CORPUS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NLP Transformer model…")
def load_nlp_model():
    from retrieval.vector_search import SemanticSearchModel
    return SemanticSearchModel(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Pre-computing 220 restaurant embeddings…")
def compute_embeddings(_corpus_df):
    """Embed all 220 restaurant descriptions once. _prefix prevents Streamlit trying to hash the df."""
    model = load_nlp_model()
    return model.embed_texts(_corpus_df["Desc"].tolist())

dataset_tensor = compute_embeddings(corpus_df)

# ─────────────────────────────────────────────
# 3. SEARCH UI
# ─────────────────────────────────────────────
st.caption(f"🗄️ Database: **{len(corpus_df)} synthetic NYC restaurants** indexed with 384-D Transformer embeddings")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "✨ Describe your ideal restaurant vibe:",
        placeholder="e.g., cozy romantic Italian with candles and jazz"
    )
with col2:
    top_k = st.slider("Results to show", 1, 15, 7)

if st.button("🔍 Search NLP Vector Space", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a vibe description!")
    else:
        with st.spinner("Computing Cosine Similarity across 220 restaurants…"):
            model = load_nlp_model()
            t0 = time.time()
            indices, scores = model.search(query.strip(), dataset_tensor, top_k=top_k)
            elapsed_ms = (time.time() - t0) * 1000

        st.success(f"Top **{len(indices)}** restaurants matched in **{elapsed_ms:.1f} ms** from a corpus of {len(corpus_df)}")

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            row = corpus_df.iloc[idx]
            results.append({
                "Rank":        rank,
                "Restaurant":  row["Restaurant"],
                "Cuisine":     row["Cuisine"],
                "Borough":     row["Borough"],
                "Similarity":  f"{score:.3f}",
                "NLP Match Evidence": row["Desc"],
            })

        result_df = pd.DataFrame(results).set_index("Rank")
        st.table(result_df)

        with st.expander("🔬 Architecture Deep-Dive"):
            st.markdown(f"""
**Live Pipeline:**
1. `retrieval/vector_search.py → SemanticSearchModel.embed_texts("{query[:40]}…")` tokenizes your query
2. HuggingFace `all-MiniLM-L6-v2` produces a `[1, 384]` float tensor
3. `torch.mm(query_vec, corpus_matrix.T)` computes **{len(corpus_df)} cosine scores** in one shot
4. `torch.topk(scores, k={top_k})` returns the top matches above

**Corpus size:** {len(corpus_df)} restaurants · **Embedding dim:** 384 · **Inference:** {elapsed_ms:.1f} ms
""")
