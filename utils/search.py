import numpy as np
import pandas as pd
import re
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
BOROUGH_KEYWORDS = ["manhattan", "brooklyn", "queens", "bronx", "staten island"]
ZIPCODE_TO_NEIGHBORHOOD = {
    "10001": "Chelsea", "10002": "Lower East Side", "10003": "East Village", "10004": "Financial District",
    "10005": "Financial District", "10006": "Financial District", "10007": "Tribeca", "10009": "East Village",
    "10010": "Flatiron", "10011": "West Village", "10012": "SoHo", "10013": "Tribeca", "10014": "West Village",
    "10016": "Murray Hill", "10017": "Midtown East", "10018": "Garment District", "10019": "Midtown West",
    "10021": "Upper East Side", "10022": "Midtown East", "10023": "Upper West Side", "10024": "Upper West Side",
    "10025": "Upper West Side", "10026": "Harlem", "10027": "Harlem", "10028": "Upper East Side",
    "10029": "East Harlem", "10030": "Harlem", "10031": "Hamilton Heights", "10032": "Washington Heights",
    "10033": "Washington Heights", "10034": "Inwood", "10035": "East Harlem", "10036": "Times Square",
    "10037": "Harlem", "10038": "Financial District", "10039": "Harlem", "10280": "Battery Park City",
    "10282": "Battery Park City",
    "11101": "Long Island City", "11102": "Astoria", "11103": "Astoria", "11104": "Sunnyside",
    "11105": "Astoria", "11106": "Astoria", "11109": "Long Island City", "11354": "Flushing",
    "11355": "Flushing", "11358": "Flushing", "11360": "Bayside", "11361": "Bayside", "11364": "Bayside",
    "11372": "Jackson Heights", "11373": "Elmhurst", "11374": "Rego Park", "11375": "Forest Hills",
    "11377": "Woodside", "11385": "Ridgewood", "11432": "Jamaica", "11433": "Jamaica",
    "11435": "Jamaica", "11436": "Jamaica",
    "11201": "Brooklyn Heights", "11205": "Clinton Hill", "11206": "Williamsburg", "11209": "Bay Ridge",
    "11211": "Williamsburg", "11213": "Crown Heights", "11215": "Park Slope", "11216": "Bedford-Stuyvesant",
    "11217": "Boerum Hill", "11220": "Sunset Park", "11221": "Bushwick", "11222": "Greenpoint",
    "11225": "Prospect Lefferts Gardens", "11226": "Flatbush", "11231": "Carroll Gardens", "11232": "Sunset Park",
    "11233": "Bedford-Stuyvesant", "11235": "Brighton Beach", "11237": "Bushwick", "11238": "Prospect Heights",
    "11249": "Williamsburg",
    "10451": "South Bronx", "10454": "Mott Haven", "10455": "Mott Haven", "10458": "Fordham",
    "10460": "West Farms", "10461": "Pelham Bay", "10463": "Riverdale", "10468": "Kingsbridge",
    "10471": "Riverdale", "10474": "Hunts Point",
    "10301": "St. George", "10304": "Tompkinsville", "10305": "South Beach", "10306": "New Dorp",
    "10312": "Eltingville", "10314": "New Springville",
}
NEIGHBORHOOD_TO_ZIPCODES = {}
for _zipcode, _neighborhood in ZIPCODE_TO_NEIGHBORHOOD.items():
    NEIGHBORHOOD_TO_ZIPCODES.setdefault(_neighborhood, set()).add(_zipcode)
MANUAL_NEIGHBORHOOD_ZIPCODES = {
    "Hell's Kitchen": {"10019", "10036"},
    "Hells Kitchen": {"10019", "10036"},
    "Midtown West": {"10019", "10036"},
    "Times Square": {"10036", "10019"},
    "Theater District": {"10019", "10036"},
    "Hudson Yards": {"10001", "10018", "10019"},
    "Koreatown": {"10001", "10010"},
    "NoMad": {"10001", "10010", "10016"},
    "Nomad": {"10001", "10010", "10016"},
    "Greenwich Village": {"10011", "10012", "10014"},
    "Alphabet City": {"10009"},
    "Loisaida": {"10009"},
    "NoHo": {"10003", "10012"},
    "Hudson Square": {"10013", "10014"},
    "Chinatown": {"10002", "10013"},
    "Little Italy": {"10012", "10013"},
    "Kips Bay": {"10016"},
    "Curry Hill": {"10016"},
    "Tenderloin": {"10001", "10018", "10019"},
    "Hudson Heights": {"10033"},
    "West Harlem": {"10027", "10031"},
    "Spanish Harlem": {"10029"},
    "El Barrio": {"10029"},
    "Lower Manhattan": {"10004", "10005", "10006", "10007", "10038"},
    "Downtown Brooklyn": {"11201", "11217"},
    "DUMBO": {"11201"},
    "Dumbo": {"11201"},
    "Fort Greene": {"11201", "11205", "11217"},
    "Prospect Heights": {"11217", "11238"},
    "Cobble Hill": {"11201", "11231"},
    "Red Hook": {"11231"},
    "Columbia Street Waterfront District": {"11231"},
    "Gowanus": {"11215", "11217", "11231"},
    "South Slope": {"11215"},
    "South Park Slope": {"11215"},
    "Ditmas Park": {"11226"},
    "Windsor Terrace": {"11215"},
    "Ocean Hill": {"11233"},
    "Stuyvesant Heights": {"11216", "11233"},
    "Little Odessa": {"11235"},
    "Downtown Flushing": {"11354", "11355"},
    "Koreatown Flushing": {"11354", "11355"},
    "Hunters Point": {"11101", "11109"},
    "Ditmars": {"11105"},
    "South Jamaica": {"11432", "11433", "11435", "11436"},
    "St George": {"10301"},
}
for _neighborhood, _zipcodes in MANUAL_NEIGHBORHOOD_ZIPCODES.items():
    NEIGHBORHOOD_TO_ZIPCODES.setdefault(_neighborhood, set()).update(_zipcodes)
QUERY_CUISINE_HINTS = {
    "american": {"American"},
    "burger": {"Hamburgers", "American"},
    "burgers": {"Hamburgers", "American"},
    "bbq": {"Korean", "American"},
    "barbecue": {"American"},
    "italian": {"Italian"},
    "ramen": {"Japanese"},
    "sushi": {"Japanese"},
    "omakase": {"Japanese"},
    "japanese": {"Japanese"},
    "chinese": {"Chinese"},
    "dim sum": {"Chinese"},
    "dumpling": {"Chinese"},
    "dumplings": {"Chinese"},
    "korean bbq": {"Korean"},
    "korean": {"Korean"},
    "pho": {"Vietnamese"},
    "vietnamese": {"Vietnamese"},
    "pad thai": {"Thai"},
    "thai": {"Thai"},
    "curry": {"Indian", "Thai"},
    "indian": {"Indian"},
    "taco": {"Mexican"},
    "tacos": {"Mexican"},
    "mexican": {"Mexican"},
    "pasta": {"Italian"},
    "pizza": {"Pizza"},
    "french": {"French"},
    "greek": {"Greek"},
    "mediterranean": {"Mediterranean"},
    "middle eastern": {"Middle Eastern"},
    "shawarma": {"Middle Eastern"},
    "falafel": {"Middle Eastern"},
    "ethiopian": {"Ethiopian"},
    "caribbean": {"Caribbean"},
    "seafood": {"Seafood"},
    "steak": {"Steak"},
    "bakery": {"Bakery"},
    "dessert": {"Bakery", "Ice Cream"},
    "ice cream": {"Ice Cream"},
    "gelato": {"Ice Cream"},
    "cafe": {"Cafe"},
    "coffee": {"Cafe"},
    "juice": {"Juice Bar"},
    "juice bar": {"Juice Bar"},
    "vegetarian": {"Vegetarian"},
    "vegan": {"Vegetarian"},
    "sandwich": {"Sandwiches"},
    "sandwiches": {"Sandwiches"},
    "chicken": {"Chicken"},
    "spanish": {"Spanish"},
    "bagel": {"Sandwiches", "Bakery", "Coffee/Tea"},
}
PRICE_HINTS = {
    "$": 1,
    "$$": 2,
    "$$$": 3,
    "$$$$": 4,
    "cheap": 1,
    "affordable": 1,
    "budget": 1,
    "inexpensive": 1,
    "upscale": 3,
    "fancy": 3,
    "luxury": 4,
}


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
    neighborhood = row.get("neighborhood", "")
    location_text = f"{neighborhood}, {boro}" if neighborhood and neighborhood != boro else boro
    grade = row.get("grade", "")
    score = row.get("score", 0)
    address = row.get("address", "")
    extras = CUISINE_KEYWORDS.get(cuisine, cuisine)
    summary = row.get("g_summary", "")
    rating = pd.to_numeric(row.get("g_rating"), errors="coerce")
    price = pd.to_numeric(row.get("g_price"), errors="coerce")
    reviews = pd.to_numeric(row.get("g_reviews", 0), errors="coerce")
    hygiene = {"A": "excellent hygiene", "B": "good hygiene", "C": "acceptable hygiene"}.get(grade, "")
    if pd.notna(rating) and pd.notna(reviews) and reviews > 0:
        rating_str = f"Google rating {float(rating):.1f}/5 based on {int(reviews):,} reviews."
    elif pd.notna(rating):
        rating_str = f"Google rating {float(rating):.1f}/5."
    else:
        rating_str = ""
    price_str = f"Price level: {price_label(int(price))}." if pd.notna(price) and price > 0 else ""

    return (
        f"{name} is a {cuisine} restaurant in {location_text}, New York City. "
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


def _normalize_location_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _register_neighborhood_aliases():
    alias_lookup = {}

    def add_alias(alias, canonical_name):
        normalized_alias = _normalize_location_text(alias)
        if normalized_alias:
            alias_lookup[normalized_alias] = canonical_name

    for canonical_name in NEIGHBORHOOD_TO_ZIPCODES:
        add_alias(canonical_name, canonical_name)

    manual_aliases = {
        "Hell's Kitchen": ["hells kitchen", "hell's kitchen", "clinton"],
        "Times Square": ["times square"],
        "Theater District": ["theater district", "theatre district"],
        "Hudson Yards": ["hudson yards"],
        "Koreatown": ["koreatown", "k town", "k-town"],
        "NoMad": ["nomad", "no mad"],
        "Greenwich Village": ["greenwich village", "the village"],
        "Alphabet City": ["alphabet city"],
        "Loisaida": ["loisaida"],
        "NoHo": ["noho", "no ho"],
        "Hudson Square": ["hudson square"],
        "Chinatown": ["chinatown"],
        "Little Italy": ["little italy"],
        "Kips Bay": ["kips bay"],
        "Curry Hill": ["curry hill", "little india"],
        "Tenderloin": ["tenderloin"],
        "Hudson Heights": ["hudson heights"],
        "West Harlem": ["west harlem"],
        "Lower East Side": ["les", "lower east side"],
        "Upper East Side": ["ues", "upper east side"],
        "Upper West Side": ["uws", "upper west side"],
        "Financial District": ["fidi", "financial district", "financial district manhattan"],
        "Long Island City": ["lic", "long island city"],
        "Prospect Lefferts Gardens": ["plg", "prospect lefferts gardens"],
        "Bedford-Stuyvesant": ["bed stuy", "bed-stuy", "bedford stuyvesant", "bedford-stuyvesant"],
        "South Bronx": ["south bronx", "the south bronx"],
        "Mott Haven": ["mott haven"],
        "Washington Heights": ["washington heights", "wash heights"],
        "Midtown West": ["midtown west"],
        "Midtown East": ["midtown east"],
        "West Village": ["west village"],
        "East Village": ["east village"],
        "Battery Park City": ["battery park city", "bpc"],
        "Brooklyn Heights": ["brooklyn heights"],
        "Clinton Hill": ["clinton hill"],
        "Park Slope": ["park slope"],
        "Boerum Hill": ["boerum hill"],
        "Cobble Hill": ["cobble hill"],
        "Fort Greene": ["fort greene"],
        "Crown Heights": ["crown heights"],
        "Prospect Heights": ["prospect heights"],
        "Red Hook": ["red hook"],
        "Gowanus": ["gowanus"],
        "DUMBO": ["dumbo", "down under the manhattan bridge overpass"],
        "Bushwick": ["bushwick"],
        "Greenpoint": ["greenpoint"],
        "Williamsburg": ["williamsburg", "east williamsburg", "west williamsburg"],
        "Ditmas Park": ["ditmas park"],
        "Windsor Terrace": ["windsor terrace"],
        "Ocean Hill": ["ocean hill"],
        "Stuyvesant Heights": ["stuyvesant heights", "stuy heights"],
        "Little Odessa": ["little odessa"],
        "Jackson Heights": ["jackson heights"],
        "Forest Hills": ["forest hills"],
        "Rego Park": ["rego park"],
        "Flatbush": ["flatbush"],
        "Astoria": ["astoria"],
        "Hunters Point": ["hunters point"],
        "Ditmars": ["ditmars", "ditmars astoria"],
        "Sunnyside": ["sunnyside"],
        "Woodside": ["woodside"],
        "Flushing": ["flushing"],
        "Downtown Flushing": ["downtown flushing"],
        "Bayside": ["bayside"],
        "Jamaica": ["jamaica"],
        "South Jamaica": ["south jamaica"],
        "Ridgewood": ["ridgewood"],
        "Chelsea": ["chelsea"],
        "SoHo": ["soho"],
        "Tribeca": ["tribeca", "triBeCa"],
        "Flatiron": ["flatiron", "flatiron district"],
        "Murray Hill": ["murray hill"],
        "Garment District": ["garment district"],
        "Harlem": ["harlem"],
        "East Harlem": ["east harlem", "el barrio", "spanish harlem"],
        "Hamilton Heights": ["hamilton heights"],
        "Inwood": ["inwood"],
        "Riverdale": ["riverdale"],
        "Fordham": ["fordham"],
        "Pelham Bay": ["pelham bay"],
        "St. George": ["st george", "st. george"],
        "Tompkinsville": ["tompkinsville"],
        "New Dorp": ["new dorp"],
        "New Springville": ["new springville"],
    }
    for canonical_name, aliases in manual_aliases.items():
        if canonical_name not in NEIGHBORHOOD_TO_ZIPCODES:
            continue
        for alias in aliases:
            add_alias(alias, canonical_name)

    return alias_lookup


NORMALIZED_NEIGHBORHOOD_ALIASES = _register_neighborhood_aliases()


def neighborhood_from_zipcode(zipcode):
    zipcode_text = str(zipcode or "").strip()
    if len(zipcode_text) >= 5:
        zipcode_text = zipcode_text[:5]
    return ZIPCODE_TO_NEIGHBORHOOD.get(zipcode_text, "")


def _extract_query_intent(query):
    lowered = str(query or "").lower()
    tokens = {token.strip(" ,.!?") for token in lowered.split() if token.strip(" ,.!?")}
    zipcodes_in_query = re.findall(r"\b\d{5}\b", lowered)
    matched_zipcode = next((zipcode for zipcode in zipcodes_in_query if zipcode in ZIPCODE_TO_NEIGHBORHOOD), None)
    borough = next((name.title() for name in BOROUGH_KEYWORDS if name in lowered), None)
    normalized_query = _normalize_location_text(query)
    neighborhood = next(
        (
            NORMALIZED_NEIGHBORHOOD_ALIASES[key]
            for key in sorted(NORMALIZED_NEIGHBORHOOD_ALIASES.keys(), key=len, reverse=True)
            if key and key in normalized_query
        ),
        None,
    )
    neighborhood_zipcodes = sorted(NEIGHBORHOOD_TO_ZIPCODES.get(neighborhood, set())) if neighborhood else []
    if not neighborhood and matched_zipcode:
        neighborhood = ZIPCODE_TO_NEIGHBORHOOD[matched_zipcode]
        neighborhood_zipcodes = [matched_zipcode]

    desired_price = None
    for hint, value in PRICE_HINTS.items():
        if hint in lowered:
            desired_price = value
            break

    desired_cuisines = set()
    for hint, cuisines in QUERY_CUISINE_HINTS.items():
        if hint in lowered:
            desired_cuisines.update(cuisines)

    cuisine_vocab = {
        "american": "American",
        "bakery": "Bakery",
        "cafe": "Cafe",
        "caribbean": "Caribbean",
        "chicken": "Chicken",
        "chinese": "Chinese",
        "ethiopian": "Ethiopian",
        "french": "French",
        "greek": "Greek",
        "hamburger": "Hamburgers",
        "hamburgers": "Hamburgers",
        "indian": "Indian",
        "italian": "Italian",
        "japanese": "Japanese",
        "juice": "Juice Bar",
        "korean": "Korean",
        "mediterranean": "Mediterranean",
        "mexican": "Mexican",
        "middle eastern": "Middle Eastern",
        "pizza": "Pizza",
        "sandwich": "Sandwiches",
        "sandwiches": "Sandwiches",
        "seafood": "Seafood",
        "spanish": "Spanish",
        "steak": "Steak",
        "thai": "Thai",
        "vegetarian": "Vegetarian",
        "vegan": "Vegetarian",
        "vietnamese": "Vietnamese",
    }
    for token, cuisine in cuisine_vocab.items():
        if token in lowered:
            desired_cuisines.add(cuisine)

    return {
        "zipcode": matched_zipcode,
        "borough": borough,
        "neighborhood": neighborhood,
        "neighborhood_zipcodes": neighborhood_zipcodes,
        "has_location": bool(borough or neighborhood or matched_zipcode),
        "desired_price": desired_price,
        "desired_cuisines": desired_cuisines,
        "tokens": tokens,
        "lowered": lowered,
    }


def semantic_search(query, df, embeddings, top_k, boro_filter, grade_filter, min_rating, profile=None, min_match=0.45):
    profile = profile or {}
    profile_text = build_profile_prompt(profile)
    expanded_query = f"{query}. {profile_text}".strip()
    intent = _extract_query_intent(query)

    semantic_scores = np.zeros(len(df))
    model = load_model()
    if embeddings is not None and model is not None:
        try:
            query_embedding = model.encode([expanded_query], normalize_embeddings=True)
            semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
        except Exception:
            semantic_scores = np.zeros(len(df))

    semantic_norm = np.clip(semantic_scores, 0, 1)
    lexical_scores = df["description"].fillna("").apply(lambda text: lexical_score(query, text)).to_numpy()
    name_scores = df.get("dba", pd.Series([""] * len(df), index=df.index)).fillna("").apply(lambda text: lexical_score(query, text)).to_numpy()
    cuisine_series = df.get("cuisine", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    borough_series = df.get("boro", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    neighborhood_series = df.get("neighborhood", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    zipcode_series = df.get("zipcode", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str[:5]
    summary_series = df.get("g_summary", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    quality_scores = (
        0.5 * pd.to_numeric(df.get("g_rating", 3.4), errors="coerce").fillna(3.4).to_numpy() / 5
        + 0.3 * (1 - pd.to_numeric(df.get("score", 21), errors="coerce").fillna(21).clip(0, 42).to_numpy() / 42)
        + 0.2 * df["grade"].map({"A": 1.0, "B": 0.78, "C": 0.58}).fillna(0.6).to_numpy()
    )
    profile_scores = score_restaurants_for_user(df, profile)["preference_score"].to_numpy() / 10
    price_series = pd.to_numeric(df.get("g_price", 2), errors="coerce").fillna(2).clip(1, 4)

    cuisine_boost = np.zeros(len(df))
    if intent["desired_cuisines"]:
        cuisine_boost = cuisine_series.apply(
            lambda value: 1.0 if any(target.lower() in str(value).lower() for target in intent["desired_cuisines"]) else 0.0
        ).to_numpy()
    cuisine_query_present = len(intent["desired_cuisines"]) > 0
    structured_intent_present = bool(
        intent["has_location"] or cuisine_query_present or intent["desired_price"] is not None
    )

    borough_boost = np.zeros(len(df))
    if intent["borough"]:
        borough_boost = borough_series.str.lower().eq(intent["borough"].lower()).astype(float).to_numpy()

    neighborhood_boost = np.zeros(len(df))
    if intent["neighborhood"]:
        normalized_neighborhoods = neighborhood_series.map(_normalize_location_text)
        neighborhood_boost = normalized_neighborhoods.eq(_normalize_location_text(intent["neighborhood"])).astype(float).to_numpy()

    zipcode_boost = np.zeros(len(df))
    if intent["neighborhood_zipcodes"]:
        zipcode_boost = zipcode_series.isin(intent["neighborhood_zipcodes"]).astype(float).to_numpy()
    elif intent["zipcode"]:
        zipcode_boost = zipcode_series.eq(intent["zipcode"]).astype(float).to_numpy()

    location_match = np.maximum.reduce([neighborhood_boost, borough_boost, zipcode_boost])

    price_boost = np.zeros(len(df))
    if intent["desired_price"] is not None:
        price_boost = (1 - (price_series - intent["desired_price"]).abs() / 3).clip(0, 1).to_numpy()

    keyword_boost = np.zeros(len(df))
    if intent["tokens"]:
        searchable_text = (
            df.get("dba", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower()
            + " "
            + cuisine_series.str.lower()
            + " "
            + borough_series.str.lower()
            + " "
            + neighborhood_series.str.lower()
            + " "
            + summary_series.str.lower()
        )
        keyword_boost = searchable_text.apply(
            lambda text: sum(token in text for token in intent["tokens"]) / max(len(intent["tokens"]), 1)
        ).to_numpy()

    mask = np.ones(len(df), dtype=bool)
    if boro_filter != "All":
        mask &= df["boro"].str.lower() == boro_filter.lower()
    if grade_filter != "All":
        mask &= df["grade"] == grade_filter
    if min_rating > 0:
        mask &= pd.to_numeric(df.get("g_rating", 0), errors="coerce").fillna(0) >= min_rating

    filtered_scores = np.where(
        mask,
        0.28 * semantic_norm
        + 0.16 * lexical_scores
        + 0.1 * name_scores
        + 0.08 * keyword_boost
        + 0.16 * cuisine_boost
        + 0.18 * zipcode_boost
        + 0.16 * neighborhood_boost
        + 0.12 * borough_boost
        + 0.05 * price_boost
        + 0.03 * quality_scores
        + 0.02 * profile_scores,
        -1.0,
    )
    strong_text_signal = np.maximum.reduce([lexical_scores, name_scores, keyword_boost])
    query_signal = np.maximum.reduce([
        semantic_norm,
        lexical_scores,
        name_scores,
        keyword_boost,
        cuisine_boost,
        neighborhood_boost,
        borough_boost,
        zipcode_boost,
    ])
    semantic_floor = max(min_match - 0.1, 0.4) if embeddings is not None and model is not None else 0.0
    relevance_gate = (
        (semantic_norm >= semantic_floor)
        | (strong_text_signal >= 0.34)
        | (cuisine_boost >= 1.0)
    )
    relevance_gate = relevance_gate & (query_signal >= 0.12)
    if not structured_intent_present:
        relevance_gate = relevance_gate & (strong_text_signal >= 0.12)

    if intent["has_location"]:
        relevance_gate = relevance_gate & (location_match >= 1.0)
    if cuisine_query_present and not intent["has_location"]:
        relevance_gate = relevance_gate & (cuisine_boost >= 1.0)

    if intent["has_location"] and cuisine_query_present:
        filtered_scores = np.where(
            mask,
            filtered_scores + 0.12 * np.minimum(location_match, cuisine_boost),
            filtered_scores,
        )
        cuisine_location_gate = (
            (location_match >= 1.0)
            & (
                (cuisine_boost >= 1.0)
                | ((semantic_norm >= 0.62) & (strong_text_signal >= 0.45))
            )
        )
        relevance_gate = relevance_gate & cuisine_location_gate

    valid_mask = mask & relevance_gate & (filtered_scores >= min_match)
    if not valid_mask.any():
        return df.iloc[0:0].copy()

    valid_indices = np.where(valid_mask)[0]
    ranked_indices = valid_indices[np.argsort(filtered_scores[valid_indices])[::-1][:top_k]]
    results = df.iloc[ranked_indices].copy()
    results["similarity"] = filtered_scores[ranked_indices]
    results["match_percent"] = np.round(np.clip(results["similarity"], 0, 1) * 100).astype(int)
    return results.reset_index(drop=True)
