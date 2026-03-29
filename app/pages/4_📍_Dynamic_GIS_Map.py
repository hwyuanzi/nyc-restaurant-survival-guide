import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px

st.set_page_config(page_title="Geospatial Intelligence", page_icon="🗺️", layout="wide")

from app.ui_utils import apply_apple_theme
apply_apple_theme()

st.title("📍 Dynamic Multi-Feature Geospatial GIS")

st.markdown("""
### 🎯 What is this?
Advanced Geographical Information Systems (GIS) mapping overlaid with machine learning multi-dimensional attributes. It maps individual NYC restaurants directly onto a 3D Cartographic space.

### 📊 Data Source & Features
**Synthetic data** modeled on the distribution patterns of the real **NYC DOHMH Restaurant Inspections** dataset. Geolocation coordinates (Latitude & Longitude) are generated within the 5 Boroughs, with correlated operational features (Violations, Pest Control, Training Hours) that mirror actual inspection trends.

### 🧠 Why is it useful?
Physical location strongly correlates with operational standards due to rent, enforcement cycles, and localized supply chains. Visualizing multiple features allows you to discover spatial correlations—*do restaurants with poor pest control cluster in specific neighborhoods?*

### 💡 How to use it?
*   **3D Hexbin Map:** Rotate the map (`Shift` + `Drag Mouse`). Tall, brightly-colored columns indicate dense clusters of restaurants.
*   **Multi-Feature Mapbox:** Use the **Color Mapping & Analytics Metric** dropdown to map Grades (categorical) or continuous values like *Past Critical Violations*!
---
""", unsafe_allow_html=True)

np.random.seed(42)
n_restaurants = 2500

# Strict 5 Borough true central lat/lons
boro_data = {
    "Manhattan": (40.7580, -73.9855, 0.03), 
    "Brooklyn": (40.6500, -73.9499, 0.04),
    "Queens": (40.7282, -73.7948, 0.05),
    "Bronx": (40.8448, -73.8648, 0.03),
    "Staten Island": (40.5795, -74.1502, 0.04)
}

mock_data = []
for _ in range(n_restaurants):
    boro = np.random.choice(list(boro_data.keys()), p=[0.4, 0.25, 0.2, 0.1, 0.05])
    lat_center, lon_center, spread = boro_data[boro]
    
    lat = np.random.normal(lat_center, spread)
    lon = np.random.normal(lon_center, spread)
    grade = np.random.choice(["A", "B", "C"], p=[0.75, 0.15, 0.10])
    
    # Generate correlated continuous features realistically
    if grade == "A":
        viol = np.random.uniform(0, 3)
        pest = np.random.uniform(7, 30)
        train = np.random.uniform(40, 100)
    elif grade == "B":
        viol = np.random.uniform(3, 7)
        pest = np.random.uniform(30, 60)
        train = np.random.uniform(10, 40)
    else:
        viol = np.random.uniform(7, 15)
        pest = np.random.uniform(60, 120)
        train = np.random.uniform(0, 10)
        
    cuisine = np.random.choice(["Italian", "Chinese", "American", "Mexican", "Japanese", "Cafe/Coffee"])
    risk_score = {"A": 0, "B": 50, "C": 100}[grade]
    mock_data.append((lat, lon, grade, cuisine, boro, risk_score, viol, pest, train))

df = pd.DataFrame(mock_data, columns=['lat', 'lon', 'ML_Grade', 'Cuisine', 'Borough', 'Risk_Score', 'Violations', 'Pest_Control_Days', 'Training_Hours'])

# Hard clip out-of-bounds anomalies (NJ / water)
df = df[
    (df['lat'] >= 40.50) & (df['lat'] <= 40.91) &
    (df['lon'] >= -74.25) & (df['lon'] <= -73.70)
]

tab1, tab2 = st.tabs(["🔥 3D Density Hexbin Map", "🪧 Multi-Dimensional Feature Explorer"])

with tab1:
    st.subheader("3D Hexagon Density Map (PyDeck)")
    
    layer = pdk.Layer(
        'HexagonLayer',
        data=df,
        get_position='[lon, lat]',
        radius=200, 
        elevation_scale=15,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
        # Warm Visual Hierarchy Colors
        color_range=[
            [255, 236, 179, 150], # More transparent so we can see streets underneath
            [255, 204, 128, 150],
            [255, 171, 64, 150],
            [255, 138, 101, 150],
            [244, 81, 30, 200],
            [216, 67, 21, 230]    # Solid core
        ],
    )

    # NEW: Add a Scatterplot layer so users can zoom into the street level and click actual restaurants!
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_color="[0, 122, 255, 200]", # Apple Blue
        get_radius=40,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=11.0, # Zoom in slightly closer to see the streets
        pitch=45,
        bearing=0
    )

    r = pdk.Deck(
        layers=[scatter_layer, layer], # Draw individual dots FIRST, then 3D Hexagons on top
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Exploration Data:</b><br />Density Cluster Count (if hexagon): {elevationValue}<br />Cuisine Type (if dot): {Cuisine}<br />Borough: {Borough}<br />Grade: <b>{ML_Grade}</b>",
            "style": {"color": "white", "backgroundColor": "#1D1D1FB3", "borderRadius": "8px", "padding": "10px"}
        },
        # Fix: Use 'road' for high-detail streets, parks, and text labels globally without Mapbox token
        map_style="road" 
    )
    st.pydeck_chart(r)

with tab2:
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader("GIS Database Filters")
        selected_metric = st.selectbox("Color Mapping & Analytics Metric:", ["ML_Grade", "Violations", "Pest_Control_Days", "Training_Hours"])
        selected_boro = st.selectbox("Borough:", ["All NYC"] + list(df['Borough'].unique()))
        selected_cuisine = st.selectbox("Cuisine:", ["All"] + list(df['Cuisine'].unique()))
        
    with col2:
        st.subheader(f"Interactive Scatter View: Mapping {selected_metric}")
        
        filtered_df = df
        if selected_boro != "All NYC":
            filtered_df = filtered_df[filtered_df['Borough'] == selected_boro]
        if selected_cuisine != "All":
            filtered_df = filtered_df[filtered_df['Cuisine'] == selected_cuisine]
            
        if len(filtered_df) == 0:
            st.warning("No restaurants match this constraint in NYC!")
        else:
            if selected_metric == "ML_Grade":
                # Categorical Mapping
                fig = px.scatter_mapbox(
                    filtered_df, 
                    lat="lat", lon="lon", color="ML_Grade",
                    color_discrete_map={"A": "#34C759", "B": "#FFCC00", "C": "#FF3B30"},
                    hover_name="Borough",
                    hover_data={"lat": False, "lon": False, "ML_Grade": True, "Cuisine": True},
                    zoom=10, center={"lat": 40.7128, "lon": -74.0060}, height=650, opacity=0.85,
                )
            else:
                # Continuous Mapping using Advanced Color Gradients
                if selected_metric == "Training_Hours":
                    colorscale = "Aggrnyl" # High Training = Green/Yellow
                else:
                    colorscale = "OrRd"    # High Violations/Pest Days = Red Danger
                    
                fig = px.scatter_mapbox(
                    filtered_df, 
                    lat="lat", lon="lon", color=selected_metric,
                    color_continuous_scale=colorscale,
                    hover_name="Borough",
                    hover_data={"lat": False, "lon": False, "ML_Grade": True, selected_metric: True},
                    zoom=10, center={"lat": 40.7128, "lon": -74.0060}, height=650, opacity=0.85,
                    size_max=12
                )
            
            # Use light clean tiles to fit the Apple UI theme
            fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)
