import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import random

from models.custom_mlp import CustomMLP, train_mlp
from utils.user_profile import init_session_state, render_profile_sidebar

st.set_page_config(page_title="Health Analyzer Sandbox", page_icon="🧪", layout="wide")
from app.ui_utils import apply_apple_theme
apply_apple_theme()

init_session_state()

with st.sidebar:
    profile = render_profile_sidebar()
    st.markdown("---")

st.title("🧪 Full-Stack Entity ML Sandbox")

st.markdown("""
### 🎯 What is this?
An end-to-end Machine Learning Dashboard. The PyTorch MLP you see here was **actually trained** via gradient descent on a synthetic dataset modeled on DOHMH inspection patterns.

### 💡 How to use it?
1. **Search & Select:** Find a restaurant entity in the database.
2. **Override Features:** Drag sliders to modify operational parameters.
3. **Live Prediction:** The trained neural network re-evaluates the Health Grade in real-time.
4. **Visualize:** The Radar Chart shows the restaurant's risk footprint vs. NYC benchmarks, and the Probability Bar Chart shows the model's confidence across all 3 grades.
---
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# 1. GENERATE A LARGE SYNTHETIC DATABASE
# ──────────────────────────────────────────
@st.cache_data
def load_restaurant_db(n=180, seed=42):
    """Generates a diverse, realistic NYC restaurant database."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    prefixes = ["Casa", "Bistro", "Café", "Osteria", "Trattoria", "Kitchen", "Bar", "The",
                "Little", "Big", "Old", "New", "Corner", "Village", "Market", "Garden",
                "Uncle", "Mama", "Golden", "Silver", "Royal", "Lucky", "Happy", "Fresh"]
    middles = ["Bella", "Luna", "Verde", "Rosso", "Primo", "Oro", "Mare", "Terra", "Cielo",
               "Vino", "Fuego", "Azul", "Rouge", "Blanc", "Spice", "Smoke", "Salt", "Coal",
               "Oak", "Ember", "Jade", "Pearl", "Dragon", "Phoenix", "Star", "Sun", "Moon"]
    suffixes = ["& Co.", "NYC", "House", "Room", "Table", "Place", "Corner", "Spot", "Grill"]

    cuisines = ["Italian", "Chinese", "American", "Mexican", "Japanese", "French", "Indian",
                "Thai", "Korean", "Mediterranean", "Steakhouse", "Pizza", "Bakery", "Cafe",
                "Street Food", "Vietnamese", "Caribbean", "Ethiopian"]
    boros = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

    rows = []
    used_names = set()
    for _ in range(n):
        # Generate unique name
        for _try in range(30):
            name = f"{rng.choice(prefixes)} {rng.choice(middles)}"
            if rng.random() > 0.5:
                name += f" {rng.choice(suffixes)}"
            if name not in used_names:
                used_names.add(name)
                break

        cuisine = rng.choice(cuisines)
        boro = rng.choices(boros, weights=[0.38, 0.28, 0.20, 0.10, 0.04])[0]

        # Generate correlated features — cleaner restaurants get better stats
        profile = rng.random()  # 0 = terrible, 1 = excellent
        viol = max(0, round(np_rng.normal(12 - 10 * profile, 2), 1))
        pest = max(1, round(np_rng.normal(140 - 120 * profile, 15), 1))
        train = max(0, min(100, round(np_rng.normal(10 + 70 * profile, 10), 1)))
        hours = max(20, min(168, round(np_rng.normal(100 - 30 * profile, 15), 1)))

        rows.append({
            "Restaurant": name, "Borough": boro, "Cuisine": cuisine,
            "Violations": viol, "Pest_Control_Days": pest,
            "Training_Hours": train, "Operating_Hours_Wk": hours,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# 2. ACTUALLY TRAIN THE MLP WITH GRADIENT DESCENT
# ──────────────────────────────────────────
@st.cache_resource
def get_trained_model():
    """
    Trains the CustomMLP on a synthetic labeled dataset using real gradient descent.
    Labels are derived from a ground-truth function: restaurants with high violations,
    long pest-control gaps, low training, and high hours are graded C; the opposite get A.
    """
    np_rng = np.random.default_rng(123)
    n_train = 2000

    # Generate synthetic training data
    violations = np_rng.uniform(0, 15, n_train)
    pest_days = np_rng.uniform(1, 180, n_train)
    training_hrs = np_rng.uniform(0, 100, n_train)
    op_hours = np_rng.uniform(20, 168, n_train)

    X = np.column_stack([violations, pest_days, training_hrs, op_hours]).astype(np.float32)

    # Ground-truth labeling function (the "oracle" the model must learn)
    risk_score = (violations / 15) * 0.4 + (pest_days / 180) * 0.25 + ((100 - training_hrs) / 100) * 0.25 + (op_hours / 168) * 0.1
    labels = np.where(risk_score < 0.35, 0, np.where(risk_score < 0.55, 1, 2))  # A=0, B=1, C=2

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Build and train
    model = CustomMLP(input_dim=4, hidden_dim=64, output_dim=3, dropout=0.1)
    model, history = train_mlp(model, X_tensor, y_tensor, epochs=80, lr=0.005)

    return model


df = load_restaurant_db()
model = get_trained_model()

# ──────────────────────────────────────────
# 3. SEARCH & SELECT UI
# ──────────────────────────────────────────
st.caption(f"🗄️ Database: **{len(df)} synthetic NYC restaurants**")
search_query = st.text_input("🔍 Search by Name, Borough, or Cuisine:", "")

if search_query:
    filtered_df = df[df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
else:
    filtered_df = df

st.markdown("##### Select a restaurant from the database:")
selected_row = st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    height=250,
)

if len(selected_row.selection.rows) > 0:
    row_idx = selected_row.selection.rows[0]
    restaurant_data = filtered_df.iloc[row_idx].to_dict()
else:
    restaurant_data = df.iloc[0].to_dict()

st.divider()

# ──────────────────────────────────────────
# 4. SANDBOX + LIVE PREDICTION + VIZ
# ──────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader(f"🛠️ Sandbox: {restaurant_data['Restaurant']}")
    st.caption(f"{restaurant_data['Borough']} · {restaurant_data['Cuisine']}")

    sb_viol = st.slider("Past Critical Violations (Count)", 0.0, 15.0, float(restaurant_data["Violations"]))
    sb_pest = st.slider("Days Since Pest Control", 0.0, 180.0, float(restaurant_data["Pest_Control_Days"]))
    sb_train = st.slider("Employee Training (Hours/Mo)", 0.0, 100.0, float(restaurant_data["Training_Hours"]))
    sb_hours = st.slider("Operating (Hours/Wk)", 20.0, 168.0, float(restaurant_data["Operating_Hours_Wk"]))

with col_right:
    # Live inference
    input_tensor = torch.tensor([[sb_viol, sb_pest, sb_train, sb_hours]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    grade_map = {0: "A", 1: "B", 2: "C"}
    predicted_grade = grade_map[pred_idx]

    st.subheader(f"🧠 Neural Prediction: Grade {predicted_grade}")

    if predicted_grade == "A":
        st.success(f"**Safe** · Confidence: {probs[0]*100:.1f}%")
    elif predicted_grade == "B":
        st.warning(f"**At Risk** · Confidence: {probs[1]*100:.1f}%")
    else:
        st.error(f"**Critical** · Confidence: {probs[2]*100:.1f}%")

    # ── Probability Bar Chart ──
    st.markdown("##### 📊 Class Probability Breakdown")
    prob_fig = go.Figure(go.Bar(
        x=["Grade A", "Grade B", "Grade C"],
        y=[probs[0], probs[1], probs[2]],
        marker_color=["#34C759", "#FFCC00", "#FF3B30"],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    prob_fig.update_layout(
        yaxis=dict(range=[0, 1], title="P(class)"),
        height=220, margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # ── Radar / Spider Chart ──
    st.markdown("##### 🕸️ Multi-Dimensional Risk Radar")
    metrics = ['Violation<br>Density', 'Infestation<br>Threat', 'Training<br>Deficit', 'Employee<br>Fatigue']
    actual_risk = [
        min((sb_viol / 15.0) * 100, 100),
        min((sb_pest / 180.0) * 100, 100),
        min(((100 - sb_train) / 100.0) * 100, 100),
        min((sb_hours / 168.0) * 100, 100),
    ]
    safe_baseline = [30.0, 20.0, 40.0, 60.0]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=actual_risk + [actual_risk[0]],
        theta=metrics + [metrics[0]],
        fill='toself', name='Live Sandbox',
        line_color='#007AFF', fillcolor='rgba(0,122,255,0.35)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=safe_baseline + [safe_baseline[0]],
        theta=metrics + [metrics[0]],
        fill='none', name='NYC Safe Limit',
        line_color='#FF3B30', line_dash='dash'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True, height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)
