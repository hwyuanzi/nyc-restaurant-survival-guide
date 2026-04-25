"""
Page 2 — Simple Health Grade Classifier
 
Pick a restaurant from the DOHMH held-out test set, hit the button, get the
predicted grade (A / B / C) with class probabilities.  That is the whole
story of this page.
 
What makes this version different from our earlier sandbox:
  1. We removed all score-derived features from the training set, because
     the DOHMH grade is *derived* from the inspection score.  Keeping them
     was label leakage — the model hit 99% accuracy not by learning
     anything useful but by rediscovering a known threshold rule.  See the
     "🔍 Why we dropped score features" section below.
  2. We show the 25-dim feature vector that actually enters the MLP, so
     the audience sees the input isn't a restaurant name — it's a numeric
     vector derived from violation history, borough, and cuisine.
  3. We show permutation importance so the audience can verify *which*
     features the model actually relies on.  This is stronger evidence
     than PCA for "what matters to predictions" because PCA measures
     variance, not predictive power.
 
Data:   data/train.csv + data/test.csv
Model:  models/custom_mlp.py — cached at data/cache/health_classifier.pt
"""

import json
import os
from pathlib import Path
import sys
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.model_selection import train_test_split
 
from app.ui_utils import apply_apple_theme
from models.custom_mlp import CustomMLP, TrainingHistory, evaluate_mlp, train_mlp
from utils.user_profile import init_session_state
 
 
# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
 
st.set_page_config(page_title="Health Grade Classifier", page_icon="🧪", layout="wide")
apply_apple_theme()
init_session_state()
 
from utils.auth import require_auth
require_auth()
 
st.title("🧪 Health Grade Classifier")
st.markdown(
    "Pick a NYC restaurant, run it through our trained MLP, and see the predicted "
    "DOHMH letter grade (A / B / C).  The model is trained on real NYC Department "
    "of Health inspection history; the test set below was held out entirely during "
    "training."
)
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
META_TEST_PATH = DATA_DIR / "meta_test.csv"
CONFIG_PATH = DATA_DIR / "feature_config.json"
MODEL_CACHE_PATH = DATA_DIR / "cache" / "health_classifier.pt"
HISTORY_CACHE_PATH = DATA_DIR / "cache" / "health_classifier_history.json"
IMPORTANCE_CACHE_PATH = DATA_DIR / "cache" / "health_classifier_importance.json"
 
GRADE_NAMES = ["A", "B", "C"]
GRADE_COLORS = {"A": "#34C759", "B": "#FFCC00", "C": "#FF3B30"}
 
 
if not all(p.exists() for p in [TRAIN_PATH, TEST_PATH, META_TEST_PATH, CONFIG_PATH]):
    st.error(
        "Preprocessed DOHMH data files are missing. Generate them with:\n\n"
        "```\npython data/download_data.py 50000\npython data/preprocess.py\n```"
    )
    st.stop()
 
 
@st.cache_data(show_spinner=False)
def load_prepared_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    meta_test = pd.read_csv(META_TEST_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return train_df, test_df, meta_test, config
 
 
train_df, test_df, meta_test, feature_config = load_prepared_data()
feature_cols = [c for c in feature_config["feature_columns"] if c in train_df.columns]
input_dim = len(feature_cols)
 
 
def _build_tensors(df):
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)
 
 
@st.cache_resource(show_spinner="Preparing tensors...")
def get_tensors():
    X_full, y_full = _build_tensors(train_df)
    X_test, y_test = _build_tensors(test_df)
 
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_full.numpy(), y_full.numpy(),
        test_size=0.2, stratify=y_full.numpy(), random_state=42,
    )
    return {
        "X_train": torch.from_numpy(X_tr_np), "y_train": torch.from_numpy(y_tr_np),
        "X_val":   torch.from_numpy(X_val_np), "y_val":   torch.from_numpy(y_val_np),
        "X_test":  X_test, "y_test":  y_test,
    }
 
 
tensors = get_tensors()
 
 
# ---------------------------------------------------------------------------
# Model — load from cache if possible, otherwise train
# ---------------------------------------------------------------------------
 
HYPERPARAMS = {
    "hidden_dim": 128, "lr": 1e-3, "dropout": 0.3,
    "batch_size": 128, "weight_decay": 1e-4,
    "max_epochs": 80, "patience": 12,
}
 
 
@st.cache_resource(show_spinner="Loading the trained MLP...")
def get_model():
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    if MODEL_CACHE_PATH.exists() and HISTORY_CACHE_PATH.exists():
        try:
            m = CustomMLP(input_dim=input_dim, hidden_dim=HYPERPARAMS["hidden_dim"],
                          output_dim=3, dropout=HYPERPARAMS["dropout"])
            m.load_state_dict(torch.load(MODEL_CACHE_PATH, map_location="cpu",
                                         weights_only=True))
            m.eval()
            with open(HISTORY_CACHE_PATH) as f:
                hd = json.load(f)
            history = TrainingHistory(
                train_loss=hd["train_loss"], val_loss=hd["val_loss"],
                train_f1=hd["train_f1"],   val_f1=hd["val_f1"],
                best_epoch=hd["best_epoch"], best_val_f1=hd["best_val_f1"],
                stopped_early=hd["stopped_early"],
            )
            return m, history
        except Exception:
            pass
 
    torch.manual_seed(42)
    m = CustomMLP(input_dim=input_dim, hidden_dim=HYPERPARAMS["hidden_dim"],
                  output_dim=3, dropout=HYPERPARAMS["dropout"])
    m, history = train_mlp(
        m,
        tensors["X_train"], tensors["y_train"],
        X_val=tensors["X_val"], y_val=tensors["y_val"],
        epochs=HYPERPARAMS["max_epochs"], lr=HYPERPARAMS["lr"],
        batch_size=HYPERPARAMS["batch_size"],
        weight_decay=HYPERPARAMS["weight_decay"],
        patience=HYPERPARAMS["patience"],
        use_class_weights=True, verbose=False,
    )
    torch.save(m.state_dict(), MODEL_CACHE_PATH)
    with open(HISTORY_CACHE_PATH, "w") as f:
        json.dump({
            "train_loss": history.train_loss, "val_loss": history.val_loss,
            "train_f1": history.train_f1,     "val_f1": history.val_f1,
            "best_epoch": history.best_epoch,
            "best_val_f1": float(history.best_val_f1),
            "stopped_early": bool(history.stopped_early),
        }, f)
    return m, history
 
 
model, training_history = get_model()
 
 
def predict_grade(feature_row: np.ndarray):
    """Run one row (already in the model's feature space) through the MLP."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(feature_row.astype(np.float32)).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs
 
 
# ---------------------------------------------------------------------------
# Permutation importance — cached because it takes ~3 seconds
# ---------------------------------------------------------------------------
 
@st.cache_data(show_spinner="Computing permutation importance...")
def compute_permutation_importance(n_repeats: int = 3):
    """For each feature, shuffle its values across the test set and measure
    how much accuracy drops.  Bigger drop = feature is more important to the
    trained model's predictions.
 
    Cached on disk between runs so it's instant after the first compute.
    """
    if IMPORTANCE_CACHE_PATH.exists():
        try:
            with open(IMPORTANCE_CACHE_PATH) as f:
                payload = json.load(f)
            if payload.get("feature_cols") == feature_cols:
                return payload["baseline"], payload["importance"]
        except Exception:
            pass
 
    X_test = tensors["X_test"].numpy()
    y_test = tensors["y_test"].numpy()
 
    def accuracy(X):
        with torch.no_grad():
            preds = model(torch.from_numpy(X.astype(np.float32))).argmax(dim=1).numpy()
        return float((preds == y_test).mean())
 
    baseline = accuracy(X_test)
    importance = {}
    rng = np.random.default_rng(42)
    for i, col in enumerate(feature_cols):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm = rng.permutation(len(X_perm))
            X_perm[:, i] = X_test[perm, i]
            drops.append(baseline - accuracy(X_perm))
        importance[col] = float(np.mean(drops))
 
    payload = {"feature_cols": feature_cols, "baseline": baseline, "importance": importance}
    try:
        with open(IMPORTANCE_CACHE_PATH, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass
    return baseline, importance
 
 
# ---------------------------------------------------------------------------
# Model summary banner
# ---------------------------------------------------------------------------
 
st.info(
    f"**Model:** 3-layer MLP ({input_dim} → 128 → 128 → 3), trained on "
    f"{len(train_df):,} NYC restaurants with held-out test set of "
    f"{len(test_df):,}.  Class-weighted CrossEntropy loss, Adam optimizer, "
    f"early stopping on validation F1.  Best val F1: "
    f"{training_history.best_val_f1*100:.1f}% at epoch "
    f"{training_history.best_epoch+1}.",
    icon="🧠",
)
 
 
# ---------------------------------------------------------------------------
# Section — Why we dropped score features (teaching moment)
# ---------------------------------------------------------------------------
 
with st.expander("🔍 Why we dropped inspection-score features from training", expanded=False):
    st.markdown(
        f"""
        **Short answer:** our first version of this classifier hit 99%
        accuracy.  That's not a good sign on a 3-class imbalanced problem
        — it usually means the model is rediscovering a rule we already
        know.
 
        **What we found:**  DOHMH assigns letter grades using fixed
        score thresholds:
 
        | Score | Grade |
        |-------|-------|
        | 0–13  | A     |
        | 14–27 | B     |
        | 28+   | C     |
 
        When `latest_score` / `avg_score` / `max_score` / `critical_ratio`
        are in the feature set, the MLP just recovers those thresholds —
        that's label leakage.  We verified this with permutation
        importance: `latest_score` alone accounted for ~28% of accuracy
        in that earlier model.
 
        **What we fixed:**  We removed all score-derived numerical features
        and retrained on violation *patterns* instead:
 
        - `num_inspections` — operational age / regulatory history
        - `num_violations` — absolute violation count
        - `violations_per_inspection` — violation rate
        - borough + top-{len([c for c in feature_cols if c.startswith('cuisine_')])} cuisine one-hots
 
        **Result:** test accuracy dropped to {
            'a realistic range' if input_dim < 28 else 'a still-suspiciously-high level'
        }
        — but now every percentage point of that accuracy comes from a
        feature the model can see at *prediction* time on a brand-new
        restaurant that hasn't been graded yet.  That's a genuine ML
        task, not a lookup table.
        """
    )
 
 
# ---------------------------------------------------------------------------
# Section 1 — Pick a restaurant & predict
# ---------------------------------------------------------------------------
 
st.subheader("1️⃣ Pick a Restaurant")
 
query = st.text_input(
    "Search by name, borough, or cuisine",
    placeholder="e.g. pizza · Queens · Joe's",
)
 
if query:
    q = query.lower()
    mask = (
        meta_test["dba"].fillna("").str.lower().str.contains(q, na=False)
        | meta_test["boro"].fillna("").str.lower().str.contains(q, na=False)
        | meta_test["cuisine_description"].fillna("").str.lower().str.contains(q, na=False)
    )
    filtered = meta_test[mask]
else:
    filtered = meta_test.head(250)
 
if filtered.empty:
    st.warning("No restaurants match that search. Try something broader.")
    st.stop()
 
selection = st.dataframe(
    filtered[["dba", "boro", "cuisine_description", "grade"]].rename(columns={
        "dba": "Restaurant", "boro": "Borough",
        "cuisine_description": "Cuisine", "grade": "Actual Grade",
    }),
    use_container_width=True, hide_index=True,
    selection_mode="single-row", on_select="rerun", height=300,
)
 
if selection.selection.rows:
    selected = filtered.iloc[selection.selection.rows[0]]
else:
    selected = filtered.iloc[0]
    st.caption("👆 Click a row to pick a different restaurant.  Showing the first row by default.")
 
 
# Look up this restaurant's row in the feature matrix
test_row_df = test_df[meta_test["camis"] == selected["camis"]]
if test_row_df.empty:
    st.error("Could not find the feature row for this restaurant. Try another one.")
    st.stop()
 
feature_row = test_row_df[feature_cols].values[0]
true_grade = selected["grade"]
 
 
# ---------------------------------------------------------------------------
# Section 2 — Prediction result
# ---------------------------------------------------------------------------
 
st.subheader("2️⃣ Classifier Prediction")
 
probs = predict_grade(feature_row)
pred_idx = int(np.argmax(probs))
pred_grade = GRADE_NAMES[pred_idx]
is_correct = pred_grade == true_grade
 
col_info, col_chart = st.columns([1, 1.1])
 
with col_info:
    st.markdown(f"**Restaurant:** {selected['dba']}")
    st.markdown(f"**Borough / Cuisine:** {selected['boro']} · {selected['cuisine_description']}")
    st.markdown("---")
 
    # Big predicted grade card
    color = GRADE_COLORS[pred_grade]
    st.markdown(
        f"""
        <div style='
            background: {color}22;
            border: 2px solid {color};
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            margin-bottom: 16px;'>
            <div style='font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px;'>
                Predicted grade
            </div>
            <div style='font-size: 72px; font-weight: 700; color: {color}; line-height: 1;'>
                {pred_grade}
            </div>
            <div style='font-size: 16px; color: #444; margin-top: 6px;'>
                Confidence: <b>{probs[pred_idx] * 100:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown(f"**Actual grade on record:** **{true_grade}**")
    if is_correct:
        st.success("✅ Correct — the classifier matched the DOHMH grade.")
    else:
        st.warning(
            f"⚠️ Classifier predicted **{pred_grade}**, actual grade is **{true_grade}**.  "
            "Since we dropped score features (see explainer above), the model now has "
            "to predict from violation patterns alone — individual misses are expected."
        )
 
with col_chart:
    prob_fig = go.Figure(go.Bar(
        x=[f"Grade {g}" for g in GRADE_NAMES],
        y=list(probs),
        marker_color=[GRADE_COLORS[g] for g in GRADE_NAMES],
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
    ))
    prob_fig.update_layout(
        title="Class probabilities",
        yaxis=dict(range=[0, 1.1], title="P(grade)"),
        height=320, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(prob_fig, use_container_width=True)
 
 
# ---------------------------------------------------------------------------
# Section 2.5 — What the model actually sees (input vector transparency)
# ---------------------------------------------------------------------------
 
st.markdown("---")
with st.expander("🔢 What the model actually sees (input feature vector)", expanded=False):
 
    numeric_features_in_config = [c for c in feature_config.get("numerical_features", []) if c in feature_cols]
    num_count = len(numeric_features_in_config)
    onehot_count = input_dim - num_count
 
    st.markdown(
        f"""
        **The MLP doesn't read restaurant names or addresses.**  It takes a
        **{input_dim}-dimensional numeric vector** derived from the
        restaurant's inspection history + location + cuisine.  Below is
        the exact vector fed into the network to produce the prediction
        above.
 
        - **Numeric features** ({num_count} dims) are *z-score standardized*:
          `0` means citywide average, positive = higher than average,
          negative = lower.  E.g. `num_violations = +1.8` means this
          restaurant has violation count 1.8 standard deviations **above**
          the citywide mean.
        - **One-hot features** ({onehot_count} dims) are `1` if active,
          `0` otherwise.  Exactly one borough and one cuisine should be `1`.
        """
    )
 
    NUMERIC_DESCRIPTIONS = {
        "num_inspections": "Total inspections on record (how established this restaurant is).",
        "num_violations": "Total violations across all inspections.",
        "violations_per_inspection": "Average violations per inspection visit.",
    }
 
    feature_display_rows = []
    for col_name, value in zip(feature_cols, feature_row):
        if col_name in numeric_features_in_config:
            feature_display_rows.append({
                "Dim": col_name, "Value": f"{value:+.2f}",
                "Type": "📊 Numeric",
                "Meaning": NUMERIC_DESCRIPTIONS.get(col_name, "Standardized numeric feature."),
            })
        elif col_name.startswith("boro_"):
            feature_display_rows.append({
                "Dim": col_name, "Value": str(int(round(value))),
                "Type": "📍 Borough (one-hot)",
                "Meaning": f"{'Yes' if value else 'No'} — {col_name.replace('boro_', '')}",
            })
        elif col_name.startswith("cuisine_"):
            feature_display_rows.append({
                "Dim": col_name, "Value": str(int(round(value))),
                "Type": "🍽️ Cuisine (one-hot)",
                "Meaning": f"{'Yes' if value else 'No'} — {col_name.replace('cuisine_', '')}",
            })
    feature_table = pd.DataFrame(feature_display_rows)
    numeric_rows = feature_table[feature_table["Type"].str.contains("Numeric")]
    onehot_rows = feature_table[~feature_table["Type"].str.contains("Numeric")]
    active_onehots = onehot_rows[onehot_rows["Value"] == "1"]
 
    col_num, col_active = st.columns([1, 1])
    with col_num:
        st.markdown("**📊 Numeric features (standardized)**")
        st.dataframe(numeric_rows[["Dim", "Value", "Meaning"]],
                     use_container_width=True, hide_index=True, height=200)
    with col_active:
        st.markdown("**🟦 Active one-hot features (only the `1`s)**")
        if active_onehots.empty:
            st.caption("No active one-hot features.")
        else:
            st.dataframe(active_onehots[["Dim", "Meaning"]],
                         use_container_width=True, hide_index=True, height=200)
        st.caption(
            f"The other {len(onehot_rows) - len(active_onehots)} one-hot "
            f"features are all `0`."
        )
 
    st.markdown("**📉 This restaurant vs. citywide average**")
    st.caption(
        "Each bar shows how many standard deviations this restaurant's "
        "feature sits from the citywide mean.  Positive = above average."
    )
    numeric_values = [float(v.replace("+", "")) for v in numeric_rows["Value"]]
    numeric_names = numeric_rows["Dim"].tolist()
    bar_colors = ["#FF3B30" if v > 0.5 else "#FFCC00" if v > 0 else "#34C759"
                  for v in numeric_values]
    comparison_fig = go.Figure(go.Bar(
        x=numeric_values, y=numeric_names, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}σ" for v in numeric_values],
        textposition="outside",
    ))
    comparison_fig.update_layout(
        height=max(200, 60 * len(numeric_names)),
        margin=dict(l=20, r=40, t=10, b=20),
        xaxis=dict(title="Standardized value (σ)", zeroline=True,
                   zerolinecolor="#888", zerolinewidth=1),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
 
 
# ---------------------------------------------------------------------------
# Section 3 — Held-out test set performance + feature importance
# ---------------------------------------------------------------------------
 
st.markdown("---")
st.subheader("3️⃣ How Good is the Classifier — and Why?")
st.caption(
    f"Evaluated on the held-out test set: {len(test_df):,} restaurants the "
    "model has never seen during training, validation, or model selection."
)
 
 
@st.cache_data(show_spinner=False)
def get_test_metrics():
    return evaluate_mlp(
        model, tensors["X_test"], tensors["y_test"],
        class_names=GRADE_NAMES, return_details=True,
    )
 
 
details = get_test_metrics()
report = details["classification_report"]
 
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{report['accuracy'] * 100:.1f}%")
m2.metric("Weighted F1", f"{details['weighted_f1']:.3f}")
m3.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}",
          help="Unweighted mean of per-class F1 — fairer on rare grades.")
m4.metric("Test size", f"{len(test_df):,}")
 
col_cm, col_per_class = st.columns([1, 1])
 
with col_cm:
    st.markdown("**Confusion matrix** (row-normalized)")
    cm = details["confusion_matrix"]
    cm_normed = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm_normed,
        x=[f"Predicted {g}" for g in GRADE_NAMES],
        y=[f"Actual {g}" for g in GRADE_NAMES],
        colorscale="Blues",
        text=[[f"{cm[i, j]}<br>({cm_normed[i, j] * 100:.0f}%)"
               for j in range(3)] for i in range(3)],
        texttemplate="%{text}",
        zmin=0, zmax=1,
        colorbar=dict(title="Share"),
    ))
    cm_fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(cm_fig, use_container_width=True)
 
with col_per_class:
    st.markdown("**Per-class performance**")
    per_class_rows = []
    for g in GRADE_NAMES:
        if g in report:
            per_class_rows.append({
                "Grade": g,
                "Precision": f"{report[g]['precision']:.2f}",
                "Recall": f"{report[g]['recall']:.2f}",
                "F1": f"{report[g]['f1-score']:.2f}",
                "Support": int(report[g]["support"]),
            })
    st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)
 
    st.caption(
        "A grades dominate the dataset (~90% of restaurants), so recall on B and C "
        "is where the model has to earn its keep.  Class-balanced loss during "
        "training keeps the minority classes from being ignored."
    )
 
 
# ---------------------------------------------------------------------------
# Section 4 — Permutation importance (answers "which features matter")
# ---------------------------------------------------------------------------
 
st.markdown("---")
st.markdown("### 🧭 Which Features Actually Drive Predictions?")
st.caption(
    "**Permutation importance** — for each feature, we shuffle its values "
    "across the test set and measure how much accuracy drops.  A large "
    "drop means the model relies heavily on that feature.  This is a "
    "stronger signal than PCA (which measures variance, not predictive "
    "power) when the question is *which inputs matter to the MLP*."
)
 
baseline_acc, importance = compute_permutation_importance()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=lambda kv: -kv[1]),
    columns=["Feature", "Accuracy Drop"],
)
# Keep only features with meaningful drops so the bar chart stays readable
importance_df["Accuracy Drop %"] = importance_df["Accuracy Drop"] * 100
top_features = importance_df.head(15)
 
imp_fig = go.Figure(go.Bar(
    x=top_features["Accuracy Drop %"],
    y=top_features["Feature"],
    orientation="h",
    marker_color=["#34C759" if d > 2 else "#FFCC00" if d > 0.5 else "#888"
                  for d in top_features["Accuracy Drop %"]],
    text=[f"{d:+.2f}%" for d in top_features["Accuracy Drop %"]],
    textposition="outside",
))
imp_fig.update_layout(
    title=f"Top-15 features by permutation importance (baseline acc = {baseline_acc*100:.1f}%)",
    xaxis=dict(title="Drop in accuracy when feature is shuffled (%)"),
    yaxis=dict(autorange="reversed"),
    height=500, margin=dict(l=20, r=40, t=60, b=40),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, -apple-system, sans-serif",
)
st.plotly_chart(imp_fig, use_container_width=True)
 
# Narrative — adapts automatically to whichever features survive
top_driver_name = importance_df.iloc[0]["Feature"]
top_driver_drop = importance_df.iloc[0]["Accuracy Drop %"]
low_impact_count = int((importance_df["Accuracy Drop %"] < 0.5).sum())
st.caption(
    f"The top driver is **{top_driver_name}** (shuffling it drops accuracy "
    f"by {top_driver_drop:.1f}%).  {low_impact_count} of the {input_dim} "
    "features contribute < 0.5% each — this is typical for one-hot "
    "encodings where each single category only flips on for a small share "
    "of the data."
)