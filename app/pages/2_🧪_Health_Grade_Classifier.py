"""
Page 2 — Health Grade Risk Classifier
 
Pick a restaurant from the DOHMH held-out test set, hit the button, get the
estimated grade-risk category (A / B / C) with class probabilities.
 
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
 
This is a held-out restaurant-profile classifier, not a strict future-grade
forecasting model.  A future forecasting setup would use only inspections
before time t to predict the next inspection grade at time t.

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
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
 
from app.ui_utils import apply_apple_theme
from models.custom_mlp import CustomMLP, TrainingHistory, evaluate_mlp, train_mlp
from models.pca_scratch import PCAScratch
from utils.user_profile import init_session_state
 
 
# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------
 
st.set_page_config(page_title="Health Grade Risk Classifier", page_icon="🧪", layout="wide")
apply_apple_theme()
init_session_state()
 
from utils.auth import require_auth
require_auth()
 
st.title("🧪 Health Grade Risk Classifier")
st.markdown(
    "Pick a NYC restaurant from the held-out DOHMH test set and estimate whether "
    "its inspection profile looks more like Grade A, B, or C restaurants.  This "
    "is a risk classifier built from real inspection-history features, not a "
    "lookup table and not an official future-grade forecast."
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


FEATURE_INDEX = {name: idx for idx, name in enumerate(feature_cols)}
NUMERIC_FEATURES = [c for c in feature_config.get("numerical_features", []) if c in feature_cols]
SCALER_MEAN = dict(zip(feature_config.get("numerical_features", []),
                       feature_config.get("scaler_mean", [])))
SCALER_SCALE = dict(zip(feature_config.get("numerical_features", []),
                        feature_config.get("scaler_scale", [])))
BORO_COLS = [c for c in feature_cols if c.startswith("boro_")]
CUISINE_COLS = [c for c in feature_cols if c.startswith("cuisine_")]
ACTIONABLE_FEATURES = ["num_violations"]
COUNT_FEATURES = {"num_inspections", "num_violations"}
FEATURE_LABELS = {
    "num_inspections": "Number of inspections",
    "num_violations": "Total violations",
    "violations_per_inspection": "Violations per inspection",
}


def _raw_value(feature_name: str, standardized_value: float) -> float:
    """Convert a model-space standardized numeric feature back to app-friendly units."""
    return float(standardized_value * SCALER_SCALE[feature_name] + SCALER_MEAN[feature_name])


def _standardized_value(feature_name: str, raw_value: float) -> float:
    """Convert app-friendly raw units into the standardized value expected by the MLP."""
    return float((raw_value - SCALER_MEAN[feature_name]) / SCALER_SCALE[feature_name])


def _with_numeric_raw(feature_row: np.ndarray, feature_name: str, raw_value: float) -> np.ndarray:
    edited = feature_row.copy()
    edited[FEATURE_INDEX[feature_name]] = _standardized_value(feature_name, raw_value)
    return edited


def _with_violation_counts(feature_row: np.ndarray, total_violations: int,
                           num_inspections: int) -> np.ndarray:
    """Edit discrete counts and update the derived violation-rate feature."""
    inspections = max(int(num_inspections), 1)
    total = max(int(total_violations), 0)
    edited = _with_numeric_raw(feature_row, "num_violations", float(total))
    edited = _with_numeric_raw(edited, "num_inspections", float(inspections))
    if "violations_per_inspection" in FEATURE_INDEX:
        edited = _with_numeric_raw(edited, "violations_per_inspection", total / inspections)
    return edited


def _active_one_hot(feature_row: np.ndarray, columns: list[str]) -> str | None:
    if not columns:
        return None
    values = [feature_row[FEATURE_INDEX[c]] for c in columns]
    return columns[int(np.argmax(values))]


def _with_one_hot(feature_row: np.ndarray, columns: list[str], active_column: str) -> np.ndarray:
    edited = feature_row.copy()
    for col in columns:
        edited[FEATURE_INDEX[col]] = 1.0 if col == active_column else 0.0
    return edited


def _display_category(col_name: str, prefix: str) -> str:
    label = col_name.replace(prefix, "")
    return "Unknown" if label == "0" else label


@st.cache_data(show_spinner=False)
def get_numeric_raw_ranges():
    """Observed raw-value ranges from the prepared train/test feature matrix."""
    combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0)
    ranges = {}
    for col in NUMERIC_FEATURES:
        raw_values = combined[col].astype(float).map(lambda v: _raw_value(col, v)).to_numpy()
        lo = float(np.nanmin(raw_values))
        hi = float(np.nanmax(raw_values))
        if col in COUNT_FEATURES:
            lo = float(np.floor(lo))
            hi = float(np.ceil(hi))
        ranges[col] = {
            "min": max(0.0, lo),
            "max": max(0.0, hi),
            "median_a": float(np.nanmedian(
                train_df.loc[train_df["target"] == 0, col].astype(float).map(lambda v: _raw_value(col, v))
            )),
        }
    return ranges


def _grade_from_probs(probs: np.ndarray) -> str:
    return GRADE_NAMES[int(np.argmax(probs))]


def _format_raw_feature(feature_name: str, value: float) -> str:
    if feature_name in COUNT_FEATURES:
        return f"{int(round(value))}"
    return f"{value:.2f}"


def _probability_delta_rows(before: np.ndarray, after: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Grade": [f"Grade {g}" for g in GRADE_NAMES],
        "Original": before,
        "Edited": after,
    })


@st.cache_data(show_spinner=False)
def compute_pca_context():
    """Project the held-out feature vectors with our NumPy PCA implementation."""
    X_test = test_df[feature_cols].values.astype(np.float64)
    pca = PCAScratch(n_components=2)
    coords = pca.fit_transform(X_test)
    return {
        "coords": coords,
        "mean": pca.mean_,
        "components": pca.components_,
        "explained": pca.explained_variance_ratio_,
    }


def transform_with_cached_pca(feature_matrix: np.ndarray, pca_payload: dict) -> np.ndarray:
    return (np.asarray(feature_matrix, dtype=np.float64) - pca_payload["mean"]) @ pca_payload["components"].T


def local_sensitivity(feature_row: np.ndarray) -> pd.DataFrame:
    """Rank numeric features by how much moving toward the Grade-A median changes P(A)."""
    base_probs = predict_grade(feature_row)
    ranges = get_numeric_raw_ranges()
    rows = []
    for col in NUMERIC_FEATURES:
        current_raw = _raw_value(col, feature_row[FEATURE_INDEX[col]])
        target_raw = ranges[col]["median_a"]
        edited = _with_numeric_raw(feature_row, col, target_raw)
        edited_probs = predict_grade(edited)
        rows.append({
            "Feature": col,
            "Current": current_raw,
            "Reference": target_raw,
            "Delta P(A)": float(edited_probs[0] - base_probs[0]),
            "Actionable": "Yes" if col in ACTIONABLE_FEATURES else "Derived" if col == "violations_per_inspection" else "Context",
        })
    return pd.DataFrame(rows).sort_values("Delta P(A)", ascending=False)


def find_path_to_a(feature_row: np.ndarray):
    """Search for a realistic A-path by lowering integer violations with inspection history fixed."""
    if not all(col in FEATURE_INDEX for col in ACTIONABLE_FEATURES + ["num_inspections"]):
        return None

    inspections = max(_raw_value("num_inspections", feature_row[FEATURE_INDEX["num_inspections"]]), 1.0)
    inspections_int = max(int(round(inspections)), 1)
    current_total = max(int(round(_raw_value("num_violations",
                                             feature_row[FEATURE_INDEX["num_violations"]]))), 0)
    min_total = int(get_numeric_raw_ranges()["num_violations"]["min"])

    best = None
    for candidate_total in range(current_total, min_total - 1, -1):
        candidate_vpi = candidate_total / inspections_int
        edited = _with_violation_counts(feature_row, candidate_total, inspections_int)
        probs = predict_grade(edited)
        if _grade_from_probs(probs) == "A":
            best = {
                "feature_row": edited,
                "probs": probs,
                "violations_per_inspection": candidate_vpi,
                "num_violations": float(candidate_total),
            }
            break

    if best is not None:
        return best

    # Fallback: show the best in-range P(A) improvement even if the boundary is not crossed.
    candidate_total = min_total
    candidate_vpi = candidate_total / inspections_int
    edited = _with_violation_counts(feature_row, candidate_total, inspections_int)
    return {
        "feature_row": edited,
        "probs": predict_grade(edited),
        "violations_per_inspection": candidate_vpi,
        "num_violations": candidate_total,
        "fallback": True,
    }
 
 
# ---------------------------------------------------------------------------
# Permutation importance — cached because it takes a few seconds
# ---------------------------------------------------------------------------
 
@st.cache_data(show_spinner="Computing permutation importance...")
def compute_permutation_importance(n_repeats: int = 8):
    """Shuffle feature groups and measure weighted-F1 drop.

    Numeric features are shuffled one at a time.  One-hot categories are
    shuffled as groups so the perturbed rows still contain a valid borough or
    cuisine vector.  Weighted F1 is used instead of raw accuracy because the
    grade distribution is strongly imbalanced toward A.
    """
    metric_version = "weighted_f1_group_v2"
    if IMPORTANCE_CACHE_PATH.exists():
        try:
            with open(IMPORTANCE_CACHE_PATH) as f:
                payload = json.load(f)
            if payload.get("feature_cols") == feature_cols and payload.get("metric_version") == metric_version:
                return payload["baseline"], payload["importance"], payload["metric_name"]
        except Exception:
            pass

    X_test = tensors["X_test"].numpy()

    def weighted_f1(X):
        score, _ = evaluate_mlp(model, torch.from_numpy(X.astype(np.float32)), tensors["y_test"])
        return float(score)

    baseline = weighted_f1(X_test)
    groups = []
    for col in NUMERIC_FEATURES:
        groups.append((FEATURE_LABELS.get(col, col), [FEATURE_INDEX[col]]))
    if BORO_COLS:
        groups.append(("Borough one-hot group", [FEATURE_INDEX[c] for c in BORO_COLS]))
    if CUISINE_COLS:
        groups.append(("Cuisine one-hot group", [FEATURE_INDEX[c] for c in CUISINE_COLS]))
    if NUMERIC_FEATURES:
        groups.append(("All inspection-pattern numerics", [FEATURE_INDEX[c] for c in NUMERIC_FEATURES]))

    importance = {}
    rng = np.random.default_rng(42)
    for name, indices in groups:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm = rng.permutation(len(X_perm))
            X_perm[:, indices] = X_test[perm][:, indices]
            drops.append(baseline - weighted_f1(X_perm))
        importance[name] = float(np.mean(drops))

    payload = {
        "metric_version": metric_version,
        "metric_name": "weighted F1",
        "feature_cols": feature_cols,
        "baseline": baseline,
        "importance": importance,
    }
    try:
        with open(IMPORTANCE_CACHE_PATH, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass
    return baseline, importance, "weighted F1"
 
 
# ---------------------------------------------------------------------------
# Model summary banner
# ---------------------------------------------------------------------------
 
st.info(
    f"**Model:** 3-layer MLP ({input_dim} → 128 → 128 → 3), trained on "
    f"{len(train_df):,} NYC restaurants with held-out test set of "
    f"{len(test_df):,}.  Class-weighted CrossEntropy loss, AdamW optimizer, "
    f"early stopping on validation F1.  Best val F1: "
    f"{training_history.best_val_f1*100:.1f}% at epoch "
    f"{training_history.best_epoch+1}.  The app reports probabilities as "
    f"risk signals because the no-score feature set is intentionally leakage-reduced.",
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
 
        **Result:** held-out accuracy dropped to {
            'a realistic range' if input_dim < 28 else 'a still-suspiciously-high level'
        }
        — but the score is no longer inflated by a feature that directly
        determines the label.  The model now estimates grade-risk patterns
        from inspection counts, violation rate, borough, and cuisine.  It is
        a genuine ML classifier on held-out restaurant profiles, not a
        threshold lookup table.
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
top_probability = float(probs[pred_idx])
probability_note = "Concentrated probability" if top_probability >= 0.65 else "Moderately spread probabilities" if top_probability >= 0.50 else "Broad probability spread"
 
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
                Estimated risk category
            </div>
            <div style='font-size: 72px; font-weight: 700; color: {color}; line-height: 1;'>
                {pred_grade}
            </div>
            <div style='font-size: 16px; color: #444; margin-top: 6px;'>
                Top class probability: <b>{top_probability * 100:.1f}%</b>
            </div>
            <div style='font-size: 13px; color: #777; margin-top: 4px;'>
                {probability_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
    st.markdown(f"**Actual grade on record:** **{true_grade}**")
    if top_probability < 0.50:
        st.info(
            "The class probabilities are close together for this restaurant, so the "
            "profile sits near the model's decision boundary.  Read this as a risk "
            "distribution rather than an official grade assignment.",
            icon="ℹ️",
        )
    if is_correct:
        st.success("✅ Correct — the classifier matched the DOHMH grade.")
    else:
        st.warning(
            f"⚠️ Classifier predicted **{pred_grade}**, actual grade is **{true_grade}**.  "
            "Since we dropped score features (see explainer above), the model now has "
            "to estimate risk from coarse profile features — individual misses are expected."
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
# Section 3 — What-if explorer + actionable path to A
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("3️⃣ What-if Explorer")
st.caption(
    "Change integer inspection counts and watch the predicted grade update.  The "
    "violation rate is recalculated as total violations divided by inspections, then "
    "standardized before it enters the MLP. "
    "Borough and cuisine selectors are for correlation exploration only."
)

raw_ranges = get_numeric_raw_ranges()
edited_row = feature_row.copy()

col_sliders, col_profile = st.columns([1.15, 0.85])

with col_sliders:
    st.markdown("**Actionable inspection pattern**")
    current_total_raw = _raw_value("num_violations", feature_row[FEATURE_INDEX["num_violations"]])
    total_min = min(raw_ranges["num_violations"]["min"], current_total_raw)
    total_max = max(raw_ranges["num_violations"]["max"], current_total_raw)
    edited_total = st.slider(
        FEATURE_LABELS["num_violations"],
        min_value=int(np.floor(total_min)),
        max_value=int(np.ceil(total_max)),
        value=int(round(current_total_raw)),
        step=1,
        help=(
            "Integer count. Range uses the observed min/max in the prepared DOHMH "
            "train/test feature matrix."
        ),
    )

    if "num_inspections" in FEATURE_INDEX:
        st.markdown("**Historical context**")
        col = "num_inspections"
        current_raw = _raw_value(col, feature_row[FEATURE_INDEX[col]])
        min_value = min(raw_ranges[col]["min"], current_raw)
        max_value = max(raw_ranges[col]["max"], current_raw)
        value = st.slider(
            FEATURE_LABELS.get(col, "Number of inspections"),
            min_value=int(np.floor(min_value)),
            max_value=int(np.ceil(max_value)),
            value=int(round(current_raw)),
            step=1,
            help=(
                "This is context, not a direct improvement lever.  A restaurant cannot "
                "erase inspection history, but changing it shows how exposure affects the model. "
                "Range uses the observed min/max in the prepared train/test feature matrix."
            ),
        )
        edited_inspections = int(value)
    else:
        edited_inspections = 1
        min_value = max_value = 1

    edited_vpi = edited_total / max(edited_inspections, 1)
    edited_row = _with_violation_counts(edited_row, edited_total, edited_inspections)

    st.metric(
        "Derived violations per inspection",
        f"{edited_vpi:.2f}",
        help="Derived from the two integer sliders above: total violations / number of inspections.",
    )
    st.caption(
        "Variable selection rule: only model input features are exposed. Counts are sliders; "
        "the rate is derived; borough/cuisine remain selectors because they are one-hot categories."
    )
    st.caption(
        f"Current slider ranges: total violations {int(np.floor(total_min))}-"
        f"{int(np.ceil(total_max))}; inspections {int(np.floor(min_value))}-"
        f"{int(np.ceil(max_value))}. These are observed bounds in the prepared "
        "train/test data, widened when needed to include the selected restaurant."
    )

with col_profile:
    st.markdown("**Profile selectors**")
    active_boro = _active_one_hot(feature_row, BORO_COLS)
    if active_boro is not None:
        selected_boro_col = st.selectbox(
            "Borough",
            BORO_COLS,
            index=BORO_COLS.index(active_boro),
            format_func=lambda c: _display_category(c, "boro_"),
            help="Exploratory only. Borough is a context feature, not an action recommendation.",
        )
        edited_row = _with_one_hot(edited_row, BORO_COLS, selected_boro_col)

    active_cuisine = _active_one_hot(feature_row, CUISINE_COLS)
    if active_cuisine is not None:
        selected_cuisine_col = st.selectbox(
            "Cuisine Group",
            CUISINE_COLS,
            index=CUISINE_COLS.index(active_cuisine),
            format_func=lambda c: _display_category(c, "cuisine_"),
            help="Exploratory only. Cuisine captures group-level correlation, not a direct health intervention.",
        )
        edited_row = _with_one_hot(edited_row, CUISINE_COLS, selected_cuisine_col)

    st.info(
        "Use the sliders for realistic operational changes.  Use selectors to inspect "
        "dataset correlations, not as business advice.",
        icon="ℹ️",
    )

edited_probs = predict_grade(edited_row)
edited_grade = _grade_from_probs(edited_probs)
pa_delta = edited_probs[0] - probs[0]

col_before_after, col_delta = st.columns([1.2, 0.8])

with col_before_after:
    before_after = _probability_delta_rows(probs, edited_probs)
    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(
        x=before_after["Grade"], y=before_after["Original"],
        name="Original", marker_color="#8E8E93",
        text=[f"{v * 100:.1f}%" for v in before_after["Original"]],
        textposition="outside",
    ))
    compare_fig.add_trace(go.Bar(
        x=before_after["Grade"], y=before_after["Edited"],
        name="Edited", marker_color=[GRADE_COLORS[g] for g in GRADE_NAMES],
        text=[f"{v * 100:.1f}%" for v in before_after["Edited"]],
        textposition="outside",
    ))
    compare_fig.update_layout(
        title="Original vs edited class probabilities",
        barmode="group", yaxis=dict(range=[0, 1.1], title="P(grade)"),
        height=340, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(compare_fig, use_container_width=True)

with col_delta:
    st.metric("Edited Prediction", edited_grade, delta=f"from {pred_grade}")
    st.metric("P(A) Change", f"{pa_delta * 100:+.1f} pts")
    if edited_grade != pred_grade:
        st.success(f"The edited profile crosses the decision boundary: {pred_grade} → {edited_grade}.")
    else:
        st.caption("The probabilities changed, but the top predicted grade did not switch.")

sensitivity_df = local_sensitivity(feature_row)
st.markdown("**Main prediction drivers for this restaurant**")
st.caption(
    "Each row moves one numeric feature to the median value among Grade A restaurants "
    "and measures the change in P(A).  This is local sensitivity, not PCA variance."
)
driver_display = sensitivity_df.copy()
driver_display["Current"] = driver_display.apply(
    lambda row: _format_raw_feature(row["Feature"], row["Current"]), axis=1,
)
driver_display["Grade A Reference"] = driver_display.apply(
    lambda row: _format_raw_feature(row["Feature"], row["Reference"]), axis=1,
)
driver_display["Change in P(A)"] = driver_display["Delta P(A)"].map(lambda v: f"{v * 100:+.1f} pts")
st.dataframe(
    driver_display[["Feature", "Current", "Grade A Reference", "Change in P(A)", "Actionable"]],
    use_container_width=True,
    hide_index=True,
)

if pred_grade != "A" or true_grade != "A":
    st.markdown("**Path to A**")
    path = find_path_to_a(feature_row)
    if path is None:
        st.warning("This model does not have enough actionable numeric inputs to compute a path to A.")
    else:
        path_grade = _grade_from_probs(path["probs"])
        current_total = _raw_value("num_violations", feature_row[FEATURE_INDEX["num_violations"]])
        current_vpi = _raw_value("violations_per_inspection",
                                 feature_row[FEATURE_INDEX["violations_per_inspection"]])
        if path.get("fallback"):
            st.warning(
                "Even the most aggressive reduction in actionable violation features did not "
                "flip this profile to A.  The recommendation below shows the strongest model-implied move."
            )
        else:
            st.success(
                f"The smallest searched actionable move that flips the model to Grade A reaches "
                f"P(A) = {path['probs'][0] * 100:.1f}%."
            )
        rec_rows = pd.DataFrame([
            {
                "Actionable feature": "Total violations",
                "Current": _format_raw_feature("num_violations", current_total),
                "Suggested target": _format_raw_feature("num_violations", path["num_violations"]),
            },
            {
                "Actionable feature": "Violations per inspection",
                "Current": f"{current_vpi:.2f}",
                "Suggested target": f"{path['violations_per_inspection']:.2f}",
            },
        ])
        st.dataframe(rec_rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Recommended targets hold borough, cuisine, and inspection history fixed.  "
            f"The counterfactual prediction is Grade {path_grade}; actual DOHMH grades still "
            "depend on future inspections and official score thresholds."
        )

with st.expander("🧭 PCA context map: where this profile sits", expanded=False):
    st.caption(
        "This uses the project's NumPy PCA implementation to visualize the held-out feature "
        "space.  PCA shows broad data geometry; improvement advice above comes from the MLP "
        "counterfactual search, not from PCA alone."
    )
    pca_payload = compute_pca_context()
    coords = pca_payload["coords"]
    selected_matches = meta_test.index[meta_test["camis"] == selected["camis"]].tolist()
    selected_idx = selected_matches[0] if selected_matches else 0
    edited_coord = transform_with_cached_pca(edited_row.reshape(1, -1), pca_payload)[0]

    pca_fig = go.Figure()
    pca_fig.add_trace(go.Scattergl(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(
            size=6,
            color=[GRADE_COLORS.get(g, "#8E8E93") for g in meta_test["grade"].tolist()],
            opacity=0.45,
        ),
        text=meta_test["dba"],
        hovertemplate="%{text}<extra></extra>",
        name="Held-out restaurants",
    ))
    pca_fig.add_trace(go.Scatter(
        x=[coords[selected_idx, 0]],
        y=[coords[selected_idx, 1]],
        mode="markers+text",
        marker=dict(size=15, color="#111111", symbol="star"),
        text=["Original"],
        textposition="top center",
        name="Original profile",
    ))
    pca_fig.add_trace(go.Scatter(
        x=[edited_coord[0]],
        y=[edited_coord[1]],
        mode="markers+text",
        marker=dict(size=15, color="#007AFF", symbol="diamond"),
        text=["Edited"],
        textposition="bottom center",
        name="Edited profile",
    ))
    pca_fig.update_layout(
        title=(
            f"PCA projection of model inputs "
            f"(PC1 {pca_payload['explained'][0] * 100:.1f}%, "
            f"PC2 {pca_payload['explained'][1] * 100:.1f}% variance)"
        ),
        xaxis_title="PC1", yaxis_title="PC2",
        height=440, margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(pca_fig, use_container_width=True)
 
 
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
# Section 4 — Held-out test set performance + feature importance
# ---------------------------------------------------------------------------
 
st.markdown("---")
st.subheader("4️⃣ How Good is the Classifier — and Why?")
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


@st.cache_data(show_spinner=False)
def get_baseline_metrics():
    """Majority-class benchmark: always predict Grade A."""
    y_true = tensors["y_test"].numpy()
    baseline_preds = np.zeros_like(y_true)
    bc_mask = y_true != 0
    return {
        "accuracy": accuracy_score(y_true, baseline_preds),
        "weighted_f1": f1_score(y_true, baseline_preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, baseline_preds, average="macro", zero_division=0),
        "bc_recall": recall_score(y_true[bc_mask], baseline_preds[bc_mask],
                                  average="micro", zero_division=0) if bc_mask.any() else 0.0,
    }
 
 
details = get_test_metrics()
report = details["classification_report"]
baseline = get_baseline_metrics()
bc_recall = (report["B"]["recall"] + report["C"]["recall"]) / 2
 
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{report['accuracy'] * 100:.1f}%")
m2.metric("B/C Recall", f"{bc_recall:.3f}",
          help="Average recall for the two risky minority grades. The always-A baseline is 0.")
m3.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}",
          help="Unweighted mean of per-class F1 — fairer on rare grades.")
m4.metric("Test size", f"{len(test_df):,}")

benchmark_df = pd.DataFrame([
    {
        "Model": "Custom MLP",
        "Accuracy": f"{report['accuracy']:.3f}",
        "Weighted F1": f"{details['weighted_f1']:.3f}",
        "Macro F1": f"{report['macro avg']['f1-score']:.3f}",
        "B/C Recall": f"{bc_recall:.3f}",
    },
    {
        "Model": "Always predict A baseline",
        "Accuracy": f"{baseline['accuracy']:.3f}",
        "Weighted F1": f"{baseline['weighted_f1']:.3f}",
        "Macro F1": f"{baseline['macro_f1']:.3f}",
        "B/C Recall": f"{baseline['bc_recall']:.3f}",
    },
])
st.markdown("**Benchmark against a majority-class baseline**")
st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
st.caption(
    "Because Grade A dominates the data, an always-A baseline can score high raw "
    "accuracy while completely missing B/C restaurants.  We emphasize macro F1, "
    "minority-class recall, and the confusion matrix to show what the MLP learns "
    "beyond the trivial majority-class rule."
)
 
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
    "**Permutation importance** — we shuffle each numeric feature, or a whole "
    "one-hot group, across the held-out test set and measure how much weighted "
    "F1 drops.  Grouping borough/cuisine keeps one-hot vectors valid and is "
    "easier to interpret than ranking dozens of sparse category columns."
)
 
baseline_score, importance, metric_name = compute_permutation_importance()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=lambda kv: -kv[1]),
    columns=["Feature / group", "Metric Drop"],
)
importance_df["Metric Drop %"] = importance_df["Metric Drop"] * 100
top_features = importance_df.head(8)
 
imp_fig = go.Figure(go.Bar(
    x=top_features["Metric Drop %"],
    y=top_features["Feature / group"],
    orientation="h",
    marker_color=["#34C759" if d > 2 else "#FFCC00" if d > 0.5 else "#8E8E93"
                  for d in top_features["Metric Drop %"]],
    text=[f"{d:+.2f} pts" for d in top_features["Metric Drop %"]],
    textposition="outside",
))
imp_fig.update_layout(
    title=f"Permutation importance by feature group (baseline {metric_name} = {baseline_score:.3f})",
    xaxis=dict(title=f"Drop in {metric_name} when shuffled (percentage points)"),
    yaxis=dict(autorange="reversed"),
    height=380, margin=dict(l=20, r=40, t=60, b=40),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, -apple-system, sans-serif",
)
st.plotly_chart(imp_fig, use_container_width=True)
 
top_driver_name = importance_df.iloc[0]["Feature / group"]
top_driver_drop = importance_df.iloc[0]["Metric Drop %"]
weak_count = int((importance_df["Metric Drop %"].abs() < 0.5).sum())
st.caption(
    f"The strongest global driver is **{top_driver_name}** "
    f"({top_driver_drop:+.1f} percentage points of {metric_name}).  "
    f"{weak_count} groups are within ±0.5 points, which means their effect is "
    "small or noisy under this trained model.  Negative values can happen when "
    "a weak feature adds noise and shuffling it slightly improves the metric."
)
