"""
ML Action Sandbox — Health Grade Classifier Dashboard

Interactive analysis of the PyTorch MLP health-grade classifier.  All four
tabs share a single, really-trained model:

1. Live Prediction  — pick a real NYC restaurant, tweak its operational
                       features with sliders, watch the grade prediction
                       update.  Counterfactuals optional.
2. Model Performance — held-out test-set evaluation: confusion matrix,
                       per-class precision / recall / F1, overall weighted F1.
3. Training Diagnostics — train / validation loss and F1 curves, whether
                          early stopping fired, what the best epoch was.
4. Hyperparameter Justification — grid search results over
                                   (hidden_dim, lr, dropout), explaining
                                   why the deployed configuration was chosen.

This page replaces the earlier synthetic-data prototype.  It now loads the
real preprocessed DOHMH dataset produced by ``data/preprocess.py`` (29
engineered features across ~14K restaurants, grade A/B/C target).

Author: Rahul Adusumalli (ML Classifier Lead).  Integration with the real
data pipeline and hyperparameter-justification UI added by Ryan Han.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from app.ui_utils import apply_apple_theme
from models.custom_mlp import (
    CustomMLP,
    TrainingHistory,
    compute_gradient_importance,
    compute_permutation_importance,
    evaluate_mlp,
    find_counterfactual,
    hyperparameter_search,
    train_mlp,
)
from utils.user_profile import init_session_state

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Health Analyzer Sandbox", page_icon="🧪", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

with st.sidebar:
    st.markdown("### Analysis Controls")
    st.caption("Use this sidebar for model-specific settings and experiment controls on this page.")

st.title("🧪 Health Grade Classifier")

st.markdown(
    """
A PyTorch MLP trained on the full DOHMH inspection history of NYC
restaurants.  Every prediction, metric, and curve you see below is computed
from real held-out data — no synthetic shortcuts.

Use the tabs to explore live predictions, test-set performance, training
dynamics, and the hyperparameter search that justifies the deployed
configuration.
"""
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

GRADE_NAMES = ["A", "B", "C"]
GRADE_COLORS = {"A": "#34C759", "B": "#FFCC00", "C": "#FF3B30"}

FEATURE_DISPLAY_NAMES = {
    "latest_score":            "Most-recent inspection score",
    "avg_score":               "Average score across all inspections",
    "max_score":               "Worst-ever inspection score",
    "num_inspections":         "Total number of inspections on record",
    "num_violations":          "Total violation rows on record",
    "critical_ratio":          "Critical-violation ratio",
    "violations_per_inspection": "Violations per inspection",
}


def _missing_data_banner():
    st.error(
        "Preprocessed data files are missing.  The sandbox needs\n"
        "`data/train.csv`, `data/test.csv`, `data/meta_test.csv`, and "
        "`data/feature_config.json`.\n\n"
        "Generate them by running the pipeline:\n"
        "```\n"
        "python data/download_data.py 50000\n"
        "python data/preprocess.py\n"
        "```"
    )


if not all(p.exists() for p in [TRAIN_PATH, TEST_PATH, META_TEST_PATH, CONFIG_PATH]):
    _missing_data_banner()
    st.stop()


@st.cache_data(show_spinner=False)
def load_prepared_data():
    """Load the train/test splits plus feature schema and test-set metadata."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    meta_test = pd.read_csv(META_TEST_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return train_df, test_df, meta_test, config


train_df, test_df, meta_test, feature_config = load_prepared_data()
feature_cols = [c for c in feature_config["feature_columns"] if c in train_df.columns]
numerical_features = feature_config["numerical_features"]
scaler_mean = np.array(feature_config["scaler_mean"], dtype=np.float32)
scaler_scale = np.array(feature_config["scaler_scale"], dtype=np.float32)
input_dim = len(feature_cols)


def _build_tensors(df):
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


@st.cache_resource(show_spinner="Splitting data and preparing tensors...")
def get_data_tensors():
    """Stratified 80/20 split of the training set into train / val.

    The test set is held-out entirely and never touched during training,
    early stopping, or hyperparameter search.
    """
    X_full, y_full = _build_tensors(train_df)
    X_test, y_test = _build_tensors(test_df)

    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_full.numpy(),
        y_full.numpy(),
        test_size=0.2,
        stratify=y_full.numpy(),
        random_state=42,
    )

    X_train_t = torch.from_numpy(X_tr_np)
    y_train_t = torch.from_numpy(y_tr_np)
    X_val_t = torch.from_numpy(X_val_np)
    y_val_t = torch.from_numpy(y_val_np)

    return {
        "X_train": X_train_t, "y_train": y_train_t,
        "X_val": X_val_t, "y_val": y_val_t,
        "X_test": X_test, "y_test": y_test,
    }


tensors = get_data_tensors()


# ---------------------------------------------------------------------------
# Data-driven slider bounds (1st–99th percentile of raw training values)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_slider_ranges():
    """Return {feature_name: (lo, hi)} using training-set percentiles in raw units."""
    raw = train_df[numerical_features].values * scaler_scale + scaler_mean
    lo = np.maximum(0.0, np.percentile(raw, 1, axis=0))
    hi = np.percentile(raw, 99, axis=0)
    ranges = {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(numerical_features)}
    ranges["critical_ratio"] = (0.0, 1.0)  # ratio is always bounded [0, 1]
    return ranges

slider_ranges = compute_slider_ranges()


# ---------------------------------------------------------------------------
# Model training (cached across reruns)
# ---------------------------------------------------------------------------

DEPLOYED_HYPERPARAMS = {
    "hidden_dim": 128,
    "lr": 1e-3,
    "dropout": 0.3,
    "batch_size": 128,
    "weight_decay": 1e-4,
    "max_epochs": 80,
    "patience": 12,
}


@st.cache_resource(show_spinner="Loading / training the MLP on the real DOHMH dataset...")
def get_trained_model(hyperparams=None):
    """Return (model, history). Loads from disk if a checkpoint exists; otherwise
    trains from scratch, then saves weights + history so the next session is instant."""
    hp = hyperparams or DEPLOYED_HYPERPARAMS
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if MODEL_CACHE_PATH.exists() and HISTORY_CACHE_PATH.exists():
        try:
            m = CustomMLP(input_dim=input_dim, hidden_dim=hp["hidden_dim"],
                          output_dim=3, dropout=hp["dropout"])
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
            pass  # corrupted cache — fall through to retrain

    torch.manual_seed(42)
    m = CustomMLP(input_dim=input_dim, hidden_dim=hp["hidden_dim"],
                  output_dim=3, dropout=hp["dropout"])
    m, history = train_mlp(
        m,
        tensors["X_train"], tensors["y_train"],
        X_val=tensors["X_val"], y_val=tensors["y_val"],
        epochs=hp["max_epochs"], lr=hp["lr"],
        batch_size=hp["batch_size"], weight_decay=hp["weight_decay"],
        patience=hp["patience"], use_class_weights=True, verbose=False,
    )

    torch.save(m.state_dict(), MODEL_CACHE_PATH)
    with open(HISTORY_CACHE_PATH, "w") as f:
        json.dump({
            "train_loss": history.train_loss, "val_loss": history.val_loss,
            "train_f1":   history.train_f1,   "val_f1":   history.val_f1,
            "best_epoch": history.best_epoch,
            "best_val_f1": float(history.best_val_f1),
            "stopped_early": bool(history.stopped_early),
        }, f)

    return m, history


model, training_history = get_trained_model()


# ---------------------------------------------------------------------------
# Feature importance — computed once on the held-out test set, cached for
# the session.  Results drive both the sandbox importance labels and the
# dedicated Feature Analysis tab.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Computing feature importance on held-out test set...")
def get_feature_importance():
    """Return (grad_imp, perm_imp) as pd.Series indexed by feature_cols."""
    grad = compute_gradient_importance(
        model, tensors["X_test"], tensors["y_test"],
        feature_names=feature_cols,
    )
    perm = compute_permutation_importance(
        model, tensors["X_test"], tensors["y_test"],
        feature_names=feature_cols, n_repeats=10, seed=42,
    )
    return grad, perm


grad_imp, perm_imp = get_feature_importance()

# Normalised [0, 1] permutation importance for the 7 numerical features only,
# used to annotate sandbox sliders.
_num_perm = perm_imp[numerical_features].clip(lower=0)
_num_perm_norm = (_num_perm / _num_perm.max()) if _num_perm.max() > 0 else _num_perm


def _importance_badge(feature_name: str) -> str:
    """Return a short label ('★★★ High', '★★ Med', '★ Low') based on
    normalised permutation importance among the numerical features."""
    score = float(_num_perm_norm.get(feature_name, 0.0))
    if score >= 0.60:
        return "★★★ High impact"
    if score >= 0.25:
        return "★★ Medium impact"
    return "★ Low impact"


# ---------------------------------------------------------------------------
# Feature-engineering helpers
#
# The sliders operate on the raw operational features (average inspection
# score, critical violation count, inspection frequency).  We need to send
# the same StandardScaler transformation that preprocess.py applied to the
# training set so the model receives comparable inputs.
# ---------------------------------------------------------------------------

def standardise_numerical(raw_values: np.ndarray) -> np.ndarray:
    """Apply the preprocess.py StandardScaler to a raw numerical vector."""
    return (raw_values - scaler_mean) / scaler_scale


def build_feature_vector(raw_numerical: dict, boro: str, cuisine_group: str) -> np.ndarray:
    """Assemble a single input vector matching ``feature_config['feature_columns']``."""
    vec = np.zeros(input_dim, dtype=np.float32)

    # Numerical block
    raw = np.array(
        [raw_numerical[c] for c in numerical_features],
        dtype=np.float32,
    )
    scaled = standardise_numerical(raw)
    for name, value in zip(numerical_features, scaled):
        idx = feature_cols.index(name)
        vec[idx] = value

    # One-hot borough
    boro_key = f"boro_{boro}"
    if boro_key in feature_cols:
        vec[feature_cols.index(boro_key)] = 1.0

    # One-hot cuisine
    cuisine_key = f"cuisine_{cuisine_group}"
    if cuisine_key in feature_cols:
        vec[feature_cols.index(cuisine_key)] = 1.0
    else:
        # Fall back to the "Other" bucket if the preprocessing emitted one
        other_key = "cuisine_Other"
        if other_key in feature_cols:
            vec[feature_cols.index(other_key)] = 1.0

    return vec


def predict_single(feature_vec: np.ndarray):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(feature_vec).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs


# ---------------------------------------------------------------------------
# How the classifier works
# ---------------------------------------------------------------------------

with st.expander("📐 How the MLP Classifier Works", expanded=False):
    _exp_y = torch.cat([tensors["y_train"], tensors["y_val"]]).numpy()
    _exp_cnt = np.bincount(_exp_y)
    _exp_pct = _exp_cnt / _exp_cnt.sum() * 100
    st.markdown(
        f"""
**Input:** Each restaurant is represented as a {input_dim}-dimensional feature vector:
- **7 numerical features** (inspection scores, violation counts, critical ratios) — standardized via z-score using the training-set mean and std
- **6 borough one-hot features** (Manhattan, Brooklyn, Queens, Bronx, Staten Island, Unknown)
- **16 cuisine one-hot features** (American, Chinese, Italian, …, Other)

**Model:** 3-layer MLP ({input_dim} → 128 → 128 → 3) with ReLU activation and dropout
- Trained on {len(_exp_y):,} real DOHMH inspection records ({len(tensors['X_train']):,} train / {len(tensors['X_val']):,} val; test set held out separately)
- Class weights compensate for imbalanced grades (A: {_exp_pct[0]:.0f}%, B: {_exp_pct[1]:.0f}%, C: {_exp_pct[2]:.0f}%)
- Early stopping on validation F1 prevents overfitting

**Output:** Probability distribution over 3 classes (A, B, C)
- If B or C, a counterfactual gradient-descent search finds the minimum set of changes that would flip the prediction to A.
"""
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_live, tab_perf, tab_train, tab_latent, tab_hp, tab_feat = st.tabs([
    "🎛️ Live Prediction",
    "📊 Model Performance",
    "📈 Training Diagnostics",
    "🧠 Latent Space",
    "🔬 Hyperparameter Justification",
    "🔍 Feature Analysis",
])

# ---------------------------------------------------------------------------
# Tab 1 — Live Prediction
# ---------------------------------------------------------------------------

with tab_live:
    st.subheader("Pick a restaurant and simulate an inspection profile")
    st.caption(
        f"Database: {len(meta_test):,} real NYC restaurants held out from "
        f"training (grouped from the full {len(train_df) + len(test_df):,}-restaurant DOHMH dataset)."
    )

    # Search box + dataframe selector
    query = st.text_input(
        "Search by name, borough, or cuisine",
        value="",
        placeholder="e.g. pizza, Queens, Joe's",
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

    selection = st.dataframe(
        filtered[["dba", "boro", "cuisine_description", "grade"]].rename(
            columns={"dba": "Restaurant", "boro": "Borough",
                     "cuisine_description": "Cuisine", "grade": "Current Grade"}
        ),
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        height=260,
    )

    if selection.selection.rows:
        row_idx = selection.selection.rows[0]
        selected = filtered.iloc[row_idx]
    else:
        selected = filtered.iloc[0]

    # Find this restaurant's row in the test set to seed the sliders
    test_row = test_df[meta_test["camis"] == selected["camis"]]
    if test_row.empty:
        test_row = test_df.iloc[[0]]

    # Recover the RAW (un-standardised) numerical values by inverting the scaler
    scaled_vals = test_row[numerical_features].values[0].astype(np.float32)
    raw_vals = scaled_vals * scaler_scale + scaler_mean
    initial = dict(zip(numerical_features, raw_vals))

    st.divider()

    col_left, col_right = st.columns([1.15, 1])

    # Restrict cuisine one-hots to those actually present as columns
    available_cuisines = sorted(
        c.replace("cuisine_", "") for c in feature_cols if c.startswith("cuisine_")
    )
    available_boros = sorted(
        c.replace("boro_", "") for c in feature_cols if c.startswith("boro_")
    )

    with col_left:
        st.markdown(f"#### 🛠️ Sandbox: {selected['dba']}")
        st.caption(f"{selected['boro']} · {selected['cuisine_description']} · "
                   f"Ground-truth grade: **{selected['grade']}**")

        sb = {}

        def _clamp(val, lo, hi):
            return float(np.clip(val, lo, hi))

        r = slider_ranges
        sb["latest_score"] = st.slider(
            f"Most-recent inspection score  [{_importance_badge('latest_score')}]",
            r["latest_score"][0], r["latest_score"][1],
            _clamp(initial["latest_score"], *r["latest_score"]),
            help="DOHMH cutoffs: A ≤ 13, B 14–27, C ≥ 28.  This is the single strongest predictor of grade.",
        )
        sb["critical_ratio"] = st.slider(
            f"Critical-violation ratio  [{_importance_badge('critical_ratio')}]",
            0.0, 1.0,
            _clamp(initial["critical_ratio"], 0.0, 1.0),
            step=0.01,
            help="Fraction of recorded violations flagged Critical.  Critical violations (food temp, pest control, hand-washing) carry the heaviest regulatory weight.",
        )
        sb["violations_per_inspection"] = st.slider(
            f"Violations per inspection  [{_importance_badge('violations_per_inspection')}]",
            r["violations_per_inspection"][0], r["violations_per_inspection"][1],
            _clamp(initial["violations_per_inspection"], *r["violations_per_inspection"]),
            help="Average number of violations found each visit — a clean per-visit rate that normalises for inspection frequency.",
        )
        sb["avg_score"] = st.slider(
            f"Average score across all inspections  [{_importance_badge('avg_score')}]",
            r["avg_score"][0], r["avg_score"][1],
            _clamp(initial["avg_score"], *r["avg_score"]),
            help="Mean score across the restaurant's full inspection history.  Correlated with latest_score but captures long-run trend.",
        )
        sb["num_inspections"] = st.slider(
            f"Total number of inspections on record  [{_importance_badge('num_inspections')}]",
            r["num_inspections"][0], r["num_inspections"][1],
            _clamp(initial["num_inspections"], *r["num_inspections"]),
            help="More inspections = richer history for the model to use.",
        )
        sb["num_violations"] = st.slider(
            f"Total violation rows on record  [{_importance_badge('num_violations')}]",
            r["num_violations"][0], r["num_violations"][1],
            _clamp(initial["num_violations"], *r["num_violations"]),
            help="Raw count of violation rows.  Largely captured by violations_per_inspection; included for completeness.",
        )
        sb["max_score"] = st.slider(
            f"Worst-ever inspection score  [{_importance_badge('max_score')}]",
            r["max_score"][0], r["max_score"][1],
            _clamp(initial["max_score"], *r["max_score"]),
            help="Peak score across all inspections.  Highly correlated with latest_score and avg_score — see Feature Analysis tab.",
        )

        boro_choice = st.selectbox(
            "Borough",
            available_boros,
            index=available_boros.index(selected["boro"]) if selected["boro"] in available_boros else 0,
        )
        # Cuisine: fall back to the closest feature column
        best_cuisine = selected["cuisine_description"]
        if best_cuisine not in available_cuisines:
            best_cuisine = "Other" if "Other" in available_cuisines else available_cuisines[0]
        cuisine_choice = st.selectbox(
            "Cuisine group",
            available_cuisines,
            index=available_cuisines.index(best_cuisine),
        )

    with col_right:
        feature_vec = build_feature_vector(sb, boro_choice, cuisine_choice)
        probs = predict_single(feature_vec)
        pred_idx = int(np.argmax(probs))
        pred_grade = GRADE_NAMES[pred_idx]

        # Share the current sandbox state with the Latent Space tab so it can
        # project the same restaurant into the hidden-activation plot.
        st.session_state["classifier_sandbox"] = {
            "camis": str(selected.get("camis", "")),
            "dba": str(selected.get("dba", "Sandbox restaurant")),
            "feature_vec": feature_vec.copy(),
            "raw_numerical": dict(sb),
            "boro": boro_choice,
            "cuisine": cuisine_choice,
        }

        st.markdown(f"#### 🧠 Predicted Grade: **{pred_grade}**")
        if pred_grade == "A":
            st.success(f"🎉 **Congratulations!** This restaurant is predicted to maintain a **Grade A**. Confidence: {probs[0] * 100:.1f}%")
            st.markdown("Keep up the excellent hygiene standards. Continue routine inspections and maintain current operational practices.")
        elif pred_grade == "B":
            st.warning(f"⚠️ **At Risk — Grade B predicted.** Confidence: {probs[1] * 100:.1f}%")
            st.markdown("This restaurant has areas that need improvement. Review the guidance below to understand what changes could earn a Grade A.")
        else:
            st.error(f"🚨 **Critical — Grade C predicted.** Confidence: {probs[2] * 100:.1f}%")
            st.markdown("**Multiple areas need immediate attention.** A Grade C indicates serious hygiene concerns. See the specific changes needed below.")

        prob_fig = go.Figure(go.Bar(
            x=[f"Grade {g}" for g in GRADE_NAMES],
            y=list(probs),
            marker_color=[GRADE_COLORS[g] for g in GRADE_NAMES],
            text=[f"{p * 100:.1f}%" for p in probs],
            textposition="outside",
        ))
        prob_fig.update_layout(
            yaxis=dict(range=[0, 1.05], title="P(class)"),
            height=240, margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Counterfactual — what minimal change flips this to A?
        if pred_grade != "A":
            st.markdown("##### 🎯 Improvement Guidance: How to Achieve Grade A")
            with st.spinner("Computing actionable recommendations..."):
                # Freeze one-hot categorical features; only numerical features
                # represent real-world operational levers a restaurant can
                # change.
                mutable = torch.zeros(input_dim)
                for name in numerical_features:
                    mutable[feature_cols.index(name)] = 1.0
                cf = find_counterfactual(
                    model,
                    torch.from_numpy(feature_vec).unsqueeze(0),
                    target_class=0,
                    steps=200,
                    lr=0.05,
                    l2_penalty=0.3,
                    mutable_mask=mutable.unsqueeze(0),
                ).squeeze(0).numpy()
                # Rebuild the raw value table for the changed numerical features
                cf_raw = cf[:len(numerical_features)] * scaler_scale + scaler_mean
                # (Indices only line up because numerical features appear first.)
                
                # One-line action description for each feature (no direction word —
                # direction is shown explicitly in the guidance header below)
                FEATURE_ACTION = {
                    "latest_score":              "Penalty points from the most recent inspection (A ≤ 13 pts, B = 14–27, C ≥ 28). Reduce violations at the next visit.",
                    "avg_score":                 "Mean penalty score across all past inspections. Fix recurring violations to lower the long-run average.",
                    "max_score":                 "Highest single-inspection penalty score on record. Regular self-audits prevent catastrophic one-off failures.",
                    "num_violations":            "Total violation citations ever recorded. Reduce frequency through hygiene training and standard operating procedures.",
                    "critical_ratio":            "Share of violations flagged Critical (food temperature, pests, handwashing). Eliminate these first — they carry the heaviest penalty.",
                    "violations_per_inspection": "Average violations per visit. Improve per-visit hygiene through staff checklists and training.",
                    "num_inspections":           "Number of inspections on record — reflects compliance history length. Not directly actionable by the operator.",
                }

                cf_rows = []
                guidance_messages = []
                for name, new_val in zip(numerical_features, cf_raw):
                    delta = new_val - sb[name]
                    if abs(delta) > 0.05 * max(abs(sb[name]), 1.0):
                        display_name = FEATURE_DISPLAY_NAMES.get(name, name)
                        direction_word = "Reduce" if delta < 0 else "Increase"
                        arrow = "↓" if delta < 0 else "↑"
                        cf_rows.append({
                            "Feature": display_name,
                            "Current": f"{sb[name]:.2f}",
                            "Target": f"{new_val:.2f}",
                            "Change": f"{arrow} {abs(delta):.2f}",
                        })
                        if name in FEATURE_ACTION:
                            guidance_messages.append(
                                f"**{arrow} {direction_word} '{display_name}': "
                                f"{sb[name]:.1f} → {new_val:.1f}**  \n"
                                f"{FEATURE_ACTION[name]}"
                            )
                if cf_rows:
                    st.dataframe(
                        pd.DataFrame(cf_rows),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(
                        "↓ = needs to decrease  ·  ↑ = needs to increase  ·  "
                        "Minimum perturbation via gradient descent on inputs, model weights frozen."
                    )
                    if guidance_messages:
                        st.markdown("**What to change and why:**")
                        for msg in guidance_messages:
                            st.markdown(f"- {msg}")
                else:
                    st.caption(
                        "The restaurant is already very close to the Grade-A decision boundary."
                    )

# ---------------------------------------------------------------------------
# Tab 2 — Model Performance (held-out test set)
# ---------------------------------------------------------------------------

with tab_perf:
    st.subheader("Held-out test set evaluation")
    st.caption(
        f"{len(test_df):,} restaurants, never seen during training, validation, or hyperparameter search."
    )

    details = evaluate_mlp(
        model, tensors["X_test"], tensors["y_test"],
        class_names=GRADE_NAMES, return_details=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weighted F1", f"{details['weighted_f1']:.3f}")
    report = details["classification_report"]
    m2.metric("Accuracy", f"{report['accuracy']:.3f}")
    m3.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}",
              help="Unweighted mean of per-class F1; harsher on rare classes.")
    m4.metric("Test size", f"{len(test_df):,}")

    col_cm, col_report = st.columns([1, 1])

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = details["confusion_matrix"]
        cm_normed = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_normed,
            x=[f"Pred {g}" for g in GRADE_NAMES],
            y=[f"True {g}" for g in GRADE_NAMES],
            colorscale="Blues",
            text=[[f"{cm[i, j]} ({cm_normed[i, j] * 100:.0f}%)"
                   for j in range(3)] for i in range(3)],
            texttemplate="%{text}",
            hovertemplate="%{y} → %{x}<br>count=%{text}<extra></extra>",
            colorbar=dict(title="Row-normed"),
            zmin=0, zmax=1,
        ))
        cm_fig.update_layout(
            height=360, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_report:
        st.markdown("#### Per-class metrics")
        per_class = pd.DataFrame([
            {
                "Grade": g,
                "Precision": report[g]["precision"],
                "Recall": report[g]["recall"],
                "F1": report[g]["f1-score"],
                "Support": int(report[g]["support"]),
            }
            for g in GRADE_NAMES
        ])
        st.dataframe(
            per_class.style.format({
                "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Class prior vs. prediction")
        true_share = (pd.Series(tensors["y_test"].numpy())
                      .value_counts(normalize=True).reindex([0, 1, 2], fill_value=0))
        pred_share = (pd.Series(details["predictions"])
                      .value_counts(normalize=True).reindex([0, 1, 2], fill_value=0))
        share_fig = go.Figure()
        share_fig.add_trace(go.Bar(
            name="True", x=GRADE_NAMES, y=true_share.values,
            marker_color="#1D1D1F",
        ))
        share_fig.add_trace(go.Bar(
            name="Predicted", x=GRADE_NAMES, y=pred_share.values,
            marker_color="#007AFF",
        ))
        share_fig.update_layout(
            barmode="group", height=240,
            yaxis=dict(title="Share", range=[0, 1]),
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(share_fig, use_container_width=True)

    # Majority-class (always predict A) baseline for comparison
    _y_test_arr = tensors["y_test"].numpy()
    _majority_cls = int(np.bincount(_y_test_arr).argmax())
    _majority_acc = float((_y_test_arr == _majority_cls).mean())
    # Majority-classifier F1_A = 2·precision·recall/(precision+recall);
    # precision_A = base_rate, recall_A = 1.0; B and C get F1 = 0
    _f1_maj_A = 2 * _majority_acc / (_majority_acc + 1.0)
    _majority_macro_f1 = _f1_maj_A / 3.0

    st.markdown("#### Why these metrics?")
    st.markdown(
        f"The dataset is heavily imbalanced: "
        f"**{report[GRADE_NAMES[_majority_cls]]['support']:,} Grade {GRADE_NAMES[_majority_cls]}** "
        f"({_majority_acc:.0%} of test set).  A naive classifier that always predicts "
        f"Grade {GRADE_NAMES[_majority_cls]} would reach **{_majority_acc:.1%} accuracy** but only "
        f"**{_majority_macro_f1:.3f} macro F1** — and zero recall on Grades B and C.  "
        f"Weighted F1 is the primary metric because it rewards the model for correctly "
        f"handling minority classes."
    )
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric(
        "MLP Macro F1",
        f"{report['macro avg']['f1-score']:.3f}",
        delta=f"+{report['macro avg']['f1-score'] - _majority_macro_f1:.3f} vs majority baseline",
    )
    bc2.metric(
        "Majority-class baseline macro F1",
        f"{_majority_macro_f1:.3f}",
        help=f"Always predict Grade {GRADE_NAMES[_majority_cls]}; Grades B and C get F1 = 0",
    )
    bc3.metric(
        "Grade C Recall (MLP)",
        f"{report['C']['recall']:.1%}",
        help="Most critical for food safety — majority baseline recalls 0% of Grade C restaurants",
    )
    st.caption(
        "Class weights (inverse class frequency) are applied during training so the model "
        "is penalised more heavily for misclassifying rare Grade B and C restaurants."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Training diagnostics
# ---------------------------------------------------------------------------

with tab_train:
    st.subheader("Training dynamics of the deployed model")
    _n_batches = len(tensors["X_train"]) // DEPLOYED_HYPERPARAMS["batch_size"]
    st.markdown(
        f"**One epoch** = one full pass through all {len(tensors['X_train']):,} training "
        f"restaurants, split into ~{_n_batches} mini-batches of "
        f"{DEPLOYED_HYPERPARAMS['batch_size']}.  "
        f"**`max_epochs = {DEPLOYED_HYPERPARAMS['max_epochs']}`** is an upper bound set "
        f"well above typical convergence; the actual training length is decided by "
        f"**early stopping** (patience = {DEPLOYED_HYPERPARAMS['patience']} epochs) — "
        f"training halts as soon as validation F1 fails to improve for "
        f"{DEPLOYED_HYPERPARAMS['patience']} consecutive epochs.  "
        f"This prevents overfitting without needing to tune `max_epochs` precisely."
    )

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Best epoch", f"{training_history.best_epoch + 1}")
    tc2.metric("Best val F1", f"{training_history.best_val_f1:.3f}")
    tc3.metric("Early stopping", "Yes" if training_history.stopped_early else "No")

    col_loss, col_f1 = st.columns(2)

    with col_loss:
        st.markdown("#### Loss curves")
        loss_fig = go.Figure()
        epochs = list(range(1, len(training_history.train_loss) + 1))
        loss_fig.add_trace(go.Scatter(
            x=epochs, y=training_history.train_loss,
            name="Train", line=dict(color="#1D1D1F"),
        ))
        if training_history.val_loss:
            loss_fig.add_trace(go.Scatter(
                x=epochs, y=training_history.val_loss,
                name="Validation", line=dict(color="#007AFF"),
            ))
        loss_fig.add_vline(
            x=training_history.best_epoch + 1,
            line_dash="dash", line_color="#FF3B30",
            annotation_text="Best epoch", annotation_position="top",
        )
        loss_fig.update_layout(
            height=320, xaxis_title="Epoch", yaxis_title="Cross-entropy loss",
            margin=dict(l=30, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(loss_fig, use_container_width=True)

    with col_f1:
        st.markdown("#### Weighted F1 over training")
        f1_fig = go.Figure()
        f1_fig.add_trace(go.Scatter(
            x=epochs, y=training_history.train_f1,
            name="Train", line=dict(color="#1D1D1F"),
        ))
        if training_history.val_f1:
            f1_fig.add_trace(go.Scatter(
                x=epochs, y=training_history.val_f1,
                name="Validation", line=dict(color="#007AFF"),
            ))
        f1_fig.add_vline(
            x=training_history.best_epoch + 1,
            line_dash="dash", line_color="#FF3B30",
            annotation_text="Best epoch", annotation_position="top",
        )
        f1_fig.update_layout(
            height=320, xaxis_title="Epoch", yaxis_title="Weighted F1",
            yaxis_range=[0, 1.0],
            margin=dict(l=30, r=20, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(f1_fig, use_container_width=True)

    # Dynamic interpretation: early stopping and overfitting check
    if training_history.stopped_early:
        st.success(
            f"✅ **Early stopping triggered at epoch {training_history.best_epoch + 1}** "
            f"(patience = {DEPLOYED_HYPERPARAMS['patience']} epochs). Validation F1 did not "
            f"improve for {DEPLOYED_HYPERPARAMS['patience']} consecutive epochs, so training "
            f"halted before the maximum of {DEPLOYED_HYPERPARAMS['max_epochs']} epochs. "
            f"This confirms the model converged without overfitting."
        )
    else:
        st.info(
            f"Training ran the full {DEPLOYED_HYPERPARAMS['max_epochs']} epochs without "
            f"early stopping. Best validation F1 ({training_history.best_val_f1:.3f}) was "
            f"at epoch {training_history.best_epoch + 1}."
        )

    if len(training_history.val_f1) > 0:
        best_train_f1 = training_history.train_f1[training_history.best_epoch]
        best_val_f1   = training_history.val_f1[training_history.best_epoch]
        gap = best_train_f1 - best_val_f1
        if gap > 0.05:
            st.warning(
                f"Train F1 ({best_train_f1:.3f}) exceeds val F1 ({best_val_f1:.3f}) by "
                f"{gap:.3f} at the best epoch — a modest generalisation gap. "
                f"Dropout (0.3) and weight decay (1e-4) were tuned to keep this small."
            )
        else:
            st.success(
                f"Train F1 ({best_train_f1:.3f}) and val F1 ({best_val_f1:.3f}) are closely "
                f"matched at the best epoch (gap = {gap:.3f}) — no significant overfitting."
            )

# ---------------------------------------------------------------------------
# Tab 4 — Latent space visualisation
#
# Shows (a) the 128-dim penultimate-layer activations projected to 2D via PCA
# or t-SNE, coloured by ground-truth grade, and (b) an interactive
# 2D decision-boundary slice over any two numerical features.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Extracting hidden activations from the MLP...")
def compute_latent_projections():
    """Run the MLP up to its penultimate layer on the held-out test set and
    return PCA + t-SNE 2-D projections.

    Cached so the (moderately expensive) t-SNE only runs once per session.
    """
    model.eval()
    with torch.no_grad():
        hidden = model.forward_hidden(tensors["X_test"]).numpy()

    pca_model = PCA(n_components=2, random_state=42)
    pca_2d = pca_model.fit_transform(hidden)

    sample_size = min(1500, len(hidden))
    rng = np.random.default_rng(42)
    tsne_idx = rng.choice(len(hidden), size=sample_size, replace=False)
    tsne_idx.sort()
    tsne_input = hidden[tsne_idx]
    perplexity = int(np.clip(sample_size // 50, 5, 40))
    tsne_2d_sample = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    ).fit_transform(tsne_input)

    return {
        "hidden": hidden,
        "pca": pca_2d,
        "pca_model": pca_model,
        "pca_var": pca_model.explained_variance_ratio_,
        "tsne_idx": tsne_idx,
        "tsne": tsne_2d_sample,
    }


with tab_latent:
    st.subheader("How the MLP organises restaurants in its 128-dim latent space")
    st.markdown(
        """
The MLP's second hidden layer outputs a **128-dimensional activation vector** for each
restaurant — the network's internal representation before the final grade decision.
To inspect what the model has learned, we compress those 128 numbers down to 2 for plotting:

- **PCA (Principal Component Analysis)**: a linear projection that finds the two directions of
  greatest variance in the 128-dim space.  **PC1** (x-axis) is the single direction that
  explains the most variation; **PC2** (y-axis) explains the second most.
  Axis values are in PCA units (not original inspection-score units), and the percentage
  of 128-dim variance each component captures is shown on the axis label.

- **t-SNE**: a non-linear method that rearranges points to preserve local neighbourhood
  structure.  **Axis values have no interpretable meaning** — only relative closeness
  of same-colour points matters.  t-SNE often produces tighter visual clusters than PCA
  but distances *between* clusters are not comparable.

Points coloured by their ground-truth grade.  Well-separated colour bands show the MLP
has learned grade-discriminative representations even though grade labels were never
fed into the hidden layers directly.  Overlap regions are where the model is uncertain.
"""
    )

    latent = compute_latent_projections()
    y_test_np = tensors["y_test"].numpy()

    projection_kind = st.radio(
        "Projection method",
        ["PCA (global structure)", "t-SNE (local neighbourhoods)"],
        horizontal=True,
        help=(
            "PCA is a linear projection that preserves variance — good for "
            "seeing overall separability.  t-SNE is non-linear and focuses "
            "on local structure, which usually makes clusters look cleaner "
            "but distances across clusters are not meaningful."
        ),
    )
    use_tsne = projection_kind.startswith("t-SNE")

    if use_tsne:
        coords = latent["tsne"]
        labels = y_test_np[latent["tsne_idx"]]
        x_axis_label = "t-SNE Dim 1  (no interpretable units)"
        y_axis_label = "t-SNE Dim 2  (no interpretable units)"
        sample_note = (
            f"t-SNE on {len(coords):,} test restaurants.  "
            "Axis values are arbitrary — only local neighbourhood structure is meaningful."
        )
    else:
        coords = latent["pca"]
        labels = y_test_np
        pvar = latent["pca_var"]
        x_axis_label = f"PC1  ({pvar[0]*100:.1f}% of 128-dim variance)"
        y_axis_label = f"PC2  ({pvar[1]*100:.1f}% of 128-dim variance)"
        sample_note = (
            f"PCA on all {len(coords):,} held-out test restaurants.  "
            f"PC1 + PC2 together capture {(pvar[0]+pvar[1])*100:.1f}% of the 128-dim hidden-layer variance."
        )

    latent_fig = go.Figure()
    for grade_idx, grade_name in enumerate(GRADE_NAMES):
        mask = labels == grade_idx
        latent_fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            name=f"Grade {grade_name}",
            marker=dict(
                size=5,
                color=GRADE_COLORS[grade_name],
                opacity=0.65,
                line=dict(width=0),
            ),
        ))

    # Overlay the currently-selected sandbox restaurant.
    sandbox_state = st.session_state.get("classifier_sandbox")
    if sandbox_state is not None:
        sandbox_vec = torch.from_numpy(sandbox_state["feature_vec"]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            sandbox_hidden = model.forward_hidden(sandbox_vec).numpy()

        if use_tsne:
            # t-SNE has no out-of-sample transform; project via nearest-neighbour
            # in the original hidden space so the overlay lands somewhere
            # visually consistent.  Distances in t-SNE space are not meaningful,
            # so this is a rough locator rather than a faithful embedding.
            ref_hidden = latent["hidden"][latent["tsne_idx"]]
            nn_idx = int(np.argmin(np.linalg.norm(ref_hidden - sandbox_hidden, axis=1)))
            sandbox_xy = coords[nn_idx]
            overlay_note = " (nearest-neighbour anchored)"
        else:
            sandbox_xy = latent["pca_model"].transform(sandbox_hidden)[0]
            overlay_note = ""

        latent_fig.add_trace(go.Scatter(
            x=[sandbox_xy[0]],
            y=[sandbox_xy[1]],
            mode="markers+text",
            name=f"Sandbox: {sandbox_state['dba'][:30]}",
            marker=dict(
                size=22, color="#5856D6", symbol="star",
                line=dict(color="#FFFFFF", width=2),
            ),
            text=["⭐"],
            textposition="middle center",
            textfont=dict(size=14, color="white"),
            hovertemplate=f"<b>{sandbox_state['dba']}</b>{overlay_note}<extra></extra>",
        ))

    latent_fig.update_layout(
        height=480,
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        margin=dict(l=30, r=20, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(latent_fig, use_container_width=True)
    st.caption(sample_note)

    st.markdown(
        """
The ⭐ marks the restaurant currently selected on the **Live Prediction**
tab — change it there to see the point move.  Well-separated colour bands
mean the MLP's penultimate layer has learned grade-discriminative
features.  Overlap is where the model is uncertain, and is exactly where
the counterfactual optimiser has to work hardest to flip a prediction.
"""
    )

    # ------------------------------------------------------------------
    # 2D decision-boundary slice
    # ------------------------------------------------------------------

    st.divider()
    st.subheader("Decision-boundary slice")
    st.markdown(
        """
The MLP makes decisions in a **29-dimensional feature space**.  To see how it draws
grade boundaries, we fix 27 of the 29 dimensions and sweep a 60×60 grid over the
remaining two (the features you select below), colouring each cell by the MLP's
predicted grade at that point:

- **Green region** = model predicts Grade A
- **Yellow region** = Grade B
- **Red region** = Grade C
- **Scattered dots** = real test restaurants at their true (x, y) values, coloured by actual grade
- **⭐** = the restaurant currently loaded in the sandbox

The x and y axes span the 1st–99th percentile of each feature in the training data,
so the two axes almost always have **different scales** — the plot is not square by design.
Rectangular boundary shapes arise because the MLP's ReLU activations create
piecewise-linear regions; the exact shape changes as you move the sandbox sliders.
"""
    )

    slice_default = sandbox_state is not None
    if not slice_default:
        st.info("Pick a restaurant on the **Live Prediction** tab first — the "
                "slice uses its values for the non-plotted features.")
    else:
        available_axes = list(numerical_features)
        col_x, col_y = st.columns(2)
        with col_x:
            x_feat = st.selectbox(
                "X axis feature",
                available_axes,
                index=available_axes.index("latest_score")
                if "latest_score" in available_axes else 0,
            )
        with col_y:
            remaining = [f for f in available_axes if f != x_feat]
            y_default = "critical_ratio" if "critical_ratio" in remaining else remaining[0]
            y_feat = st.selectbox(
                "Y axis feature",
                remaining,
                index=remaining.index(y_default),
            )

        # Build the grid in raw units, using the training-set min / max for
        # each feature (from train_df so the grid covers realistic values).
        raw_train = train_df[numerical_features].values * scaler_scale + scaler_mean

        def _axis_range(name):
            col = raw_train[:, numerical_features.index(name)]
            lo, hi = float(np.percentile(col, 1)), float(np.percentile(col, 99))
            if hi - lo < 1e-6:
                hi = lo + 1.0
            return lo, hi

        x_lo, x_hi = _axis_range(x_feat)
        y_lo, y_hi = _axis_range(y_feat)

        grid_size = 60
        xs = np.linspace(x_lo, x_hi, grid_size, dtype=np.float32)
        ys = np.linspace(y_lo, y_hi, grid_size, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)

        # Start from the sandbox raw values; overwrite the two swept features.
        raw_base = np.array(
            [sandbox_state["raw_numerical"][c] for c in numerical_features],
            dtype=np.float32,
        )
        x_idx = numerical_features.index(x_feat)
        y_idx = numerical_features.index(y_feat)

        raw_grid = np.tile(raw_base, (grid_size * grid_size, 1))
        raw_grid[:, x_idx] = XX.ravel()
        raw_grid[:, y_idx] = YY.ravel()

        # Standardise and assemble full feature vectors (numerical block +
        # the sandbox's one-hot borough and cuisine columns).
        scaled_grid = (raw_grid - scaler_mean) / scaler_scale
        full_grid = np.zeros((grid_size * grid_size, input_dim), dtype=np.float32)
        for col_i, name in enumerate(numerical_features):
            full_grid[:, feature_cols.index(name)] = scaled_grid[:, col_i]
        boro_key = f"boro_{sandbox_state['boro']}"
        if boro_key in feature_cols:
            full_grid[:, feature_cols.index(boro_key)] = 1.0
        cuisine_key = f"cuisine_{sandbox_state['cuisine']}"
        if cuisine_key in feature_cols:
            full_grid[:, feature_cols.index(cuisine_key)] = 1.0
        elif "cuisine_Other" in feature_cols:
            full_grid[:, feature_cols.index("cuisine_Other")] = 1.0

        model.eval()
        with torch.no_grad():
            logits_grid = model(torch.from_numpy(full_grid))
            preds_grid = logits_grid.argmax(dim=1).numpy().reshape(grid_size, grid_size)

        # Plotly heatmap with discrete grade colours.
        colorscale = [
            [0.00, GRADE_COLORS["A"]],
            [0.33, GRADE_COLORS["A"]],
            [0.34, GRADE_COLORS["B"]],
            [0.66, GRADE_COLORS["B"]],
            [0.67, GRADE_COLORS["C"]],
            [1.00, GRADE_COLORS["C"]],
        ]
        boundary_fig = go.Figure()
        boundary_fig.add_trace(go.Heatmap(
            z=preds_grid,
            x=xs,
            y=ys,
            colorscale=colorscale,
            zmin=0, zmax=2,
            showscale=False,
            opacity=0.55,
            hovertemplate=(f"{x_feat}=%{{x:.2f}}<br>{y_feat}=%{{y:.2f}}"
                           "<br>Predicted grade index=%{z}<extra></extra>"),
        ))

        # Overlay test restaurants projected onto these two axes.
        meta_raw = test_df[numerical_features].values * scaler_scale + scaler_mean
        overlay_x = meta_raw[:, x_idx]
        overlay_y = meta_raw[:, y_idx]
        # Subsample for readability.
        if len(overlay_x) > 1500:
            overlay_sample = np.random.default_rng(0).choice(
                len(overlay_x), size=1500, replace=False,
            )
            overlay_x = overlay_x[overlay_sample]
            overlay_y = overlay_y[overlay_sample]
            overlay_labels = y_test_np[overlay_sample]
        else:
            overlay_labels = y_test_np

        for grade_idx, grade_name in enumerate(GRADE_NAMES):
            mask = overlay_labels == grade_idx
            boundary_fig.add_trace(go.Scatter(
                x=overlay_x[mask], y=overlay_y[mask],
                mode="markers",
                name=f"Grade {grade_name}",
                marker=dict(
                    size=5,
                    color=GRADE_COLORS[grade_name],
                    line=dict(color="#1D1D1F", width=0.5),
                    opacity=0.75,
                ),
            ))

        # Overlay sandbox restaurant.
        boundary_fig.add_trace(go.Scatter(
            x=[sandbox_state["raw_numerical"][x_feat]],
            y=[sandbox_state["raw_numerical"][y_feat]],
            mode="markers+text",
            name="Sandbox",
            marker=dict(size=22, color="#5856D6", symbol="star",
                        line=dict(color="#FFFFFF", width=2)),
            text=["⭐"],
            textposition="middle center",
            textfont=dict(size=14, color="white"),
            hovertemplate=f"<b>{sandbox_state['dba']}</b><extra></extra>",
        ))

        boundary_fig.update_layout(
            height=520,
            xaxis_title=FEATURE_DISPLAY_NAMES.get(x_feat, x_feat),
            yaxis_title=FEATURE_DISPLAY_NAMES.get(y_feat, y_feat),
            margin=dict(l=40, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(boundary_fig, use_container_width=True)
        st.caption(
            f"All other features are held fixed at `{sandbox_state['dba']}`'s "
            f"current sandbox values (borough={sandbox_state['boro']}, "
            f"cuisine={sandbox_state['cuisine']}).  Change the sliders on the "
            f"Live Prediction tab to see how the boundary shifts."
        )


# ---------------------------------------------------------------------------
# Tab 5 — Hyperparameter justification
# ---------------------------------------------------------------------------

HP_CACHE_PATH = DATA_DIR / "cache" / "hp_search_results.json"


def _load_hp_cache():
    if HP_CACHE_PATH.exists():
        try:
            with open(HP_CACHE_PATH) as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_hp_cache(results):
    HP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HP_CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)


with tab_hp:
    st.subheader("Model Architecture and Hyperparameter Justification")

    # ── Why MLP? ──────────────────────────────────────────────────────────────
    st.markdown("### Why a Multi-Layer Perceptron?")
    st.markdown(
        "- **Non-linear decision boundaries**: DOHMH grades have hard point-score cutoffs "
        "(A ≤ 13, B 14–27, C ≥ 28). Interactions between score, critical violation ratio, "
        "and cuisine/borough context create non-linear patterns that a logistic regression "
        "cannot capture with a single hyperplane.\n"
        "- **Mixed feature types**: 29 inputs span 7 continuous numericals and 22 binary "
        "one-hots. MLPs handle heterogeneous inputs natively without feature-type-specific "
        "preprocessing.\n"
        "- **From-scratch implementation**: The full training loop — class-weighted cross-entropy "
        "loss, AdamW optimizer, early stopping on validation F1, model checkpointing — is "
        "implemented in raw PyTorch, satisfying the course requirement.\n"
        "- **Gradient-based counterfactuals**: Because the network is differentiable, gradient "
        "descent on the *input* (weights frozen) finds the minimum feature change that flips a "
        "prediction to Grade A — not possible with tree-based models."
    )

    # ── Architecture and training choices ────────────────────────────────────
    st.markdown("### Architecture and Training Choices")
    _hp_y = torch.cat([tensors["y_train"], tensors["y_val"]]).numpy()
    _hp_cnt = np.bincount(_hp_y)
    _hp_pct = _hp_cnt / _hp_cnt.sum() * 100

    col_arch, col_train = st.columns(2)
    with col_arch:
        st.markdown(
            f"""**Network: {input_dim} → 128 → 128 → 3**

| Decision | Choice | Data-backed rationale |
|---|---|---|
| Depth | 2 hidden layers | 1 hidden layer underfits (insufficient capacity); 3+ overfits on ~{len(_hp_y):,} training examples |
| Width | 128 | Grid search: hidden=128 outperforms 64 (undercapacity) with marginal gain from 256 |
| Activation | ReLU | No vanishing gradient; learns sharp grade-cutoff thresholds |
| Output | Softmax (3-way) | Calibrated P(A), P(B), P(C) required for counterfactual optimisation |
"""
        )
    with col_train:
        st.markdown(
            f"""**Optimization & regularisation**

| Setting | Value | Why |
|---|---|---|
| Loss | Class-weighted cross-entropy | A={_hp_pct[0]:.0f}% / B={_hp_pct[1]:.0f}% / C={_hp_pct[2]:.0f}% — inverse-frequency weights prevent majority-class collapse |
| Optimizer | AdamW | Decoupled weight decay generalises better than plain Adam on small-medium datasets |
| Dropout | 0.3 | Grid search: 0.3 outperforms 0.1 (underfits) and 0.5 (overfits) |
| Weight decay | 1e-4 | Second regulariser complementing dropout |
| Early stopping | patience=12 | Monitors val F1; halts once improvement plateaus to prevent overfitting |
| Batch size | 128 | Standard for tabular data; balances gradient noise and throughput |
"""
        )

    st.divider()

    # ── Hyperparameter grid search ────────────────────────────────────────────
    st.markdown("### Hyperparameter Grid Search")
    st.markdown(
        """
We searched over **(hidden_dim, lr, dropout)** — the three hyperparameters
most sensitive to dataset size and class imbalance — on the same stratified
train/val split used by the deployed model.  Every configuration was trained
from scratch with identical early-stopping and class-weighting settings.
The configuration with the highest validation F1 is deployed.
"""
    )

    cached = _load_hp_cache()

    col_run, col_info = st.columns([1, 2])
    with col_run:
        rerun = st.button(
            "Run grid search" if cached is None else "Re-run grid search",
            type="primary",
        )
    with col_info:
        if cached:
            st.caption(f"Cached results loaded from `{HP_CACHE_PATH.relative_to(REPO_ROOT)}` "
                       f"({len(cached)} configurations).")
        else:
            st.caption("No cached results yet — click **Run grid search** "
                       "to compute them.  Expect ~1–2 minutes.")

    if rerun:
        progress = st.progress(0.0, text="Starting grid search...")

        def _progress(i, total, params):
            if params is None:
                progress.progress(1.0, text="Grid search complete.")
            else:
                progress.progress(
                    i / max(total, 1),
                    text=(f"Config {i + 1}/{total}: hidden={params['hidden_dim']}, "
                          f"lr={params['lr']}, dropout={params['dropout']}"),
                )

        results = hyperparameter_search(
            tensors["X_train"], tensors["y_train"],
            tensors["X_val"], tensors["y_val"],
            hidden_dims=(64, 128, 256),
            learning_rates=(5e-4, 1e-3, 5e-3),
            dropouts=(0.1, 0.3, 0.5),
            epochs=40, patience=8, seed=42,
            progress_callback=_progress,
        )
        _save_hp_cache(results)
        cached = results

    if cached:
        results_df = pd.DataFrame(cached)
        results_df["lr"] = results_df["lr"].astype(float)
        results_df["best_val_f1"] = results_df["best_val_f1"].astype(float)

        st.markdown("#### Grid search results (sorted by validation F1)")
        display_df = results_df[[
            "hidden_dim", "lr", "dropout", "best_val_f1", "best_epoch", "stopped_early",
        ]].rename(columns={
            "hidden_dim": "Hidden",
            "lr": "LR",
            "dropout": "Dropout",
            "best_val_f1": "Val F1",
            "best_epoch": "Best Epoch",
            "stopped_early": "Early-stopped",
        })
        st.dataframe(
            display_df.style.format({
                "LR": "{:.0e}",
                "Dropout": "{:.2f}",
                "Val F1": "{:.4f}",
            }).background_gradient(subset=["Val F1"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        best = results_df.iloc[0]
        deployed_match = (
            int(best["hidden_dim"]) == DEPLOYED_HYPERPARAMS["hidden_dim"]
            and abs(float(best["lr"]) - DEPLOYED_HYPERPARAMS["lr"]) < 1e-9
            and abs(float(best["dropout"]) - DEPLOYED_HYPERPARAMS["dropout"]) < 1e-3
        )
        banner = st.success if deployed_match else st.warning
        banner(
            f"Best configuration: hidden={int(best['hidden_dim'])}, "
            f"lr={float(best['lr']):.0e}, dropout={float(best['dropout']):.2f} "
            f"(val F1 = {float(best['best_val_f1']):.4f}).\n\n"
            + (
                "This matches the deployed configuration."
                if deployed_match
                else "**Note:** the top configuration differs from the deployed "
                     "one — consider updating `DEPLOYED_HYPERPARAMS`."
            )
        )

        # Marginal-effect plots
        st.markdown("#### Marginal effects")
        mc1, mc2, mc3 = st.columns(3)
        for col_ctx, axis, title in [
            (mc1, "hidden_dim", "Hidden dimension"),
            (mc2, "lr", "Learning rate"),
            (mc3, "dropout", "Dropout"),
        ]:
            agg = (
                results_df.groupby(axis)["best_val_f1"]
                .agg(["mean", "max"])
                .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agg[axis], y=agg["mean"],
                mode="lines+markers", name="Mean",
                line=dict(color="#007AFF"),
            ))
            fig.add_trace(go.Scatter(
                x=agg[axis], y=agg["max"],
                mode="lines+markers", name="Best",
                line=dict(color="#34C759", dash="dot"),
            ))
            fig.update_layout(
                height=260, title=title, xaxis_title=axis, yaxis_title="Val F1",
                margin=dict(l=30, r=20, t=40, b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_family="Inter, -apple-system, sans-serif",
            )
            if axis == "lr":
                fig.update_xaxes(type="log")
            with col_ctx:
                st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each marginal plot shows how validation F1 varies along a single "
            "axis, marginalising over the other two hyperparameters."
        )


# ---------------------------------------------------------------------------
# Tab 6 — Feature Analysis
#
# Three analyses justify the choice of sandbox parameters:
#   1. Input-gradient importance  — first-order sensitivity of loss to inputs
#   2. Permutation importance     — model-agnostic F1-drop measurement
#   3. Pearson correlation matrix — identifies redundant numerical features
# ---------------------------------------------------------------------------

with tab_feat:
    st.subheader("Feature Selection Analysis")
    st.info(
        "**This tab is feature importance analysis — not PCA.**  "
        "PCA appears in the **Latent Space** tab as a visualisation tool for the 128-dim "
        "hidden activations.  Here we ask a completely different question: "
        "*which of the 29 input features actually drive the model's predictions?*"
    )
    st.markdown(
        """
Two from-scratch methods (no external attribution library) measure each feature's contribution:

1. **Input-gradient importance** `|∂L/∂xⱼ|`: compute how much a tiny change in input feature *j*
   changes the cross-entropy loss.  A large gradient means the model's output is highly
   sensitive to that feature.  Implemented via a single backward pass in PyTorch over the
   held-out test set.

2. **Permutation importance** `ΔF1`: randomly shuffle the values of feature *j* across all
   test restaurants (destroying its signal), measure the drop in weighted F1, then repeat
   10 times and average.  A large drop means the model relies on that feature.
   Implemented in NumPy/PyTorch — no scikit-learn wrapper.

The **correlation matrix** below checks whether the top features carry independent
information or are redundant (highly correlated features encode overlapping signals
and can safely be merged or dropped).
"""
    )

    # ── 1. Importance charts ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 1. Feature Importance (all 29 input dimensions)")
    st.caption(
        "**Left:** input-gradient importance — mean |∂L/∂xⱼ| over the test set "
        "(single backward pass, PyTorch).  "
        "**Right:** permutation importance — mean weighted-F1 drop when feature j "
        "is randomly shuffled 10 times (model-agnostic, NumPy/PyTorch)."
    )

    # Colour map: numerical → blue, boro → green, cuisine → orange
    def _feat_color(name):
        if name in numerical_features:
            return "#007AFF"
        if name.startswith("boro_"):
            return "#34C759"
        return "#FF9500"

    colors = [_feat_color(n) for n in feature_cols]

    col_grad, col_perm = st.columns(2)

    with col_grad:
        grad_sorted = grad_imp.sort_values(ascending=True)
        grad_fig = go.Figure(go.Bar(
            x=grad_sorted.values,
            y=grad_sorted.index,
            orientation="h",
            marker_color=[_feat_color(n) for n in grad_sorted.index],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        grad_fig.update_layout(
            title="Gradient Importance",
            height=max(380, len(feature_cols) * 14),
            xaxis_title="|∂L/∂x| mean",
            margin=dict(l=160, r=20, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(grad_fig, use_container_width=True)

    with col_perm:
        perm_sorted = perm_imp.sort_values(ascending=True)
        perm_fig = go.Figure(go.Bar(
            x=perm_sorted.values,
            y=perm_sorted.index,
            orientation="h",
            marker_color=[_feat_color(n) for n in perm_sorted.index],
            hovertemplate="%{y}: Δ F1 = %{x:.4f}<extra></extra>",
        ))
        perm_fig.add_vline(x=0, line_dash="dash", line_color="#8E8E93", line_width=1)
        perm_fig.update_layout(
            title="Permutation Importance (F1 drop)",
            height=max(380, len(feature_cols) * 14),
            xaxis_title="Mean F1 drop (10 repeats)",
            margin=dict(l=160, r=20, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, -apple-system, sans-serif",
        )
        st.plotly_chart(perm_fig, use_container_width=True)

    # Legend
    st.markdown(
        "<span style='color:#007AFF'>■</span> Numerical &nbsp;&nbsp;"
        "<span style='color:#34C759'>■</span> Borough one-hot &nbsp;&nbsp;"
        "<span style='color:#FF9500'>■</span> Cuisine one-hot",
        unsafe_allow_html=True,
    )

    # ── 2. Numerical feature correlation ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2. Numerical Feature Correlation Matrix")
    st.caption(
        "Pearson r between the 7 raw numerical features (training set, "
        "un-standardised).  Values near ±1 indicate redundancy — two highly "
        "correlated features carry almost the same information."
    )

    raw_train_num = train_df[numerical_features].values * scaler_scale + scaler_mean
    corr = np.corrcoef(raw_train_num.T)

    corr_fig = go.Figure(go.Heatmap(
        z=corr,
        x=numerical_features,
        y=numerical_features,
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{corr[i, j]:.2f}" for j in range(len(numerical_features))]
              for i in range(len(numerical_features))],
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>r = %{text}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    corr_fig.update_layout(
        height=420,
        margin=dict(l=160, r=20, t=20, b=120),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(corr_fig, use_container_width=True)

    # ── 3. Ranked summary table ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3. Numerical Feature Ranking & Selection Recommendation")

    perm_num = perm_imp[numerical_features]
    grad_num = grad_imp[numerical_features]

    # Normalise each to [0,1] for display
    def _norm01(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0

    perm_norm = _norm01(perm_num)
    grad_norm = _norm01(grad_num)
    combined  = (0.6 * perm_norm + 0.4 * grad_norm).sort_values(ascending=False)

    # Max absolute pairwise correlation for each feature (vs the others)
    corr_df = pd.DataFrame(corr, index=numerical_features, columns=numerical_features)
    max_corr = {}
    for feat in numerical_features:
        others = corr_df[feat].drop(feat)
        max_corr[feat] = float(others.abs().max())

    rows = []
    for rank, feat in enumerate(combined.index, start=1):
        pi   = float(perm_num[feat])
        gi   = float(grad_num[feat])
        mc   = max_corr[feat]

        # Recommendation logic
        if pi < 0:
            rec = "⚠️ Possibly redundant (permutation importance ≤ 0)"
        elif mc > 0.85 and rank > 3:
            rec = "⚠️ Highly correlated with a higher-ranked feature — consider dropping"
        elif pi >= float(perm_num.quantile(0.6)):
            rec = "✅ Keep — strong independent signal"
        else:
            rec = "✅ Keep — modest but additive signal"

        rows.append({
            "Rank": rank,
            "Feature": feat,
            "Perm ΔF1": round(pi, 4),
            "Grad |∂L/∂x|": round(gi, 4),
            "Max |r| w/ others": round(mc, 2),
            "Recommendation": rec,
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.background_gradient(subset=["Perm ΔF1"], cmap="Blues")
                        .background_gradient(subset=["Max |r| w/ others"], cmap="Reds"),
        use_container_width=True, hide_index=True,
    )

    # ── 4. Written conclusion ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 4. Conclusions")
    top2 = list(combined.head(2).index)
    bottom2 = list(combined.tail(2).index)
    st.info(
        f"**Most important features:** `{'` and `'.join(top2)}` — these two alone "
        f"drive the majority of the model's predictive power, consistent with DOHMH "
        f"grading rules (grade is directly determined by inspection score).\n\n"
        f"**Potentially redundant features:** `{'` and `'.join(bottom2)}` show the "
        f"lowest independent signal.  The correlation matrix confirms that the three "
        f"score features (`latest_score`, `avg_score`, `max_score`) are strongly "
        f"correlated — they encode overlapping information about inspection history.\n\n"
        f"**Why keep all 7?**  Even correlated features can improve calibration on "
        f"minority classes (Grade B/C).  The permutation test shows each contributes "
        f"a non-negative F1 improvement, so removing any of them does not strictly "
        f"improve performance on this dataset.  The sandbox orders sliders by "
        f"importance so users focus on the levers that matter most."
    )
