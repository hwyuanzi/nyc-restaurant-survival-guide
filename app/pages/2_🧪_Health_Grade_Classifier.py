"""
Page 2 — Simple Health Grade Classifier

Pick a restaurant from the DOHMH held-out test set, hit the button, get the
predicted grade (A / B / C) with class probabilities.  That is the whole
story of this page.

A small "Model Performance" section below shows the held-out test metrics
so the audience can see the classifier is a real ML model, not a rule.

This page replaces the earlier multi-tab sandbox (sliders / counterfactuals /
latent space / hyperparameter grid).  Those were helpful for debugging but
made the deliverable hard to read.

Data:   data/train.csv + data/test.csv  (29-feature vectors)
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
from models.custom_mlp import (
    CustomMLP,
    TrainingHistory,
    compute_gradient_importance,
    compute_permutation_importance,
    evaluate_mlp,
    find_counterfactual,
    train_mlp,
)
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
    "DOHMH letter grade (A / B / C).  The model was trained on real NYC Department "
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

GRADE_NAMES = ["A", "B", "C"]
GRADE_COLORS = {"A": "#34C759", "B": "#FFCC00", "C": "#FF3B30"}

FEATURE_DISPLAY_NAMES = {
    "latest_score": "Most-recent inspection score",
    "avg_score": "Average score across all inspections",
    "max_score": "Worst-ever inspection score",
    "num_inspections": "Total number of inspections on record",
    "num_violations": "Total violation rows on record",
    "critical_ratio": "Critical-violation ratio",
    "violations_per_inspection": "Violations per inspection",
}


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
numerical_features = feature_config["numerical_features"]
scaler_mean = np.array(feature_config["scaler_mean"], dtype=np.float32)
scaler_scale = np.array(feature_config["scaler_scale"], dtype=np.float32)
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
# Data-driven slider bounds
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_slider_ranges():
    """Return {feature_name: (lo, hi)} using training-set percentiles in raw units."""
    raw = train_df[numerical_features].values * scaler_scale + scaler_mean
    lo = np.maximum(0.0, np.percentile(raw, 1, axis=0))
    hi = np.percentile(raw, 99, axis=0)
    ranges = {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(numerical_features)}
    ranges["critical_ratio"] = (0.0, 1.0)
    return ranges


slider_ranges = compute_slider_ranges()


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


# ---------------------------------------------------------------------------
# Feature importance + sandbox feature-vector helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Computing feature importance on held-out test set...")
def get_feature_importance():
    """Return (grad_imp, perm_imp) as pd.Series indexed by feature_cols."""
    grad = compute_gradient_importance(
        model,
        tensors["X_test"],
        tensors["y_test"],
        feature_names=feature_cols,
    )
    perm = compute_permutation_importance(
        model,
        tensors["X_test"],
        tensors["y_test"],
        feature_names=feature_cols,
        n_repeats=10,
        seed=42,
    )
    return grad, perm


grad_imp, perm_imp = get_feature_importance()

_num_perm = perm_imp[numerical_features].clip(lower=0)
_num_perm_norm = (_num_perm / _num_perm.max()) if _num_perm.max() > 0 else _num_perm


def _importance_badge(feature_name: str) -> str:
    """Return a short importance label for each sandbox slider."""
    score = float(_num_perm_norm.get(feature_name, 0.0))
    if score >= 0.60:
        return "★★★ High impact"
    if score >= 0.25:
        return "★★ Medium impact"
    return "★ Low impact"


def standardise_numerical(raw_values: np.ndarray) -> np.ndarray:
    """Apply the preprocess.py StandardScaler to a raw numerical vector."""
    return (raw_values - scaler_mean) / scaler_scale


def build_feature_vector(raw_numerical: dict, boro: str, cuisine_group: str) -> np.ndarray:
    """Assemble a single input vector matching feature_config['feature_columns']."""
    vec = np.zeros(input_dim, dtype=np.float32)

    raw = np.array([raw_numerical[c] for c in numerical_features], dtype=np.float32)
    scaled = standardise_numerical(raw)

    for name, value in zip(numerical_features, scaled):
        if name in feature_cols:
            vec[feature_cols.index(name)] = value

    boro_key = f"boro_{boro}"
    if boro_key in feature_cols:
        vec[feature_cols.index(boro_key)] = 1.0

    cuisine_key = f"cuisine_{cuisine_group}"
    if cuisine_key in feature_cols:
        vec[feature_cols.index(cuisine_key)] = 1.0
    elif "cuisine_Other" in feature_cols:
        vec[feature_cols.index("cuisine_Other")] = 1.0

    return vec


def predict_single(feature_vec: np.ndarray):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(feature_vec.astype(np.float32)).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).numpy()[0]
    return probs


# ---------------------------------------------------------------------------
# How the classifier works — a one-liner for the audience
# ---------------------------------------------------------------------------

st.info(
    f"**Model:** 3-layer MLP ({input_dim} → 128 → 128 → 3), trained on "
    f"{len(train_df):,} NYC restaurants with held-out test set of "
    f"{len(test_df):,}.  Grade {training_history.best_val_f1*100:.1f}% "
    f"validation F1 at best epoch ({training_history.best_epoch+1}).",
    icon="🧠",
)


# ---------------------------------------------------------------------------
# Section 1 — Live Prediction Sandbox
# ---------------------------------------------------------------------------

st.subheader("1️⃣ Live Prediction Sandbox")
st.caption(
    "Pick a restaurant, drag the inspection-feature sliders, and watch the predicted "
    "grade update."
)

query = st.text_input(
    "Search by name, borough, or cuisine",
    value="",
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
    filtered[["dba", "boro", "cuisine_description", "grade"]].rename(
        columns={
            "dba": "Restaurant",
            "boro": "Borough",
            "cuisine_description": "Cuisine",
            "grade": "Current Grade",
        }
    ),
    use_container_width=True,
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    height=260,
)

if selection.selection.rows:
    selected = filtered.iloc[selection.selection.rows[0]]
else:
    selected = filtered.iloc[0]
    st.caption("👆 Click a row to pick a different restaurant. Showing the first row by default.")

# Find this restaurant's row in the test set to seed the sliders.
test_row = test_df.loc[meta_test["camis"].astype(str).eq(str(selected["camis"])).values]
if test_row.empty:
    st.error("Could not find the feature row for this restaurant. Try another one.")
    st.stop()

# Recover the RAW, unstandardized numerical values by inverting the scaler.
scaled_vals = test_row[numerical_features].values[0].astype(np.float32)
raw_vals = scaled_vals * scaler_scale + scaler_mean
initial = dict(zip(numerical_features, raw_vals))

st.divider()

col_left, col_right = st.columns([1.15, 1])

available_cuisines = sorted(
    c.replace("cuisine_", "") for c in feature_cols if c.startswith("cuisine_")
)
available_boros = sorted(
    c.replace("boro_", "") for c in feature_cols if c.startswith("boro_")
)

with col_left:
    st.markdown(f"#### 🛠️ Sandbox: {selected['dba']}")
    st.caption(
        f"{selected['boro']} · {selected['cuisine_description']} · "
        f"Ground-truth grade: **{selected['grade']}**"
    )

    sb = {}

    def _clamp(val, lo, hi):
        return float(np.clip(val, lo, hi))

    r = slider_ranges

    sb["latest_score"] = st.slider(
        f"Most-recent inspection score  [{_importance_badge('latest_score')}]",
        r["latest_score"][0],
        r["latest_score"][1],
        _clamp(initial["latest_score"], *r["latest_score"]),
        help="DOHMH cutoffs: A ≤ 13, B 14–27, C ≥ 28. This is the single strongest predictor of grade.",
    )
    sb["critical_ratio"] = st.slider(
        f"Critical-violation ratio  [{_importance_badge('critical_ratio')}]",
        0.0,
        1.0,
        _clamp(initial["critical_ratio"], 0.0, 1.0),
        step=0.01,
        help="Fraction of recorded violations flagged Critical. Critical violations carry the heaviest regulatory weight.",
    )
    sb["violations_per_inspection"] = st.slider(
        f"Violations per inspection  [{_importance_badge('violations_per_inspection')}]",
        r["violations_per_inspection"][0],
        r["violations_per_inspection"][1],
        _clamp(initial["violations_per_inspection"], *r["violations_per_inspection"]),
        help="Average number of violations found each visit, normalized for inspection frequency.",
    )
    sb["avg_score"] = st.slider(
        f"Average score across all inspections  [{_importance_badge('avg_score')}]",
        r["avg_score"][0],
        r["avg_score"][1],
        _clamp(initial["avg_score"], *r["avg_score"]),
        help="Mean score across the restaurant's full inspection history.",
    )
    sb["num_inspections"] = st.slider(
        f"Total number of inspections on record  [{_importance_badge('num_inspections')}]",
        r["num_inspections"][0],
        r["num_inspections"][1],
        _clamp(initial["num_inspections"], *r["num_inspections"]),
        help="More inspections mean richer history for the model to use.",
    )
    sb["num_violations"] = st.slider(
        f"Total violation rows on record  [{_importance_badge('num_violations')}]",
        r["num_violations"][0],
        r["num_violations"][1],
        _clamp(initial["num_violations"], *r["num_violations"]),
        help="Raw count of violation rows. Largely captured by violations_per_inspection; included for completeness.",
    )
    sb["max_score"] = st.slider(
        f"Worst-ever inspection score  [{_importance_badge('max_score')}]",
        r["max_score"][0],
        r["max_score"][1],
        _clamp(initial["max_score"], *r["max_score"]),
        help="Peak score across all inspections.",
    )

    boro_choice = st.selectbox(
        "Borough",
        available_boros,
        index=available_boros.index(selected["boro"]) if selected["boro"] in available_boros else 0,
    )

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
        st.success(
            f"🎉 **Congratulations!** This restaurant is predicted to maintain a "
            f"**Grade A**. Confidence: {probs[0] * 100:.1f}%"
        )
        st.markdown(
            "Keep up the excellent hygiene standards. Continue routine inspections "
            "and maintain current operational practices."
        )
    elif pred_grade == "B":
        st.warning(
            f"⚠️ **At Risk — Grade B predicted.** Confidence: {probs[1] * 100:.1f}%"
        )
        st.markdown(
            "This restaurant has areas that need improvement. Review the guidance "
            "below to understand what changes could earn a Grade A."
        )
    else:
        st.error(
            f"🚨 **Critical — Grade C predicted.** Confidence: {probs[2] * 100:.1f}%"
        )
        st.markdown(
            "**Multiple areas need immediate attention.** A Grade C indicates serious "
            "hygiene concerns. See the specific changes needed below."
        )

    prob_fig = go.Figure(go.Bar(
        x=[f"Grade {g}" for g in GRADE_NAMES],
        y=list(probs),
        marker_color=[GRADE_COLORS[g] for g in GRADE_NAMES],
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
    ))
    prob_fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="P(class)"),
        height=240,
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="Inter, -apple-system, sans-serif",
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    if pred_grade != "A":
        st.markdown("##### 🎯 Improvement Guidance: How to Achieve Grade A")
        with st.spinner("Computing actionable recommendations..."):
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

            cf_raw = cf[:len(numerical_features)] * scaler_scale + scaler_mean

            FEATURE_ACTION = {
                "latest_score": "Penalty points from the most recent inspection. Reduce violations at the next visit.",
                "avg_score": "Mean penalty score across all past inspections. Fix recurring violations to lower the long-run average.",
                "max_score": "Highest single-inspection penalty score on record. Regular self-audits prevent catastrophic one-off failures.",
                "num_violations": "Total violation citations ever recorded. Reduce frequency through hygiene training and standard operating procedures.",
                "critical_ratio": "Share of violations flagged Critical. Eliminate these first because they carry the heaviest penalty.",
                "violations_per_inspection": "Average violations per visit. Improve per-visit hygiene through staff checklists and training.",
                "num_inspections": "Number of inspections on record; this reflects compliance history length and is not directly actionable.",
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
                st.dataframe(pd.DataFrame(cf_rows), use_container_width=True, hide_index=True)
                st.caption(
                    "↓ = needs to decrease · ↑ = needs to increase · "
                    "Minimum perturbation via gradient descent on inputs, model weights frozen."
                )
                if guidance_messages:
                    st.markdown("**What to change and why:**")
                    for msg in guidance_messages:
                        st.markdown(f"- {msg}")
            else:
                st.caption("The restaurant is already very close to the Grade-A decision boundary.")


# ---------------------------------------------------------------------------
# Section 3 — Held-out test set performance
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("2️⃣ How Good is the Classifier?")
st.caption(
    f"Evaluated on the held-out test set: {len(test_df):,} restaurants the model "
    "has never seen during training, validation, or model selection."
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
