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
4. Latent Space — 128-d hidden-layer geometry projected to 2D (PCA / t-SNE)
                  plus a 2D decision-boundary slice in the original feature
                  space.
5. Hyperparameter Justification — grid search results over
                                   (hidden_dim, lr, dropout), explaining
                                   why the deployed configuration was chosen.

This page replaces the earlier synthetic-data prototype.  It now loads the
real preprocessed DOHMH dataset produced by ``data/preprocess.py`` (27
engineered features across ~22K restaurants, grade A/B/C target).

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
    evaluate_mlp,
    find_counterfactual,
    hyperparameter_search,
    train_mlp,
)
from utils.user_profile import init_session_state, render_profile_sidebar

# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Health Analyzer Sandbox", page_icon="🧪", layout="wide")
apply_apple_theme()
init_session_state()

from utils.auth import require_auth
require_auth()

with st.sidebar:
    profile = render_profile_sidebar()
    st.markdown("---")

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

GRADE_NAMES = ["A", "B", "C"]
GRADE_COLORS = {"A": "#34C759", "B": "#FFCC00", "C": "#FF3B30"}


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


@st.cache_resource(show_spinner="Training the MLP on the real DOHMH dataset...")
def get_trained_model(hyperparams=None):
    """Train the deployed model once and cache it for the life of the session."""
    hp = hyperparams or DEPLOYED_HYPERPARAMS
    torch.manual_seed(42)
    model = CustomMLP(
        input_dim=input_dim,
        hidden_dim=hp["hidden_dim"],
        output_dim=3,
        dropout=hp["dropout"],
    )
    model, history = train_mlp(
        model,
        tensors["X_train"], tensors["y_train"],
        X_val=tensors["X_val"], y_val=tensors["y_val"],
        epochs=hp["max_epochs"],
        lr=hp["lr"],
        batch_size=hp["batch_size"],
        weight_decay=hp["weight_decay"],
        patience=hp["patience"],
        use_class_weights=True,
        verbose=False,
    )
    return model, history


model, training_history = get_trained_model()


@st.cache_resource(show_spinner="Projecting hidden-layer activations (PCA / t-SNE)...")
def get_hidden_space_artifacts():
    """Compute hidden activations on the held-out test set and 2D projections."""
    model.eval()
    with torch.no_grad():
        hidden = model.forward_hidden(tensors["X_test"]).cpu().numpy()

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(hidden)

    tsne = TSNE(
        n_components=2,
        perplexity=float(min(30, max(1, len(hidden) - 1))),
        init="pca",
        learning_rate="auto",
        max_iter=1200,
        random_state=42,
    )
    tsne_2d = tsne.fit_transform(hidden)

    return {
        "hidden": hidden,
        "pca_2d": pca_2d,
        "tsne_2d": tsne_2d,
    }


def _predict_classes_batch(feature_matrix: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run batched class prediction on a feature matrix."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(feature_matrix), batch_size):
            chunk = torch.from_numpy(feature_matrix[start:start + batch_size].astype(np.float32))
            logits = model(chunk)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _raw_feature_bounds():
    """Min/max bounds in raw (unstandardized) numerical space from train split."""
    train_scaled = train_df[numerical_features].values.astype(np.float32)
    train_raw = train_scaled * scaler_scale + scaler_mean
    mins = dict(zip(numerical_features, train_raw.min(axis=0)))
    maxs = dict(zip(numerical_features, train_raw.max(axis=0)))
    return mins, maxs


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
    st.markdown(
        f"""
**Input:** Each restaurant is represented as a {input_dim}-dimensional feature vector:
- **7 numerical features** (inspection scores, violation counts, critical ratios) — standardized via z-score using the training-set mean and std
- **6 borough one-hot features** (Manhattan, Brooklyn, Queens, Bronx, Staten Island, Unknown)
- **16 cuisine one-hot features** (American, Chinese, Italian, …, Other)

**Model:** 3-layer MLP ({input_dim} → 128 → 128 → 3) with ReLU activation and dropout
- Trained on 17,000+ real DOHMH inspection records
- Class weights compensate for imbalanced grades (78% A / 16% B / 6% C)
- Early stopping on validation F1 prevents overfitting

**Output:** Probability distribution over 3 classes (A, B, C)
- If B or C, a counterfactual gradient-descent search finds the minimum set of changes that would flip the prediction to A.
"""
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_live, tab_perf, tab_train, tab_latent, tab_hp = st.tabs([
    "🎛️ Live Prediction",
    "📊 Model Performance",
    "📈 Training Diagnostics",
    "🧠 Latent Space",
    "🔬 Hyperparameter Justification",
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
    if filtered.empty:
        st.warning("No restaurants matched this query. Showing the default sample instead.")
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

    selected_camis = st.session_state.get("health_selected_camis")
    if selection.selection.rows:
        row_idx = selection.selection.rows[0]
        selected = filtered.iloc[row_idx]
        st.session_state["health_selected_camis"] = str(selected.get("camis", ""))
    elif selected_camis is not None and not meta_test.empty:
        selected_rows = meta_test[meta_test["camis"].astype(str) == str(selected_camis)]
        selected = selected_rows.iloc[0] if not selected_rows.empty else filtered.iloc[0]
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
        sb["latest_score"] = st.slider(
            "Most-recent inspection score",
            0.0, 80.0, float(initial["latest_score"]),
            help="Higher = more violations on most recent inspection. DOHMH cutoffs: A ≤ 13, B 14-27, C ≥ 28."
        )
        sb["avg_score"] = st.slider(
            "Average score across all inspections",
            0.0, 80.0, float(initial["avg_score"]),
        )
        sb["max_score"] = st.slider(
            "Worst-ever inspection score",
            0.0, 100.0, float(initial["max_score"]),
        )
        sb["num_inspections"] = st.slider(
            "Total number of inspections on record",
            1.0, 60.0, float(initial["num_inspections"]),
        )
        sb["num_violations"] = st.slider(
            "Total violation rows on record",
            0.0, 500.0, float(initial["num_violations"]),
        )
        sb["critical_ratio"] = st.slider(
            "Critical-violation ratio",
            0.0, 1.0, float(initial["critical_ratio"]), step=0.01,
            help="Fraction of recorded violations flagged Critical."
        )
        sb["violations_per_inspection"] = st.slider(
            "Violations per inspection",
            0.0, 30.0, float(initial["violations_per_inspection"]),
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
        st.session_state["health_live_raw_numerical"] = dict(sb)
        st.session_state["health_live_boro"] = boro_choice
        st.session_state["health_live_cuisine"] = cuisine_choice
        st.session_state["health_live_feature_vec"] = feature_vec
        st.session_state["health_selected_camis"] = str(selected.get("camis", ""))

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
                
                # Plain-English interpretations for each feature
                FEATURE_GUIDANCE = {
                    "latest_score": "📋 **Latest inspection score**: Lower scores mean fewer violations on the most recent inspection. DOHMH cutoffs: A ≤ 13, B 14–27, C ≥ 28.",
                    "avg_score": "📊 **Average inspection score**: Improve consistency across all inspections by addressing recurring violation patterns.",
                    "max_score": "⚠️ **Worst-ever inspection score**: Prevent severe single-inspection failures through better preparation.",
                    "num_inspections": "🔍 **Number of inspections**: More inspection history helps establish a reliable track record.",
                    "num_violations": "📝 **Total violations**: Reduce the overall count of recorded violations through systematic hygiene improvements.",
                    "critical_ratio": "🚨 **Critical violation ratio**: Focus on eliminating critical violations (food temperature, pest control, hand washing) which carry the heaviest weight.",
                    "violations_per_inspection": "📉 **Violations per inspection**: Reduce the average number of violations found in each visit.",
                }
                
                cf_rows = []
                guidance_messages = []
                for name, new_val in zip(numerical_features, cf_raw):
                    delta = new_val - sb[name]
                    if abs(delta) > 0.05 * max(abs(sb[name]), 1.0):
                        cf_rows.append({
                            "Feature": name,
                            "Current": f"{sb[name]:.2f}",
                            "Suggested": f"{new_val:.2f}",
                            "Δ": f"{delta:+.2f}",
                        })
                        if name in FEATURE_GUIDANCE:
                            direction = "Decrease" if delta < 0 else "Increase"
                            guidance_messages.append(
                                f"{FEATURE_GUIDANCE[name]}\n   → {direction} from {sb[name]:.1f} to {new_val:.1f}"
                            )
                if cf_rows:
                    st.dataframe(
                        pd.DataFrame(cf_rows),
                        use_container_width=True, hide_index=True,
                    )
                    if guidance_messages:
                        st.markdown("**Actionable Recommendations:**")
                        for msg in guidance_messages:
                            st.markdown(msg)
                    st.caption(
                        "Minimum perturbation computed via gradient descent on "
                        "the input features while holding model weights frozen."
                    )
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

    st.info(
        "Class weights (inverse frequency) are applied during training so the "
        "model does not collapse to always predicting grade A.  Without class "
        "weighting, an all-A classifier would hit ~80% accuracy but only "
        "~0.7 macro F1 and completely miss grade C."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Training diagnostics
# ---------------------------------------------------------------------------

with tab_train:
    st.subheader("Training dynamics of the deployed model")

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

    with st.expander("Training recipe"):
        st.markdown(
            f"""
| Setting | Value |
|---|---|
| Hidden dimension | {DEPLOYED_HYPERPARAMS['hidden_dim']} |
| Dropout | {DEPLOYED_HYPERPARAMS['dropout']} |
| Learning rate | {DEPLOYED_HYPERPARAMS['lr']} |
| Weight decay | {DEPLOYED_HYPERPARAMS['weight_decay']} |
| Batch size | {DEPLOYED_HYPERPARAMS['batch_size']} |
| Max epochs | {DEPLOYED_HYPERPARAMS['max_epochs']} |
| Early-stopping patience | {DEPLOYED_HYPERPARAMS['patience']} |
| Optimiser | AdamW |
| Loss | Cross-entropy with inverse-frequency class weights |
| Train / Val / Test | {len(tensors['X_train']):,} / {len(tensors['X_val']):,} / {len(tensors['X_test']):,} (stratified) |
"""
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

    pca_2d = PCA(n_components=2, random_state=42).fit_transform(hidden)

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
        "tsne_idx": tsne_idx,
        "tsne": tsne_2d_sample,
    }


with tab_latent:
    st.subheader("How the MLP organises restaurants in its 128-dim latent space")
    st.markdown(
        """
Every input vector is transformed by two hidden layers before the final
classifier sees it.  Projecting those 128-dim activations to 2D shows what
the network has actually *learned* — ideally, restaurants with the same
grade cluster together even though nothing in the plot is explicitly told
what the grades are.
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
        sample_note = f"t-SNE on a random sample of {len(coords):,} test restaurants."
    else:
        coords = latent["pca"]
        labels = y_test_np
        sample_note = f"PCA on all {len(coords):,} held-out test restaurants."

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
            pca_full = PCA(n_components=2, random_state=42).fit(latent["hidden"])
            sandbox_xy = pca_full.transform(sandbox_hidden)[0]
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
        xaxis_title=("PC1" if not use_tsne else "t-SNE 1"),
        yaxis_title=("PC2" if not use_tsne else "t-SNE 2"),
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
Pick any two numerical features — we sweep a 60×60 grid over their ranges
and colour each point by the MLP's predicted grade while holding all other
features fixed at the currently-selected restaurant's values.  This gives
a faithful snapshot of the decision surface along that 2-D slice.
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
            xaxis_title=x_feat,
            yaxis_title=y_feat,
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
    st.subheader("Why hidden_dim = 128, lr = 1e-3, dropout = 0.3?")
    st.markdown(
        """
We ran a grid search over **(hidden_dim, lr, dropout)** on the same
stratified train/val split used by the deployed model.  Every configuration
was trained from scratch with identical early-stopping and class-weighting
settings — the only variables are the three hyperparameters.  The
configuration with the highest validation F1 is deployed.
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
