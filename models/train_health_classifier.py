"""
train_health_classifier.py — Reproducible offline trainer for the health MLP.

This mirrors the exact training recipe used by the Streamlit page
(``app/pages/2_🧪_Health_Grade_Classifier.py``) so the committed checkpoint
(``data/cache/health_classifier.pt``) and its history can be regenerated from a
terminal without launching the app.  It also evaluates the trained model on the
held-out test split and prints the metrics that the README quotes.

Leakage note: ``data/train.csv`` / ``data/test.csv`` are produced by
``data/preprocess.py``, which fits all imputation/scaling/vocabulary statistics
on the training split only.  This script never recomputes those statistics; it
just consumes the already-transformed feature matrices.

Usage:
    python -m models.train_health_classifier

Course: CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models.custom_mlp import CustomMLP, evaluate_mlp, train_mlp

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
CONFIG_PATH = DATA_DIR / "feature_config.json"
MODEL_CACHE_PATH = CACHE_DIR / "health_classifier.pt"
HISTORY_CACHE_PATH = CACHE_DIR / "health_classifier_history.json"
IMPORTANCE_CACHE_PATH = CACHE_DIR / "health_classifier_importance.json"

GRADE_NAMES = ["A", "B", "C"]

# Identical to HYPERPARAMS in the Streamlit page.
HYPERPARAMS = {
    "hidden_dim": 128, "lr": 1e-3, "dropout": 0.3,
    "batch_size": 128, "weight_decay": 1e-4,
    "max_epochs": 80, "patience": 12,
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    feature_cols = [c for c in config["feature_columns"] if c in train_df.columns]
    return train_df, test_df, feature_cols


def _tensors(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, feature_cols = _load()
    input_dim = len(feature_cols)

    X_full, y_full = _tensors(train_df, feature_cols)
    X_test, y_test = _tensors(test_df, feature_cols)

    # Stratified train/validation split from the training rows (matches page).
    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_full.numpy(), y_full.numpy(),
        test_size=0.2, stratify=y_full.numpy(), random_state=42,
    )
    X_train, y_train = torch.from_numpy(X_tr_np), torch.from_numpy(y_tr_np)
    X_val, y_val = torch.from_numpy(X_val_np), torch.from_numpy(y_val_np)

    torch.manual_seed(42)
    model = CustomMLP(input_dim=input_dim, hidden_dim=HYPERPARAMS["hidden_dim"],
                      output_dim=3, dropout=HYPERPARAMS["dropout"])
    model, history = train_mlp(
        model, X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=HYPERPARAMS["max_epochs"], lr=HYPERPARAMS["lr"],
        batch_size=HYPERPARAMS["batch_size"],
        weight_decay=HYPERPARAMS["weight_decay"],
        patience=HYPERPARAMS["patience"],
        use_class_weights=True, verbose=True,
    )

    torch.save(model.state_dict(), MODEL_CACHE_PATH)
    with open(HISTORY_CACHE_PATH, "w") as f:
        json.dump({
            "train_loss": history.train_loss, "val_loss": history.val_loss,
            "train_f1": history.train_f1, "val_f1": history.val_f1,
            "best_epoch": history.best_epoch,
            "best_val_f1": float(history.best_val_f1),
            "stopped_early": bool(history.stopped_early),
        }, f)

    # Invalidate the cached permutation-importance file: feature schema may have
    # changed, and the page regenerates it lazily on next load.
    if IMPORTANCE_CACHE_PATH.exists():
        IMPORTANCE_CACHE_PATH.unlink()

    details = evaluate_mlp(model, X_test, y_test, class_names=GRADE_NAMES,
                           return_details=True)
    preds = details["predictions"]
    y_true = y_test.numpy()
    acc = float((preds == y_true).mean())
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = float(details["weighted_f1"])

    print("\nHeld-out test metrics (train-only preprocessing):")
    print(f"  Test restaurants: {len(test_df):,}")
    print(f"  Accuracy:    {acc * 100:.1f}%")
    print(f"  Weighted F1: {weighted_f1:.3f}")
    print(f"  Macro F1:    {macro_f1:.3f}")
    print(f"  Best val F1: {history.best_val_f1 * 100:.1f}% @ epoch {history.best_epoch + 1}")
    print(f"\n  Saved checkpoint -> {MODEL_CACHE_PATH}")
    print(f"  Saved history    -> {HISTORY_CACHE_PATH}")


if __name__ == "__main__":
    main()
