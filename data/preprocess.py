"""
preprocess.py — NYC DOHMH Restaurant Inspection Data Preprocessing Pipeline
 
Aggregates raw inspection rows into one row per restaurant, engineers numerical
and categorical features, and produces train/test splits ready for the MLP and
Autoencoder models.
 
Author: Ryan Han (Data & DevOps)
Course: CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026
"""
 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
 
 
# ---------------------------------------------------------------------------
# 1. Core Preprocessing
# ---------------------------------------------------------------------------
 
def load_and_clean(input_path="data/raw_dohmh.csv"):
    """Load the raw DOHMH CSV and perform initial cleaning."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Run download_data.py first:\n"
            f"  python data/download_data.py"
        )
 
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")
 
    # Parse inspection date
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
 
    # Drop rows that have never been inspected (date = 1900-01-01)
    df = df[df["inspection_date"] > "1901-01-01"]
 
    # Keep only rows with a valid grade (A / B / C)
    df = df.dropna(subset=["grade"])
    df = df[df["grade"].isin(["A", "B", "C"])]
    print(f"  Rows after filtering to graded inspections: {len(df):,}")
 
    return df
 
 
def aggregate_per_restaurant(df):
    """
    The raw data has one row per *violation* — a single inspection can have
    many rows.  We aggregate to one row per restaurant (identified by CAMIS),
    keeping the most recent inspection's grade as the prediction target and
    engineering meaningful features from the full inspection history.
    """
 
    # Sort so the most recent inspection comes last
    df = df.sort_values("inspection_date")
 
    agg = df.groupby("camis").agg(
        # ---- identification (kept for the app / semantic search) ----
        dba=("dba", "last"),
        boro=("boro", "last"),
        cuisine_description=("cuisine_description", "last"),
        building=("building", "last"),
        street=("street", "last"),
        zipcode=("zipcode", "last"),
 
        # ---- target: most recent grade ----
        grade=("grade", "last"),
 
        # ---- numerical features ----
        latest_score=("score", "last"),                 # score of most recent inspection
        avg_score=("score", "mean"),                    # average score across all inspections
        max_score=("score", "max"),                     # worst single inspection
        num_inspections=("inspection_date", "nunique"),  # how many inspections
        num_violations=("camis", "size"),                # total violation rows
 
        # ---- critical flag counts ----
        num_critical=("critical_flag", lambda x: (x == "Critical").sum()),
        num_not_critical=("critical_flag", lambda x: (x == "Not Critical").sum()),
    ).reset_index()
 
    # Derived features
    agg["critical_ratio"] = agg["num_critical"] / agg["num_violations"].clip(lower=1)
    agg["violations_per_inspection"] = agg["num_violations"] / agg["num_inspections"].clip(lower=1)
 
    print(f"  Unique restaurants after aggregation: {len(agg):,}")
    return agg
 
 
# ---------------------------------------------------------------------------
# 2. Feature Engineering for Models
# ---------------------------------------------------------------------------
 
# Numerical features the MLP / Autoencoder will use
NUMERICAL_FEATURES = [
    "num_inspections",
    "num_violations",
    "violations_per_inspection",
]
 
# Categorical features to one-hot encode
CATEGORICAL_FEATURES = ["boro"]
 
# Top-N cuisines to keep (rest → "Other") to avoid sparse matrix explosion
TOP_N_CUISINES = 15
 
GRADE_MAP = {"A": 0, "B": 1, "C": 2}
 
 
def engineer_features(agg_df):
    """
    Build the final feature matrix:
      - Standardised numerical columns
      - One-hot encoded borough
      - One-hot encoded top-N cuisines
    Also returns metadata (restaurant info) and the target column.
    """
 
    df = agg_df.copy()
 
    # ---- target ----
    df["target"] = df["grade"].map(GRADE_MAP)
 
    # ---- numerical: fill NaN with median, then standardise ----
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
 
    scaler = StandardScaler()
    df[NUMERICAL_FEATURES] = scaler.fit_transform(df[NUMERICAL_FEATURES])
 
    # ---- categorical: borough one-hot ----
    df["boro"] = df["boro"].fillna("Unknown")
    boro_dummies = pd.get_dummies(df["boro"], prefix="boro").astype(int)
 
    # ---- categorical: cuisine one-hot (top N) ----
    df["cuisine_description"] = df["cuisine_description"].fillna("Unknown")
    top_cuisines = df["cuisine_description"].value_counts().head(TOP_N_CUISINES).index.tolist()
    df["cuisine_group"] = df["cuisine_description"].where(
        df["cuisine_description"].isin(top_cuisines), other="Other"
    )
    cuisine_dummies = pd.get_dummies(df["cuisine_group"], prefix="cuisine").astype(int)
 
    # ---- assemble feature matrix ----
    feature_cols = NUMERICAL_FEATURES + list(boro_dummies.columns) + list(cuisine_dummies.columns)
    features_df = pd.concat([df[NUMERICAL_FEATURES], boro_dummies, cuisine_dummies], axis=1)
    features_df["target"] = df["target"].values
 
    # ---- metadata for the app (not used in training) ----
    meta_cols = ["camis", "dba", "boro", "cuisine_description", "building",
                 "street", "zipcode", "grade"]
    # Use original boro before one-hot (still in df)
    meta_df = agg_df[["camis", "dba", "boro", "cuisine_description",
                       "building", "street", "zipcode", "grade"]].copy()
 
    return features_df, meta_df, feature_cols, scaler, top_cuisines
 
 
# ---------------------------------------------------------------------------
# 3. Split & Save
# ---------------------------------------------------------------------------
 
def save_splits(features_df, meta_df, feature_cols, scaler, top_cuisines,
                output_dir="data"):
    """Save train/test CSVs and a config JSON so the app knows the feature schema."""
 
    os.makedirs(output_dir, exist_ok=True)
 
    train_df, test_df, train_meta, test_meta = train_test_split(
        features_df, meta_df, test_size=0.2, random_state=42, stratify=features_df["target"]
    )
 
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    meta_train_path = os.path.join(output_dir, "meta_train.csv")
    meta_test_path = os.path.join(output_dir, "meta_test.csv")
    config_path = os.path.join(output_dir, "feature_config.json")
 
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_meta.to_csv(meta_train_path, index=False)
    test_meta.to_csv(meta_test_path, index=False)
 
    # Save feature schema so the app and models know what to expect
    config = {
        "numerical_features": NUMERICAL_FEATURES,
        "feature_columns": feature_cols,
        "input_dim": len(feature_cols),
        "grade_map": GRADE_MAP,
        "top_cuisines": top_cuisines,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
 
    print(f"\nPreprocessing complete!")
    print(f"  Train set: {train_path} ({len(train_df):,} restaurants)")
    print(f"  Test set:  {test_path} ({len(test_df):,} restaurants)")
    print(f"  Metadata:  {meta_train_path}, {meta_test_path}")
    print(f"  Config:    {config_path}")
    print(f"  Input dim: {len(feature_cols)} features")
    print(f"  Target distribution (train):")
    for grade, idx in GRADE_MAP.items():
        count = (train_df["target"] == idx).sum()
        print(f"    Grade {grade}: {count:,} ({count / len(train_df) * 100:.1f}%)")
 
 
# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
 
def preprocess_dohmh(input_path="data/raw_dohmh.csv", output_dir="data"):
    """Full pipeline: load → clean → aggregate → feature engineer → split → save."""
    df = load_and_clean(input_path)
    agg_df = aggregate_per_restaurant(df)
    features_df, meta_df, feature_cols, scaler, top_cuisines = engineer_features(agg_df)
    save_splits(features_df, meta_df, feature_cols, scaler, top_cuisines, output_dir)
 
 
if __name__ == "__main__":
    preprocess_dohmh()
 