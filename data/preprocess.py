"""
preprocess.py — NYC DOHMH Restaurant Inspection Data Preprocessing Pipeline

Aggregates raw inspection rows into one row per restaurant, then learns every
preprocessing statistic (median imputation values, standardisation mean/scale,
and the top-cuisine vocabulary) **only from the training split**.  The fitted
statistics are replayed unchanged on the held-out test rows so the reported
classifier metrics are not contaminated by test-set distribution information.

Pipeline order (leakage-safe):

    load -> clean -> aggregate
         -> train_test_split (on the raw aggregated rows)
         -> HealthFeaturePreprocessor.fit(train rows only)
         -> transform(train) and transform(test) with the *same* fitted object
         -> save train/test/meta CSVs, feature_config.json, and the fitted
            preprocessor (data/cache/health_preprocessor.joblib)

Why this matters: imputation medians, the StandardScaler statistics, and the
cuisine vocabulary are all parameters learned from data.  If they are fit on
the full dataset before splitting, the test rows leak their distribution into
the training-time representation and held-out metrics look more stable than a
clean protocol would produce.  Treating preprocessing as part of the model and
fitting it on the training split only is the production-correct habit.

Course: CSCI-UA 473 · Fundamentals of Machine Learning · Spring 2026
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:  # joblib is a hard dependency, but keep the import defensive for tooling
    import joblib
except ImportError:  # pragma: no cover - joblib is in requirements.txt
    joblib = None


# ---------------------------------------------------------------------------
# Feature schema constants
# ---------------------------------------------------------------------------

# Numerical inspection-history features used by the health grade MLP.
NUMERICAL_FEATURES = [
    "num_inspections",
    "num_violations",
    "violations_per_inspection",
]

# Categorical column that is one-hot encoded directly.
BORO_COLUMN = "boro"

# Top-N cuisines to keep (the rest collapse to "Other") to avoid a sparse
# matrix explosion.  The actual vocabulary is learned from the training split.
TOP_N_CUISINES = 15

GRADE_MAP = {"A": 0, "B": 1, "C": 2}

# Path where the fitted training-only preprocessor is cached for reuse.
PREPROCESSOR_PATH = os.path.join("data", "cache", "health_preprocessor.joblib")


# ---------------------------------------------------------------------------
# 1. Load / clean / aggregate (no statistics are learned here)
# ---------------------------------------------------------------------------

def load_and_clean(input_path="data/raw_dohmh.csv"):
    """Load the raw DOHMH CSV and perform initial, row-local cleaning.

    Nothing learned here depends on the train/test split: every operation is a
    per-row filter or a type coercion, so it is safe to run before splitting.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Run download_data.py first:\n"
            f"  python data/download_data.py"
        )

    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")

    # Drop rows that have never been inspected (placeholder date 1900-01-01).
    df = df[df["inspection_date"] > "1901-01-01"]

    # Keep only rows with a valid letter grade (A / B / C).
    df = df.dropna(subset=["grade"])
    df = df[df["grade"].isin(["A", "B", "C"])]
    print(f"  Rows after filtering to graded inspections: {len(df):,}")

    return df


def aggregate_per_restaurant(df):
    """Collapse one-row-per-violation data into one row per restaurant (CAMIS).

    The most recent inspection's grade becomes the prediction target.  The
    engineered numerical columns are left in *raw* units here — standardisation
    happens later, inside the training-only preprocessor.
    """
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

        # ---- numerical features (raw, unscaled) ----
        latest_score=("score", "last"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        num_inspections=("inspection_date", "nunique"),
        num_violations=("camis", "size"),

        # ---- critical flag counts ----
        num_critical=("critical_flag", lambda x: (x == "Critical").sum()),
        num_not_critical=("critical_flag", lambda x: (x == "Not Critical").sum()),
    ).reset_index()

    # Derived rate features (still raw units).
    agg["critical_ratio"] = agg["num_critical"] / agg["num_violations"].clip(lower=1)
    agg["violations_per_inspection"] = agg["num_violations"] / agg["num_inspections"].clip(lower=1)

    print(f"  Unique restaurants after aggregation: {len(agg):,}")
    return agg


# ---------------------------------------------------------------------------
# 2. Training-only preprocessing object (explicit fit / transform stages)
# ---------------------------------------------------------------------------

@dataclass
class HealthFeaturePreprocessor:
    """Learns every preprocessing statistic from the training rows only.

    ``fit`` records imputation medians, standardisation statistics, the
    cuisine vocabulary, and the one-hot column schema from the training split.
    ``transform`` replays those fixed statistics on any rows (train, test, or a
    single live inference row) without ever recomputing them.  This guarantees
    the held-out test set never influences the feature representation.
    """

    top_n_cuisines: int = TOP_N_CUISINES
    numerical_features: list = field(default_factory=lambda: list(NUMERICAL_FEATURES))

    # --- learned during fit (populated only from training rows) ---
    numerical_medians_: dict = field(default_factory=dict)
    scaler_mean_: list = field(default_factory=list)
    scaler_scale_: list = field(default_factory=list)
    top_cuisines_: list = field(default_factory=list)
    boro_categories_: list = field(default_factory=list)
    cuisine_categories_: list = field(default_factory=list)
    feature_columns_: list = field(default_factory=list)
    fitted_: bool = False

    # ---- column-name helpers (kept consistent between fit and transform) ----

    @staticmethod
    def _boro_col(value: str) -> str:
        return f"boro_{value}"

    @staticmethod
    def _cuisine_col(value: str) -> str:
        return f"cuisine_{value}"

    def fit(self, train_df: pd.DataFrame) -> "HealthFeaturePreprocessor":
        """Learn imputation, scaling, and vocabulary from the training rows."""
        df = train_df.copy()

        # ---- numerical imputation medians (training rows only) ----
        numeric_clean = pd.DataFrame()
        for col in self.numerical_features:
            series = pd.to_numeric(df[col], errors="coerce")
            median = float(series.median())
            self.numerical_medians_[col] = median
            numeric_clean[col] = series.fillna(median)

        # ---- standardisation statistics (fit on imputed training rows) ----
        scaler = StandardScaler()
        scaler.fit(numeric_clean[self.numerical_features].to_numpy())
        self.scaler_mean_ = scaler.mean_.tolist()
        # Guard against zero-variance columns producing div-by-zero at transform.
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        self.scaler_scale_ = scale.tolist()

        # ---- borough one-hot vocabulary (training rows only) ----
        boro_values = df[BORO_COLUMN].fillna("Unknown").astype(str)
        self.boro_categories_ = sorted(boro_values.unique().tolist())

        # ---- cuisine vocabulary: top-N from training rows only ----
        cuisine_values = df["cuisine_description"].fillna("Unknown").astype(str)
        self.top_cuisines_ = (
            cuisine_values.value_counts().head(self.top_n_cuisines).index.tolist()
        )
        # "Other" is always present so unseen test cuisines have a valid bucket.
        self.cuisine_categories_ = sorted(self.top_cuisines_) + ["Other"]

        # ---- final, ordered feature column schema ----
        self.feature_columns_ = (
            list(self.numerical_features)
            + [self._boro_col(b) for b in self.boro_categories_]
            + [self._cuisine_col(c) for c in self.cuisine_categories_]
        )

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """Apply the fitted statistics to any set of aggregated rows."""
        if not self.fitted_:
            raise RuntimeError("HealthFeaturePreprocessor.transform called before fit.")

        source = df.copy()
        out = pd.DataFrame(index=source.index)

        # ---- numerical: impute with training medians, then standardise ----
        for i, col in enumerate(self.numerical_features):
            series = pd.to_numeric(source[col], errors="coerce")
            series = series.fillna(self.numerical_medians_[col])
            out[col] = (series - self.scaler_mean_[i]) / self.scaler_scale_[i]

        # ---- borough one-hot, reindexed to the training vocabulary ----
        boro_values = source[BORO_COLUMN].fillna("Unknown").astype(str)
        for category in self.boro_categories_:
            out[self._boro_col(category)] = (boro_values == category).astype(int)

        # ---- cuisine one-hot, collapsing unseen cuisines into "Other" ----
        cuisine_values = source["cuisine_description"].fillna("Unknown").astype(str)
        cuisine_group = cuisine_values.where(
            cuisine_values.isin(self.top_cuisines_), other="Other"
        )
        for category in self.cuisine_categories_:
            out[self._cuisine_col(category)] = (cuisine_group == category).astype(int)

        # ---- enforce the exact trained column order ----
        out = out.reindex(columns=self.feature_columns_, fill_value=0)

        if include_target and "grade" in source.columns:
            out["target"] = source["grade"].map(GRADE_MAP).values

        return out

    def fit_transform(self, train_df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df, include_target=include_target)

    # ---- serialisation ----

    def to_config(self) -> dict:
        """Export the schema/statistics the Streamlit app and models rely on."""
        return {
            "numerical_features": list(self.numerical_features),
            "feature_columns": list(self.feature_columns_),
            "input_dim": len(self.feature_columns_),
            "grade_map": GRADE_MAP,
            "top_cuisines": list(self.top_cuisines_),
            "boro_categories": list(self.boro_categories_),
            "cuisine_categories": list(self.cuisine_categories_),
            "numerical_medians": dict(self.numerical_medians_),
            "scaler_mean": list(self.scaler_mean_),
            "scaler_scale": list(self.scaler_scale_),
            "fit_on": "train_split_only",
        }

    def save(self, path: str = PREPROCESSOR_PATH) -> None:
        if joblib is None:  # pragma: no cover
            raise RuntimeError("joblib is required to save the preprocessor.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = PREPROCESSOR_PATH) -> "HealthFeaturePreprocessor":
        if joblib is None:  # pragma: no cover
            raise RuntimeError("joblib is required to load the preprocessor.")
        return joblib.load(path)


def build_meta(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Restaurant identification metadata (never fed to the model)."""
    meta_cols = ["camis", "dba", "boro", "cuisine_description",
                 "building", "street", "zipcode", "grade"]
    return agg_df[meta_cols].copy()


# ---------------------------------------------------------------------------
# 3. Split first, fit on train only, then transform and save
# ---------------------------------------------------------------------------

def split_aggregated(agg_df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split the *raw* aggregated rows before any statistic is learned."""
    train_agg, test_agg = train_test_split(
        agg_df,
        test_size=test_size,
        random_state=seed,
        stratify=agg_df["grade"],
    )
    return train_agg.reset_index(drop=True), test_agg.reset_index(drop=True)


def save_splits(train_agg, test_agg, preprocessor: HealthFeaturePreprocessor,
                output_dir="data"):
    """Transform both splits with the train-fit preprocessor and write outputs."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = preprocessor.transform(train_agg)
    test_df = preprocessor.transform(test_agg)
    train_meta = build_meta(train_agg)
    test_meta = build_meta(test_agg)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    meta_train_path = os.path.join(output_dir, "meta_train.csv")
    meta_test_path = os.path.join(output_dir, "meta_test.csv")
    config_path = os.path.join(output_dir, "feature_config.json")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_meta.to_csv(meta_train_path, index=False)
    test_meta.to_csv(meta_test_path, index=False)

    with open(config_path, "w") as f:
        json.dump(preprocessor.to_config(), f, indent=2)

    preprocessor.save(os.path.join(output_dir, "cache", "health_preprocessor.joblib"))

    print("\nPreprocessing complete (train-only statistics):")
    print(f"  Train set: {train_path} ({len(train_df):,} restaurants)")
    print(f"  Test set:  {test_path} ({len(test_df):,} restaurants)")
    print(f"  Metadata:  {meta_train_path}, {meta_test_path}")
    print(f"  Config:    {config_path}")
    print(f"  Preproc:   {os.path.join(output_dir, 'cache', 'health_preprocessor.joblib')}")
    print(f"  Input dim: {len(preprocessor.feature_columns_)} features")
    print("  Target distribution (train):")
    for grade, idx in GRADE_MAP.items():
        count = int((train_df["target"] == idx).sum())
        print(f"    Grade {grade}: {count:,} ({count / len(train_df) * 100:.1f}%)")


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def preprocess_dohmh(input_path="data/raw_dohmh.csv", output_dir="data"):
    """Full pipeline: load -> clean -> aggregate -> split -> fit(train) -> save."""
    df = load_and_clean(input_path)
    agg_df = aggregate_per_restaurant(df)
    train_agg, test_agg = split_aggregated(agg_df)
    preprocessor = HealthFeaturePreprocessor().fit(train_agg)
    save_splits(train_agg, test_agg, preprocessor, output_dir)


if __name__ == "__main__":
    preprocess_dohmh()
