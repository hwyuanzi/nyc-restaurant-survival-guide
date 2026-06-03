"""
Regression tests guarding the train/test leakage fix in data/preprocess.py.

These assert that the HealthFeaturePreprocessor learns its imputation medians,
standardisation statistics, and cuisine vocabulary from the *training rows
only*, and that transforming the held-out rows never recomputes or shifts those
statistics.  They are deliberately constructed so that fitting on the combined
train+test data would produce visibly different numbers — if the leakage ever
returns, these tests fail.
"""

import numpy as np
import pandas as pd
import pytest

from data.preprocess import HealthFeaturePreprocessor


def _agg_frame(num_inspections, num_violations, cuisines, boros, grades):
    """Build an aggregated-restaurant frame like aggregate_per_restaurant()."""
    n = len(num_inspections)
    vpi = [v / max(i, 1) for v, i in zip(num_violations, num_inspections)]
    return pd.DataFrame({
        "camis": list(range(n)),
        "dba": [f"R{i}" for i in range(n)],
        "boro": boros,
        "cuisine_description": cuisines,
        "building": ["1"] * n,
        "street": ["Main St"] * n,
        "zipcode": ["10001"] * n,
        "grade": grades,
        "num_inspections": num_inspections,
        "num_violations": num_violations,
        "violations_per_inspection": vpi,
    })


@pytest.fixture
def train_and_test():
    # Training rows: tightly distributed, dominant cuisines are Pizza/Chinese.
    train = _agg_frame(
        num_inspections=[1, 1, 2, 2, 2, 3, 3, 1, 2, 2],
        num_violations=[1, 2, 2, 3, 3, 4, 4, 1, 2, 3],
        cuisines=["Pizza", "Pizza", "Pizza", "Chinese", "Chinese",
                  "Chinese", "Italian", "Pizza", "Chinese", "Italian"],
        boros=["Manhattan"] * 5 + ["Brooklyn"] * 5,
        grades=["A", "A", "A", "B", "A", "B", "A", "A", "C", "A"],
    )
    # Test rows: deliberately extreme counts and a cuisine ("Ethiopian") and a
    # borough ("Queens") that never appear in the training split.
    test = _agg_frame(
        num_inspections=[40, 50],
        num_violations=[200, 250],
        cuisines=["Ethiopian", "Pizza"],
        boros=["Queens", "Manhattan"],
        grades=["C", "A"],
    )
    return train, test


def test_scaler_statistics_are_train_only(train_and_test):
    train, test = train_and_test
    pre = HealthFeaturePreprocessor().fit(train)

    # Expected statistics from the TRAIN rows only (StandardScaler uses ddof=0).
    for i, col in enumerate(pre.numerical_features):
        train_vals = train[col].to_numpy(dtype=float)
        assert pre.scaler_mean_[i] == pytest.approx(train_vals.mean())
        assert pre.scaler_scale_[i] == pytest.approx(train_vals.std())

    # The combined train+test mean is very different (test has extreme counts),
    # so a leaky fit on all rows would land far from the train-only statistics.
    combined = pd.concat([train, test], ignore_index=True)
    for i, col in enumerate(pre.numerical_features):
        combined_mean = combined[col].to_numpy(dtype=float).mean()
        assert abs(pre.scaler_mean_[i] - combined_mean) > 1e-6


def test_imputation_medians_are_train_only(train_and_test):
    train, test = train_and_test
    pre = HealthFeaturePreprocessor().fit(train)
    for col in pre.numerical_features:
        assert pre.numerical_medians_[col] == pytest.approx(
            float(train[col].median())
        )


def test_cuisine_vocabulary_is_train_only(train_and_test):
    train, test = train_and_test
    pre = HealthFeaturePreprocessor(top_n_cuisines=3).fit(train)

    # Vocabulary comes from train counts; the test-only cuisine cannot appear.
    assert "Ethiopian" not in pre.top_cuisines_
    assert "cuisine_Ethiopian" not in pre.feature_columns_
    assert set(pre.top_cuisines_) <= set(train["cuisine_description"])


def test_transform_does_not_mutate_fitted_statistics(train_and_test):
    train, test = train_and_test
    pre = HealthFeaturePreprocessor().fit(train)

    before_mean = list(pre.scaler_mean_)
    before_scale = list(pre.scaler_scale_)
    before_vocab = list(pre.top_cuisines_)

    pre.transform(test)  # transforming held-out rows must not relearn anything

    assert pre.scaler_mean_ == before_mean
    assert pre.scaler_scale_ == before_scale
    assert pre.top_cuisines_ == before_vocab


def test_unseen_test_categories_fall_back_safely(train_and_test):
    train, test = train_and_test
    pre = HealthFeaturePreprocessor().fit(train)
    transformed = pre.transform(test)

    # Columns are exactly the trained schema, in order.
    assert list(transformed.columns) == pre.feature_columns_ + ["target"]
    # Unseen "Ethiopian" cuisine is bucketed into the always-present "Other".
    assert transformed.iloc[0]["cuisine_Other"] == 1
    # Unseen "Queens" borough produces an all-zero borough sub-vector.
    boro_cols = [c for c in pre.feature_columns_ if c.startswith("boro_")]
    assert transformed.iloc[0][boro_cols].sum() == 0


def test_transform_is_consistent_for_identical_rows(train_and_test):
    train, _ = train_and_test
    pre = HealthFeaturePreprocessor().fit(train)
    first = pre.transform(train)
    second = pre.transform(train)
    pd.testing.assert_frame_equal(first, second)
