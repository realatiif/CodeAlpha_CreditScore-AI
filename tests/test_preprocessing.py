"""Unit tests for data loading, dataset validation, preprocessing, and feature engineering.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_dataset,
    validate_dataset,
    detect_target_candidates,
    inspect_target_distribution,
    prepare_labeled_data,
    get_default_dataset_path,
)
from src.feature_engineering import CreditFeatureEngineer
from src.preprocessing import (
    identify_column_types,
    create_preprocessor,
    build_full_preprocessing_pipeline,
    split_data,
    compute_class_imbalance_metrics,
)


@pytest.fixture
def sample_credit_data() -> pd.DataFrame:
    """Return a clean mock credit DataFrame resembling the German Credit schema."""
    np.random.seed(42)
    n = 100
    data = {
        "checking_status": np.random.choice(["<0", "0<=X<200", ">=200", "no checking"], size=n),
        "duration": np.random.randint(6, 48, size=n),
        "credit_history": np.random.choice(["critical", "all paid", "existing credits"], size=n),
        "credit_amount": np.random.randint(500, 10000, size=n),
        "savings_status": np.random.choice(["<100", ">=1000", "no known savings"], size=n),
        "employment": np.random.choice(["<1", "1<=X<4", ">=7"], size=n),
        "installment_commitment": np.random.choice([1, 2, 3, 4], size=n),
        "age": np.random.randint(20, 70, size=n),
        "housing": np.random.choice(["own", "rent", "for free"], size=n),
        "class": np.random.choice(["good", "bad"], size=n, p=[0.7, 0.3]),
    }
    return pd.DataFrame(data)


def test_load_default_german_credit():
    """Verify that the packaged German Credit CSV loads properly."""
    path = get_default_dataset_path()
    assert path.exists(), f"Default dataset file does not exist at {path}"
    df = load_dataset(path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 1000
    assert "class" in df.columns


def test_load_dataset_from_buffer():
    """Verify loading from a StringIO CSV buffer."""
    csv_str = "col1,col2,target\n10,a,good\n20,b,bad\n30,c,good\n"
    buffer = io.StringIO(csv_str)
    df = load_dataset(buffer)
    assert len(df) == 3
    assert list(df.columns) == ["col1", "col2", "target"]


def test_validate_dataset_valid(sample_credit_data):
    """Test validation with a structurally sound dataset."""
    val = validate_dataset(sample_credit_data, min_rows=20)
    assert val["is_valid"] is True
    assert val["error_message"] is None
    assert val["shape"] == (100, 10)
    assert len(val["columns"]) == 10
    assert "credit_amount" in val["numeric_columns"]
    assert "checking_status" in val["categorical_columns"]


def test_validate_dataset_empty_and_too_small():
    """Test validation error reporting on empty or insufficient datasets."""
    empty_df = pd.DataFrame()
    val_empty = validate_dataset(empty_df)
    assert val_empty["is_valid"] is False
    assert "0 rows" in val_empty["error_message"]

    small_df = pd.DataFrame({"a": [1, 2, 3], "target": ["good", "bad", "good"]})
    val_small = validate_dataset(small_df, min_rows=20)
    assert val_small["is_valid"] is False
    assert "minimum of 20 rows" in val_small["error_message"]


def test_detect_target_candidates(sample_credit_data):
    """Test automated candidate target discovery."""
    candidates = detect_target_candidates(sample_credit_data)
    assert "class" in candidates
    assert candidates[0] == "class"


def test_prepare_labeled_data_mapping(sample_credit_data):
    """Verify label encoding into binary 0 and 1."""
    X, y, meta = prepare_labeled_data(
        sample_credit_data,
        target_column="class",
        positive_value="good",
        negative_value="bad",
    )
    assert len(X) == len(sample_credit_data)
    assert set(y.unique()).issubset({0, 1})
    assert meta["positive_code"] == 1
    assert meta["negative_code"] == 0
    assert "class" not in X.columns


def test_prepare_labeled_data_single_class_raises_error():
    """Verify that a single-class dataset is rejected for binary classification."""
    single_class_df = pd.DataFrame({
        "credit_amount": [1000, 2000, 3000],
        "class": ["good", "good", "good"],
    })
    with pytest.raises(ValueError, match="at least 2 distinct target classes"):
        prepare_labeled_data(
            single_class_df,
            target_column="class",
            positive_value="good",
            negative_value="bad",
        )


def test_credit_feature_engineer(sample_credit_data):
    """Test domain financial ratio creation without leakage or index mutation."""
    fe = CreditFeatureEngineer(include_ratios=True)
    X = sample_credit_data.drop(columns=["class"])
    X_fe = fe.fit_transform(X)

    assert "fe_monthly_credit_burden" in X_fe.columns
    assert "fe_credit_duration_exposure" in X_fe.columns
    assert "fe_credit_to_age_ratio" in X_fe.columns
    # Check that values are finite and non-negative
    assert not X_fe["fe_monthly_credit_burden"].isna().any()
    assert (X_fe["fe_monthly_credit_burden"] >= 0).all()
    # Check original input shape is unaffected
    assert "fe_monthly_credit_burden" not in X.columns


def test_column_transformer_handles_missing_and_unseen_categories():
    """Verify that missing values are imputed and unseen categories in test don't crash."""
    train_df = pd.DataFrame({
        "num1": [10.0, 20.0, np.nan, 40.0],
        "cat1": ["A", "B", "A", None],
    })
    test_df = pd.DataFrame({
        "num1": [np.nan, 30.0],
        "cat1": ["UNSEEN_CATEGORY", "B"],  # Unseen category!
    })

    preprocessor = create_preprocessor(
        numeric_features=["num1"],
        categorical_features=["cat1"],
        scale_numeric=True,
    )

    # Fit ONLY on train
    X_train_trans = preprocessor.fit_transform(train_df)
    assert not np.isnan(X_train_trans).any()

    # Transform test without refitting (prevents data leakage)
    X_test_trans = preprocessor.transform(test_df)
    assert not np.isnan(X_test_trans).any()
    assert X_test_trans.shape[1] == X_train_trans.shape[1]


def test_split_data_reproducibility_and_stratification(sample_credit_data):
    """Verify reproducible stratified train/test splitting."""
    X, y, _ = prepare_labeled_data(sample_credit_data, target_column="class")
    X_train1, X_test1, y_train1, y_test1 = split_data(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_data(X, y, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    assert len(X_train1) == 80
    assert len(X_test1) == 20
    # Stratified balance check
    assert abs(y_train1.mean() - y.mean()) < 0.05


def test_data_leakage_prevention(sample_credit_data):
    """Strictly verify that test data distribution does NOT influence train transformation."""
    X, y, _ = prepare_labeled_data(sample_credit_data, target_column="class")
    X_train, X_test, _, _ = split_data(X, y, test_size=0.2, random_state=42)

    pipeline, _, _ = build_full_preprocessing_pipeline(X_train, scale_numeric=True)

    # Fit on train only
    X_train_transformed_1 = pipeline.fit_transform(X_train)

    # Now transform an artificially extreme test set (huge outliers)
    extreme_test = X_test.copy()
    if "credit_amount" in extreme_test.columns:
        extreme_test["credit_amount"] = 1_000_000.0  # extreme outlier

    # Transform extreme test
    _ = pipeline.transform(extreme_test)

    # Re-transform train and ensure it remains 100% identical
    X_train_transformed_2 = pipeline.transform(X_train)
    np.testing.assert_array_almost_equal(X_train_transformed_1, X_train_transformed_2)
