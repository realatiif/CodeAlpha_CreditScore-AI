"""Unit tests for model training pipelines, cross-validation, tuning, and persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset, prepare_labeled_data, get_default_dataset_path
from src.preprocessing import split_data
from src.train import (
    MODEL_IDENTIFIERS,
    build_model_pipeline,
    train_single_model,
    train_all_models,
    select_best_model,
    save_trained_pipeline,
    load_trained_pipeline,
)
from src.evaluate import evaluate_model_performance


@pytest.fixture(scope="module")
def prepared_german_credit():
    """Load and prepare a slice of the real German Credit dataset for fast test execution."""
    df = load_dataset(get_default_dataset_path()).head(200)
    X, y, meta = prepare_labeled_data(df, target_column="class", positive_value="good", negative_value="bad")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, meta


def test_build_model_pipelines(prepared_german_credit):
    """Test that valid pipelines are constructed for all 3 algorithms."""
    X_train, _, _, _, _ = prepared_german_credit
    num_cols = ["duration", "credit_amount", "age"]
    cat_cols = ["checking_status", "housing"]

    for name in MODEL_IDENTIFIERS:
        pipe = build_model_pipeline(
            model_name=name,
            numeric_features=num_cols,
            categorical_features=cat_cols,
            class_weight="balanced",
            random_state=42,
        )
        assert pipe is not None
        assert "feature_engineer" in pipe.named_steps
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps


def test_train_single_model_without_tuning(prepared_german_credit):
    """Test training Logistic Regression without hyperparameter tuning."""
    X_train, X_test, y_train, y_test, _ = prepared_german_credit
    res = train_single_model(
        model_name="Logistic Regression",
        X_train=X_train,
        y_train=y_train,
        tune_hyperparameters=False,
        cv_folds=3,
        random_state=42,
    )

    assert res["model_name"] == "Logistic Regression"
    assert res["pipeline"] is not None
    assert res["tuning_performed"] is False
    assert len(res["cv_scores"]) == 3
    assert res["cv_mean_score"] > 0.0

    # Ensure pipeline can predict on test
    preds = res["pipeline"].predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})


def test_train_all_three_models(prepared_german_credit):
    """Test that all three models train successfully on consistent training splits."""
    X_train, X_test, y_train, y_test, _ = prepared_german_credit
    results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        tune_hyperparameters=False,
        cv_folds=3,
        random_state=42,
    )

    for name in MODEL_IDENTIFIERS:
        assert name in results
        assert results[name]["pipeline"] is not None
        assert results[name]["training_time_seconds"] >= 0.0


def test_model_selection_strategy():
    """Verify that model selection correctly picks best model according to priority metric."""
    mock_evals = {
        "Logistic Regression": {"f1_score": 0.78, "roc_auc": 0.75, "recall_poor_credit_risk": 0.65},
        "Decision Tree": {"f1_score": 0.72, "roc_auc": 0.68, "recall_poor_credit_risk": 0.70},
        "Random Forest": {"f1_score": 0.81, "roc_auc": 0.79, "recall_poor_credit_risk": 0.60},
    }

    # Best F1 should be Random Forest
    best_f1, exp_f1 = select_best_model(mock_evals, priority_metric="f1")
    assert best_f1 == "Random Forest"
    assert "F1-Score" in exp_f1

    # Best Poor Risk Recall should be Decision Tree
    best_rec, exp_rec = select_best_model(mock_evals, priority_metric="recall_poor_risk")
    assert best_rec == "Decision Tree"
    assert "Poor Credit Risk" in exp_rec


def test_save_and_load_trained_pipeline(prepared_german_credit):
    """Verify serialization and deserialization using joblib with identical predictions."""
    X_train, X_test, y_train, _, meta = prepared_german_credit
    train_res = train_single_model("Random Forest", X_train, y_train, cv_folds=3, random_state=42)
    pipeline = train_res["pipeline"]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.joblib"
        saved_file = save_trained_pipeline(
            pipeline=pipeline,
            model_name="Random Forest",
            feature_names_in=X_train.columns.tolist(),
            target_metadata=meta,
            filepath=save_path,
        )

        assert saved_file.exists()
        loaded_bundle = load_trained_pipeline(saved_file)

        assert loaded_bundle["model_name"] == "Random Forest"
        assert loaded_bundle["feature_names_in"] == X_train.columns.tolist()

        # Check identical predictions
        loaded_pipeline = loaded_bundle["pipeline"]
        orig_preds = pipeline.predict(X_test)
        loaded_preds = loaded_pipeline.predict(X_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

        orig_probas = pipeline.predict_proba(X_test)
        loaded_probas = loaded_pipeline.predict_proba(X_test)
        np.testing.assert_array_almost_equal(orig_probas, loaded_probas)
