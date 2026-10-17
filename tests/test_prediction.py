"""Unit tests for inference, batch predictions, edge cases, explainability, and fairness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset, prepare_labeled_data, get_default_dataset_path
from src.preprocessing import split_data
from src.train import train_single_model
from src.predict import predict_credit_risk, batch_predict_credit_risk, validate_input_features
from src.evaluate import evaluate_model_performance, get_metric_educational_definitions
from src.explainability import (
    extract_global_feature_importance,
    audit_demographic_fairness,
)


@pytest.fixture(scope="module")
def trained_logistic_bundle():
    """Train a Logistic Regression model on German Credit data for inference testing."""
    df = load_dataset(get_default_dataset_path()).head(200)
    X, y, meta = prepare_labeled_data(df, target_column="class")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

    res = train_single_model("Logistic Regression", X_train, y_train, cv_folds=3, random_state=42)
    pipeline = res["pipeline"]
    return pipeline, X_train.columns.tolist(), meta, X_test, y_test, df


def test_single_applicant_prediction(trained_logistic_bundle):
    """Test single applicant prediction dictionary and probability outputs."""
    pipeline, expected_features, meta, X_test, _, _ = trained_logistic_bundle
    sample_record = X_test.iloc[0].to_dict()

    pred_res = predict_credit_risk(
        pipeline=pipeline,
        applicant_data=sample_record,
        expected_features=expected_features,
        decision_threshold=0.5,
        target_metadata=meta,
    )

    assert "predicted_code" in pred_res
    assert pred_res["predicted_code"] in [0, 1]
    assert "predicted_label" in pred_res
    assert pred_res["predicted_label"] in ["Good Credit Risk", "Poor Credit Risk"]
    assert pred_res["has_probabilities"] is True
    assert 0.0 <= pred_res["probability_good_credit"] <= 1.0
    assert 0.0 <= pred_res["probability_poor_credit"] <= 1.0
    assert abs(pred_res["probability_good_credit"] + pred_res["probability_poor_credit"] - 1.0) < 1e-4
    assert "disclaimer" in pred_res


def test_prediction_with_missing_inputs(trained_logistic_bundle):
    """Verify inference succeeds safely even when several applicant fields are missing."""
    pipeline, expected_features, meta, _, _, _ = trained_logistic_bundle
    # Provide sparse dictionary
    partial_applicant = {
        "credit_amount": 3500,
        "duration": 24,
    }

    pred_res = predict_credit_risk(
        pipeline=pipeline,
        applicant_data=partial_applicant,
        expected_features=expected_features,
        target_metadata=meta,
    )

    assert pred_res["predicted_code"] in [0, 1]
    assert pred_res["probability_good_credit"] is not None


def test_prediction_with_unseen_categories(trained_logistic_bundle):
    """Verify that completely novel categorical values do not trigger runtime crashes."""
    pipeline, expected_features, meta, X_test, _, _ = trained_logistic_bundle
    unseen_applicant = X_test.iloc[0].to_dict()
    unseen_applicant["checking_status"] = "completely_novel_unseen_bank_tier_XYZ"
    unseen_applicant["housing"] = "space_station_capsule"

    pred_res = predict_credit_risk(
        pipeline=pipeline,
        applicant_data=unseen_applicant,
        expected_features=expected_features,
        target_metadata=meta,
    )

    assert pred_res["predicted_code"] in [0, 1]


def test_batch_predict_credit_risk(trained_logistic_bundle):
    """Test batch prediction across multiple applicants."""
    pipeline, expected_features, meta, X_test, _, _ = trained_logistic_bundle
    batch = X_test.head(10)

    res_df = batch_predict_credit_risk(
        pipeline=pipeline,
        batch_df=batch,
        expected_features=expected_features,
        decision_threshold=0.5,
        target_metadata=meta,
    )

    assert len(res_df) == 10
    assert "Predicted_Risk_Code" in res_df.columns
    assert "Predicted_Credit_Risk" in res_df.columns
    assert "Prob_Good_Credit" in res_df.columns
    assert "Prob_Poor_Credit" in res_df.columns


def test_roc_auc_single_class_edge_case(trained_logistic_bundle):
    """Verify that evaluation handles subsets with only one target class gracefully without crashing."""
    pipeline, _, _, X_test, _, _ = trained_logistic_bundle
    # Construct artificial single-class target series (all 1s)
    single_class_y = pd.Series([1] * len(X_test), index=X_test.index)

    eval_res = evaluate_model_performance(pipeline, X_test, single_class_y)
    assert eval_res["roc_auc"] is None  # Handled gracefully, no crash!
    assert "accuracy" in eval_res


def test_educational_definitions_completeness():
    """Verify that all core metrics have educational definitions and financial context."""
    defs = get_metric_educational_definitions()
    for metric in ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "ROC-AUC", "Poor Credit Risk Recall"]:
        assert metric in defs
        assert "definition" in defs[metric]
        assert "financial_context" in defs[metric]


def test_feature_importance_extraction(trained_logistic_bundle):
    """Verify that feature importance extraction works for Logistic Regression."""
    pipeline, _, _, _, _, _ = trained_logistic_bundle
    imp = extract_global_feature_importance(pipeline, "Logistic Regression", top_n=10)
    assert imp["metric_type"] != "Unsupported"
    assert len(imp["features"]) > 0
    assert len(imp["scores"]) > 0


def test_fairness_audit(trained_logistic_bundle):
    """Verify that demographic fairness audit computes approval rates and disparate impact."""
    pipeline, _, _, _, _, full_df = trained_logistic_bundle
    audit = audit_demographic_fairness(
        pipeline=pipeline,
        df=full_df,
        sensitive_column="personal_status",
        target_column="class",
    )

    assert audit["is_evaluated"] is True
    assert len(audit["group_statistics"]) > 0
    assert 0.0 <= audit["disparate_impact_ratio"] <= 1.0
    assert "passes_four_fifths_rule" in audit
