"""Inference and credit risk prediction module.

Performs schema validation, single-applicant and batch predictions, probability
estimation, and sanity checking using trained scikit-learn pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def validate_input_features(
    input_df: pd.DataFrame,
    expected_features: List[str],
) -> Tuple[bool, Optional[str], pd.DataFrame]:
    """Validate that input data contains expected columns and handle missing fields.

    Parameters
    ----------
    input_df : pd.DataFrame
        Applicant features to validate.
    expected_features : List[str]
        List of column names required by the trained model.

    Returns
    -------
    Tuple[bool, Optional[str], pd.DataFrame]
        (is_valid, error_message, aligned_dataframe)
    """
    if input_df is None or input_df.empty:
        return False, "Input data is empty. Please supply applicant attributes.", pd.DataFrame()

    aligned = input_df.copy()

    # Check for missing expected columns; fill with NaN so imputer handles them gracefully
    missing_cols = [c for c in expected_features if c not in aligned.columns]
    for col in missing_cols:
        aligned[col] = np.nan

    # Retain strictly the expected features in identical column order
    aligned = aligned[expected_features]

    return True, None, aligned


def predict_credit_risk(
    pipeline: Pipeline,
    applicant_data: Union[pd.DataFrame, Dict[str, Any]],
    expected_features: List[str],
    decision_threshold: float = 0.5,
    target_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate credit risk prediction and estimated probabilities for an applicant.

    Parameters
    ----------
    pipeline : Pipeline
        Trained scikit-learn pipeline.
    applicant_data : Union[pd.DataFrame, Dict[str, Any]]
        Input applicant record or DataFrame.
    expected_features : List[str]
        Feature columns expected by the pipeline.
    decision_threshold : float, default=0.5
        Threshold above which positive class (Good Credit Risk) is predicted.
    target_metadata : Optional[Dict[str, Any]], default=None
        Mapping metadata for class names.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predicted class, label, probabilities, and confidence.

    Raises
    ------
    ValueError
        If applicant data fails validation.
    """
    if isinstance(applicant_data, dict):
        df_input = pd.DataFrame([applicant_data])
    else:
        df_input = applicant_data.copy()

    is_valid, err, aligned = validate_input_features(df_input, expected_features)
    if not is_valid:
        raise ValueError(f"Input validation error: {err}")

    # Probability estimation
    has_proba = hasattr(pipeline, "predict_proba")
    prob_good = None
    prob_poor = None

    if has_proba:
        try:
            probas = pipeline.predict_proba(aligned)
            if probas.shape[1] == 2:
                # Binary classification [prob_class_0, prob_class_1]
                prob_poor = float(probas[0, 0])
                prob_good = float(probas[0, 1])
            elif probas.shape[1] == 1:
                prob_good = float(probas[0, 0])
                prob_poor = 1.0 - prob_good
        except Exception:
            has_proba = False

    # Apply decision threshold if probabilities are available
    if prob_good is not None:
        predicted_code = 1 if prob_good >= decision_threshold else 0
    else:
        predicted_code = int(pipeline.predict(aligned)[0])

    pos_name = (
        target_metadata.get("positive_label_name", "Good Credit Risk")
        if target_metadata
        else "Good Credit Risk"
    )
    neg_name = (
        target_metadata.get("negative_label_name", "Poor Credit Risk")
        if target_metadata
        else "Poor Credit Risk"
    )

    predicted_label = pos_name if predicted_code == 1 else neg_name
    confidence_score = (
        (prob_good if predicted_code == 1 else prob_poor)
        if prob_good is not None
        else 1.0
    )

    return {
        "predicted_code": predicted_code,
        "predicted_label": predicted_label,
        "is_creditworthy": predicted_code == 1,
        "has_probabilities": has_proba and (prob_good is not None),
        "probability_good_credit": prob_good,
        "probability_poor_credit": prob_poor,
        "confidence_score": confidence_score,
        "decision_threshold_used": decision_threshold,
        "disclaimer": (
            "Model estimate based on statistical historical data. "
            "Does not constitute a legal, financial, or guaranteed credit underwriting commitment."
        ),
    }


def batch_predict_credit_risk(
    pipeline: Pipeline,
    batch_df: pd.DataFrame,
    expected_features: List[str],
    decision_threshold: float = 0.5,
    target_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Run credit risk inference across a batch dataframe of applicants.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline.
    batch_df : pd.DataFrame
        Applicant records.
    expected_features : List[str]
        Expected feature names.
    decision_threshold : float, default=0.5
        Probability threshold.
    target_metadata : Optional[Dict[str, Any]], default=None
        Label metadata.

    Returns
    -------
    pd.DataFrame
        Input dataframe with prediction columns appended:
        'Predicted_Risk_Code', 'Predicted_Credit_Risk', 'Prob_Good_Credit', 'Prob_Poor_Credit'.
    """
    is_valid, err, aligned = validate_input_features(batch_df, expected_features)
    if not is_valid:
        raise ValueError(f"Batch validation failure: {err}")

    results_df = batch_df.copy()
    has_proba = hasattr(pipeline, "predict_proba")

    pos_name = (
        target_metadata.get("positive_label_name", "Good Credit Risk")
        if target_metadata
        else "Good Credit Risk"
    )
    neg_name = (
        target_metadata.get("negative_label_name", "Poor Credit Risk")
        if target_metadata
        else "Poor Credit Risk"
    )

    if has_proba:
        try:
            probas = pipeline.predict_proba(aligned)
            if probas.shape[1] == 2:
                results_df["Prob_Poor_Credit"] = probas[:, 0]
                results_df["Prob_Good_Credit"] = probas[:, 1]
                preds = (probas[:, 1] >= decision_threshold).astype(int)
            else:
                results_df["Prob_Good_Credit"] = probas[:, 0]
                results_df["Prob_Poor_Credit"] = 1.0 - probas[:, 0]
                preds = (probas[:, 0] >= decision_threshold).astype(int)
        except Exception:
            preds = pipeline.predict(aligned)
            results_df["Prob_Good_Credit"] = np.nan
            results_df["Prob_Poor_Credit"] = np.nan
    else:
        preds = pipeline.predict(aligned)
        results_df["Prob_Good_Credit"] = np.nan
        results_df["Prob_Poor_Credit"] = np.nan

    results_df["Predicted_Risk_Code"] = preds
    results_df["Predicted_Credit_Risk"] = [
        pos_name if p == 1 else neg_name for p in preds
    ]

    return results_df
