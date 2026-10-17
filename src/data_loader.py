"""Data loader and dataset validation module for CreditScore AI.

Handles loading, validating, inspecting, and target label mapping for credit
risk datasets, including local defaults and user-uploaded CSV files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "german_credit.csv"

# Canonical target names
POSITIVE_CLASS_NAME = "Good Credit Risk"   # Encoded as 1 (Creditworthy / Low risk)
NEGATIVE_CLASS_NAME = "Poor Credit Risk"   # Encoded as 0 (High risk / Default)


def get_default_dataset_path() -> Path:
    """Return the absolute path to the local default German Credit dataset."""
    return DEFAULT_DATASET_PATH


def load_dataset(
    source: Union[str, Path, io.BytesIO, io.StringIO, pd.DataFrame]
) -> pd.DataFrame:
    """Load a dataset from file path, buffer, or existing DataFrame.

    Parameters
    ----------
    source : Union[str, Path, io.BytesIO, io.StringIO, pd.DataFrame]
        CSV file path, file buffer from upload, or pre-loaded DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If file path does not exist.
    ValueError
        If CSV format is corrupt or unparseable.
    """
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found at: {p.resolve()}")
        try:
            return pd.read_csv(p)
        except Exception as e:
            raise ValueError(f"Could not parse CSV file at {p}: {str(e)}") from e

    # Assume file-like buffer (e.g. Streamlit UploadedFile)
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source)
    except Exception as e:
        raise ValueError(f"Failed to read uploaded file buffer: {str(e)}") from e


def validate_dataset(df: pd.DataFrame, min_rows: int = 20) -> Dict[str, Any]:
    """Validate dataset structure, schema integrity, and basic statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to inspect.
    min_rows : int, optional
        Minimum number of instances required, by default 20.

    Returns
    -------
    Dict[str, Any]
        Dictionary with validation results:
        - is_valid (bool)
        - error_message (Optional[str])
        - shape (Tuple[int, int])
        - columns (List[str])
        - missing_summary (Dict[str, int])
        - total_missing (int)
        - duplicate_rows (int)
        - dtypes (Dict[str, str])
        - numeric_columns (List[str])
        - categorical_columns (List[str])
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "error_message": "Supplied input is not a valid pandas DataFrame.",
            "shape": (0, 0),
            "columns": [],
            "missing_summary": {},
            "total_missing": 0,
            "duplicate_rows": 0,
            "dtypes": {},
            "numeric_columns": [],
            "categorical_columns": [],
        }

    n_rows, n_cols = df.shape

    if n_rows == 0:
        return {
            "is_valid": False,
            "error_message": "Dataset contains 0 rows. Please provide a non-empty CSV.",
            "shape": (0, n_cols),
            "columns": df.columns.tolist(),
            "missing_summary": {},
            "total_missing": 0,
            "duplicate_rows": 0,
            "dtypes": {},
            "numeric_columns": [],
            "categorical_columns": [],
        }

    if n_rows < min_rows:
        return {
            "is_valid": False,
            "error_message": (
                f"Dataset contains only {n_rows} rows. A minimum of {min_rows} rows "
                "is required for statistical reliability and stratified training."
            ),
            "shape": (n_rows, n_cols),
            "columns": df.columns.tolist(),
            "missing_summary": {},
            "total_missing": 0,
            "duplicate_rows": 0,
            "dtypes": {},
            "numeric_columns": [],
            "categorical_columns": [],
        }

    if n_cols < 2:
        return {
            "is_valid": False,
            "error_message": "Dataset must contain at least 2 columns (at least 1 feature and 1 target).",
            "shape": (n_rows, n_cols),
            "columns": df.columns.tolist(),
            "missing_summary": {},
            "total_missing": 0,
            "duplicate_rows": 0,
            "dtypes": {},
            "numeric_columns": [],
            "categorical_columns": [],
        }

    missing_series = df.isnull().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_series.items() if cnt > 0}
    total_missing = int(missing_series.sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    return {
        "is_valid": True,
        "error_message": None,
        "shape": (n_rows, n_cols),
        "columns": df.columns.tolist(),
        "missing_summary": missing_dict,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """Suggest likely target column candidates based on naming and cardinality.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.

    Returns
    -------
    List[str]
        Prioritized list of suggested target column names.
    """
    candidates: List[str] = []
    preferred_names = [
        "class",
        "target",
        "risk",
        "credit_risk",
        "default",
        "loan_status",
        "creditworthiness",
        "bad_credit",
        "status",
    ]

    # Exact or lowercase match
    lower_map = {c.lower(): c for c in df.columns}
    for pref in preferred_names:
        if pref in lower_map and lower_map[pref] not in candidates:
            candidates.append(lower_map[pref])

    # Check remaining columns with cardinality 2
    for col in df.columns:
        if col not in candidates:
            uniques = df[col].dropna().unique()
            if len(uniques) == 2:
                candidates.append(col)

    # If still none, add the last column by default
    if not candidates and len(df.columns) > 0:
        candidates.append(df.columns[-1])

    return candidates


def inspect_target_distribution(
    df: pd.DataFrame, target_column: str
) -> Dict[str, Any]:
    """Inspect the frequency and percentage distribution of a target column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the target column.
    target_column : str
        Target column name.

    Returns
    -------
    Dict[str, Any]
        Detailed summary of unique values, counts, and percentages.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' does not exist in dataset.")

    series = df[target_column].dropna()
    counts = series.value_counts().to_dict()
    total = len(series)
    proportions = {k: float(v / total) for k, v in counts.items()}
    unique_vals = list(counts.keys())

    return {
        "target_column": target_column,
        "unique_values": unique_vals,
        "class_count": len(unique_vals),
        "counts": counts,
        "proportions": proportions,
        "total_valid_rows": total,
        "missing_count": int(df[target_column].isnull().sum()),
    }


def prepare_labeled_data(
    df: pd.DataFrame,
    target_column: str,
    positive_value: Any = "good",
    negative_value: Any = "bad",
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Split dataset into features X and binary target y, applying explicit label mapping.

    Positive class is encoded as 1 (Good Credit Risk / Creditworthy).
    Negative class is encoded as 0 (Poor Credit Risk / High Risk).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str
        Name of the target column.
    positive_value : Any, optional
        Value representing Good Credit Risk, by default "good".
    negative_value : Any, optional
        Value representing Poor Credit Risk, by default "bad".

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]
        - X (pd.DataFrame): Features without target.
        - y (pd.Series): Binary target encoded as 0 and 1.
        - metadata (Dict[str, Any]): Label mapping details.

    Raises
    ------
    ValueError
        If target column has fewer than 2 unique classes (e.g. single-class)
        or if values do not match specified positive/negative definitions.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_column]).copy()
    unique_vals = clean_df[target_column].unique()

    if len(unique_vals) < 2:
        raise ValueError(
            f"Binary classification requires at least 2 distinct target classes, "
            f"but target '{target_column}' contains only {len(unique_vals)} unique value(s): {unique_vals.tolist()}. "
            "Please upload or select a dataset with both credit risk outcomes."
        )

    # Cast positive and negative values to match types in column if necessary
    # Check if target values can match strings or numbers
    target_sample = unique_vals[0]
    pos_val = positive_value
    neg_val = negative_value

    if isinstance(target_sample, (int, np.integer)) and not isinstance(positive_value, (int, np.integer)):
        try:
            pos_val = int(positive_value)
            neg_val = int(negative_value)
        except (ValueError, TypeError):
            pass

    # Build mapping
    # 1: Good Credit Risk, 0: Poor Credit Risk
    y_raw = clean_df[target_column]

    # Check membership
    matched_pos = y_raw == pos_val
    matched_neg = y_raw == neg_val

    # If not matched directly, check case-insensitive string matching
    if not matched_pos.any() or not matched_neg.any():
        pos_str = str(positive_value).strip().lower()
        neg_str = str(negative_value).strip().lower()
        y_str = y_raw.astype(str).str.strip().str.lower()
        matched_pos = y_str == pos_str
        matched_neg = y_str == neg_str

    if not matched_pos.any() or not matched_neg.any():
        raise ValueError(
            f"Could not map target column values. "
            f"Target has unique values: {unique_vals.tolist()}, "
            f"but mapping specified positive={positive_value} and negative={negative_value}."
        )

    y = pd.Series(np.nan, index=clean_df.index, dtype=float)
    y[matched_pos] = 1.0
    y[matched_neg] = 0.0

    # Filter out any unmapped instances
    valid_mask = ~y.isna()
    y = y[valid_mask].astype(int)
    X = clean_df.loc[valid_mask].drop(columns=[target_column])

    metadata = {
        "target_column": target_column,
        "positive_raw_value": positive_value,
        "negative_raw_value": negative_value,
        "positive_label_name": POSITIVE_CLASS_NAME,
        "negative_label_name": NEGATIVE_CLASS_NAME,
        "positive_code": 1,
        "negative_code": 0,
        "mapping_description": (
            f"'{positive_value}' -> 1 ({POSITIVE_CLASS_NAME}), "
            f"'{negative_value}' -> 0 ({NEGATIVE_CLASS_NAME})"
        ),
        "total_instances": len(y),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "positive_ratio": float((y == 1).mean()),
        "negative_ratio": float((y == 0).mean()),
    }

    return X, y, metadata
