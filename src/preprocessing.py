"""Data preprocessing and transformation pipeline module.

Constructs scikit-learn ColumnTransformer preprocessors that fit strictly
on training data to prevent data leakage, handling missing value imputation,
standard scaling, and one-hot encoding for categorical variables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import CreditFeatureEngineer


def identify_column_types(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """Separate dataframe columns into numerical and categorical types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_column : Optional[str], default=None
        Target column name to exclude if present.

    Returns
    -------
    Tuple[List[str], List[str]]
        (numerical_columns, categorical_columns)
    """
    candidates = [c for c in df.columns if c != target_column]
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True
) -> ColumnTransformer:
    """Build a ColumnTransformer preprocessor for numeric and categorical features.

    Parameters
    ----------
    numeric_features : List[str]
        List of numeric column names.
    categorical_features : List[str]
        List of categorical column names.
    scale_numeric : bool, default=True
        Whether to standardize numeric features using StandardScaler.

    Returns
    -------
    ColumnTransformer
        Assembled scikit-learn ColumnTransformer.
    """
    # Numeric pipeline: median imputation + optional standardization
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    # Categorical pipeline: most frequent imputation + one-hot encoding with ignore unknown
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_preprocessing_pipeline(
    X_train: pd.DataFrame,
    include_feature_engineering: bool = True,
    scale_numeric: bool = True
) -> Tuple[Pipeline, List[str], List[str]]:
    """Assemble the complete feature engineering and preprocessing pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features used to determine base and engineered column types.
    include_feature_engineering : bool, default=True
        Whether to include domain financial ratio extraction.
    scale_numeric : bool, default=True
        Whether to apply standard scaling to numeric attributes.

    Returns
    -------
    Tuple[Pipeline, List[str], List[str]]
        - Full preprocessing pipeline (FeatureEngineer + ColumnTransformer).
        - Final list of numeric feature names.
        - Final list of categorical feature names.
    """
    steps = []

    # 1. Feature Engineering step
    fe = CreditFeatureEngineer(include_ratios=include_feature_engineering)
    steps.append(("feature_engineer", fe))

    # Temporarily transform a copy of X_train to discover final column names
    X_sample = fe.fit_transform(X_train)
    num_cols, cat_cols = identify_column_types(X_sample)

    # 2. Preprocessor step
    preprocessor = create_preprocessor(
        numeric_features=num_cols,
        categorical_features=cat_cols,
        scale_numeric=scale_numeric,
    )
    steps.append(("preprocessor", preprocessor))

    full_pipeline = Pipeline(steps=steps)
    return full_pipeline, num_cols, cat_cols


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform reproducible train/test split with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Binary target.
    test_size : float, default=0.20
        Proportion of dataset for testing.
    random_state : int, default=42
        Seed for reproducibility.
    stratify : bool, default=True
        Whether to preserve class distribution across splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    stratify_target = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def compute_class_imbalance_metrics(y: pd.Series) -> Dict[str, Any]:
    """Compute detailed class balance metrics, ratios, and recommended class weights.

    Parameters
    ----------
    y : pd.Series
        Binary target series (values 0 and 1).

    Returns
    -------
    Dict[str, Any]
        Class counts, proportions, imbalance ratio, and suggested weights.
    """
    total = len(y)
    count_pos = int((y == 1).sum())
    count_neg = int((y == 0).sum())

    pct_pos = float(count_pos / total) if total > 0 else 0.0
    pct_neg = float(count_neg / total) if total > 0 else 0.0

    imbalance_ratio = (
        float(max(count_pos, count_neg) / max(min(count_pos, count_neg), 1))
    )

    # Balanced class weights: total / (2 * count)
    weight_0 = float(total / (2 * max(count_neg, 1)))
    weight_1 = float(total / (2 * max(count_pos, 1)))

    return {
        "total_instances": total,
        "positive_count": count_pos,
        "negative_count": count_neg,
        "positive_percentage": pct_pos * 100.0,
        "negative_percentage": pct_neg * 100.0,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "is_imbalanced": imbalance_ratio >= 1.5,
        "recommended_class_weights": {0: weight_0, 1: weight_1},
    }
