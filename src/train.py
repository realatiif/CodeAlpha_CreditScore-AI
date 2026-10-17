"""Model training, hyperparameter optimization, and persistence module.

Implements pipelines for Logistic Regression, Decision Tree, and Random Forest
classifiers with class-imbalance weighting, stratified cross-validation on
training splits, metric-based model selection, and joblib persistence.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import CreditFeatureEngineer
from src.preprocessing import create_preprocessor, identify_column_types

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SAVED_MODEL_PATH = MODELS_DIR / "best_credit_scoring_pipeline.joblib"

MODEL_IDENTIFIERS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]


def build_model_pipeline(
    model_name: str,
    numeric_features: List[str],
    categorical_features: List[str],
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    custom_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a complete scikit-learn Pipeline with feature engineering, preprocessing, and classifier.

    Parameters
    ----------
    model_name : str
        Algorithm name ('Logistic Regression', 'Decision Tree', 'Random Forest').
    numeric_features : List[str]
        List of numeric column names (before feature engineering).
    categorical_features : List[str]
        List of categorical column names.
    class_weight : Optional[str], default='balanced'
        Class weighting strategy to handle class imbalance.
    random_state : int, default=42
        Reproducibility seed.
    custom_params : Optional[Dict[str, Any]], default=None
        Custom classifier hyperparameters.

    Returns
    -------
    Pipeline
        Unfitted full end-to-end scikit-learn Pipeline.

    Raises
    ------
    ValueError
        If model_name is unrecognized.
    """
    params = custom_params.copy() if custom_params else {}

    if model_name == "Logistic Regression":
        clf_params = {
            "class_weight": class_weight,
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "lbfgs",
        }
        clf_params.update(params)
        classifier = LogisticRegression(**clf_params)
        scale_numeric = True

    elif model_name == "Decision Tree":
        clf_params = {
            "class_weight": class_weight,
            "max_depth": 5,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": random_state,
        }
        clf_params.update(params)
        classifier = DecisionTreeClassifier(**clf_params)
        scale_numeric = False

    elif model_name == "Random Forest":
        clf_params = {
            "class_weight": class_weight,
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf_params.update(params)
        classifier = RandomForestClassifier(**clf_params)
        scale_numeric = False

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Supported options are: {MODEL_IDENTIFIERS}"
        )

    # Note: Feature engineer adds domain columns if source columns exist.
    # We dynamically inspect what columns are created to configure the ColumnTransformer.
    dummy_df = pd.DataFrame(
        {col: [0.0] for col in numeric_features}
        | {col: ["unknown"] for col in categorical_features}
    )
    fe = CreditFeatureEngineer(include_ratios=True)
    dummy_transformed = fe.fit_transform(dummy_df)
    final_num_cols, final_cat_cols = identify_column_types(dummy_transformed)

    preprocessor = create_preprocessor(
        numeric_features=final_num_cols,
        categorical_features=final_cat_cols,
        scale_numeric=scale_numeric,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", fe),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def get_hyperparameter_search_grid(model_name: str) -> Dict[str, List[Any]]:
    """Return GridSearchCV parameter grid for the specified model.

    Parameters
    ----------
    model_name : str
        Algorithm name.

    Returns
    -------
    Dict[str, List[Any]]
        Scikit-learn parameter grid mapping.
    """
    if model_name == "Logistic Regression":
        return {
            "classifier__C": [0.05, 0.2, 1.0, 5.0],
            "classifier__solver": ["lbfgs"],
        }
    elif model_name == "Decision Tree":
        return {
            "classifier__max_depth": [3, 5, 8],
            "classifier__min_samples_split": [5, 10, 20],
            "classifier__min_samples_leaf": [2, 5, 10],
        }
    elif model_name == "Random Forest":
        return {
            "classifier__n_estimators": [50, 100, 150],
            "classifier__max_depth": [5, 8, 12],
            "classifier__min_samples_leaf": [2, 4],
        }
    return {}


def train_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = False,
    tuning_scoring: str = "roc_auc",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a single model pipeline with optional stratified hyperparameter tuning.

    Parameters
    ----------
    model_name : str
        Algorithm name.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training binary target.
    tune_hyperparameters : bool, default=False
        Whether to perform GridSearchCV.
    tuning_scoring : str, default='roc_auc'
        Metric to optimize in CV ('roc_auc', 'f1', 'accuracy', 'recall').
    cv_folds : int, default=5
        Number of stratified CV splits on training data.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with fitted pipeline, best parameters, CV score, and timing.
    """
    start_time = datetime.datetime.now()
    num_cols, cat_cols = identify_column_types(X_train)

    base_pipeline = build_model_pipeline(
        model_name=model_name,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        class_weight="balanced",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    best_params = {}
    tuning_performed = False

    if tune_hyperparameters:
        param_grid = get_hyperparameter_search_grid(model_name)
        if param_grid:
            grid_search = GridSearchCV(
                estimator=base_pipeline,
                param_grid=param_grid,
                scoring=tuning_scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            fitted_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            tuning_performed = True
        else:
            fitted_pipeline = base_pipeline.fit(X_train, y_train)
    else:
        fitted_pipeline = base_pipeline.fit(X_train, y_train)

    # Perform stratified CV on training data to record CV score distribution
    try:
        cv_scores = cross_val_score(
            fitted_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=tuning_scoring if tuning_scoring != "roc_auc" else "roc_auc",
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_scores = np.array([])
        cv_mean = 0.0
        cv_std = 0.0

    duration = (datetime.datetime.now() - start_time).total_seconds()

    return {
        "model_name": model_name,
        "pipeline": fitted_pipeline,
        "best_params": best_params,
        "tuning_performed": tuning_performed,
        "cv_metric": tuning_scoring,
        "cv_mean_score": cv_mean,
        "cv_std_score": cv_std,
        "cv_scores": cv_scores.tolist(),
        "training_time_seconds": round(duration, 3),
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = False,
    tuning_scoring: str = "roc_auc",
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Train and cross-validate all 3 target classifiers on consistent training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training binary target.
    tune_hyperparameters : bool, default=False
        Whether to run grid search tuning.
    tuning_scoring : str, default='roc_auc'
        Optimization metric for CV.
    cv_folds : int, default=5
        CV folds.
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary keyed by model identifier containing training outcomes.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name in MODEL_IDENTIFIERS:
        results[name] = train_single_model(
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            tune_hyperparameters=tune_hyperparameters,
            tuning_scoring=tuning_scoring,
            cv_folds=cv_folds,
            random_state=random_state,
        )
    return results


def select_best_model(
    model_evaluations: Dict[str, Dict[str, Any]],
    priority_metric: str = "f1",
) -> Tuple[str, str]:
    """Select recommended model based on chosen priority metric with justification.

    Parameters
    ----------
    model_evaluations : Dict[str, Dict[str, Any]]
        Dictionary containing test evaluation metrics for each model.
    priority_metric : str, default='f1'
        Priority metric: 'f1', 'roc_auc', 'recall_poor_risk', 'accuracy', 'precision'.

    Returns
    -------
    Tuple[str, str]
        (recommended_model_name, justification_explanation)
    """
    metric_keys_map = {
        "f1": "f1_score",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall_poor_risk": "recall_poor_credit_risk",
    }

    target_key = metric_keys_map.get(priority_metric, "f1_score")

    best_model = None
    best_score = -1.0

    for name, eval_dict in model_evaluations.items():
        score = eval_dict.get(target_key, 0.0)
        if score is None:
            score = 0.0
        if score > best_score:
            best_score = score
            best_model = name

    if best_model is None:
        best_model = MODEL_IDENTIFIERS[0]
        best_score = 0.0

    explanation = (
        f"Selected **{best_model}** because it achieved the highest score on the chosen priority metric "
        f"(`{priority_metric}`: {best_score:.4f}). "
    )

    if priority_metric == "recall_poor_risk":
        explanation += (
            "Maximizing recall on the Poor Credit Risk class directly mitigates catastrophic loan defaults "
            "by aggressively identifying applicants likely to experience repayment distress."
        )
    elif priority_metric == "roc_auc":
        explanation += (
            "ROC-AUC evaluates discriminative capability across all potential decision thresholds, "
            "providing robust separation between creditworthy applicants and potential default risks."
        )
    elif priority_metric == "f1":
        explanation += (
            "The F1-Score balances precision and recall, ensuring the model avoids extreme bias "
            "between falsely rejecting creditworthy applicants and under-detecting credit defaults."
        )
    else:
        explanation += (
            f"Evaluated across balanced test benchmarks using {target_key}."
        )

    return best_model, explanation


def save_trained_pipeline(
    pipeline: Pipeline,
    model_name: str,
    feature_names_in: List[str],
    target_metadata: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
    filepath: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize the trained pipeline and comprehensive metadata to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline.
    model_name : str
        Algorithm name.
    feature_names_in : List[str]
        Expected input column names.
    target_metadata : Dict[str, Any]
        Label mapping information and class codes.
    eval_metrics : Optional[Dict[str, Any]], default=None
        Evaluated performance metrics on test split.
    filepath : Optional[Union[str, Path]], default=None
        Destination file path. If None, saves to default models directory.

    Returns
    -------
    Path
        Absolute path to saved artifact.
    """
    dest = Path(filepath) if filepath else DEFAULT_SAVED_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "model_name": model_name,
        "feature_names_in": feature_names_in,
        "target_metadata": target_metadata,
        "eval_metrics": eval_metrics or {},
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "threshold": 0.5,
    }

    joblib.dump(artifact, dest)
    return dest


def load_trained_pipeline(filepath: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load serialized pipeline bundle and associated metadata.

    Parameters
    ----------
    filepath : Optional[Union[str, Path]], default=None
        Path to joblib file. Defaults to default saved model path.

    Returns
    -------
    Dict[str, Any]
        Loaded bundle dictionary.

    Raises
    ------
    FileNotFoundError
        If model artifact does not exist.
    """
    target_path = Path(filepath) if filepath else DEFAULT_SAVED_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at: {target_path}")

    artifact = joblib.load(target_path)
    return artifact
