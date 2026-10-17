"""Model explainability, feature importance, and responsible AI fairness module.

Extracts model coefficients and Gini importance metrics, generates interactive
Plotly visualizations, and provides ethical lending fairness audits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline


def get_preprocessed_feature_names(pipeline: Pipeline) -> List[str]:
    """Extract post-transformation feature names from the fitted preprocessor.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline containing a 'preprocessor' step.

    Returns
    -------
    List[str]
        List of generated column names after encoding and scaling.
    """
    if "preprocessor" not in pipeline.named_steps:
        return []

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    # Fallback inspection of sub-transformers
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                names.extend(trans.get_feature_names_out(cols))
                continue
            except Exception:
                pass
        if hasattr(trans, "named_steps"):
            # Check last step of sub-pipeline
            last_step = list(trans.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    names.extend(last_step.get_feature_names_out(cols))
                    continue
                except Exception:
                    pass
        names.extend(cols)

    return names


def extract_global_feature_importance(
    pipeline: Pipeline,
    model_name: str,
    top_n: int = 15,
) -> Dict[str, Any]:
    """Extract and rank the most influential features for the given model.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.
    model_name : str
        Algorithm name ('Logistic Regression', 'Decision Tree', 'Random Forest').
    top_n : int, default=15
        Number of top features to return.

    Returns
    -------
    Dict[str, Any]
        Dictionary with feature names, importance scores, metric type, and rankings.
    """
    if "classifier" not in pipeline.named_steps:
        return {"features": [], "scores": [], "metric_type": "N/A", "top_df": pd.DataFrame()}

    classifier = pipeline.named_steps["classifier"]
    feature_names = get_preprocessed_feature_names(pipeline)

    if model_name == "Logistic Regression" and hasattr(classifier, "coef_"):
        # Coefficients
        coefs = classifier.coef_[0]
        # Match lengths
        min_len = min(len(feature_names), len(coefs))
        names = feature_names[:min_len] if feature_names else [f"feature_{i}" for i in range(min_len)]
        scores = coefs[:min_len]

        df = pd.DataFrame({"feature": names, "score": scores, "abs_score": np.abs(scores)})
        df = df.sort_values(by="abs_score", ascending=False).head(top_n)

        return {
            "metric_type": "Logistic Regression Coefficients (log-odds impact)",
            "interpretation": (
                "Positive values increase likelihood of predicting 'Good Credit Risk'; "
                "Negative values increase likelihood of predicting 'Poor Credit Risk'."
            ),
            "features": df["feature"].tolist(),
            "scores": df["score"].tolist(),
            "top_df": df,
        }

    elif hasattr(classifier, "feature_importances_"):
        # Tree-based Gini / Mean Decrease in Impurity
        importances = classifier.feature_importances_
        min_len = min(len(feature_names), len(importances))
        names = feature_names[:min_len] if feature_names else [f"feature_{i}" for i in range(min_len)]
        scores = importances[:min_len]

        df = pd.DataFrame({"feature": names, "score": scores})
        df = df.sort_values(by="score", ascending=False).head(top_n)

        return {
            "metric_type": "Mean Decrease in Impurity (Gini Importance)",
            "interpretation": (
                "Higher values indicate features that create the most significant purity gains "
                "and splits in decision trees across the ensemble."
            ),
            "features": df["feature"].tolist(),
            "scores": df["score"].tolist(),
            "top_df": df,
        }

    return {
        "metric_type": "Unsupported",
        "interpretation": "Feature importance extraction not supported for this classifier architecture.",
        "features": [],
        "scores": [],
        "top_df": pd.DataFrame(),
    }


def create_feature_importance_figure(
    importance_dict: Dict[str, Any],
    title: str = "Key Financial Risk Drivers"
) -> go.Figure:
    """Generate an interactive Plotly horizontal bar chart of top feature drivers.

    Parameters
    ----------
    importance_dict : Dict[str, Any]
        Output of extract_global_feature_importance.
    title : str, default='Key Financial Risk Drivers'

    Returns
    -------
    go.Figure
        Plotly Figure.
    """
    df = importance_dict.get("top_df", pd.DataFrame())
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature importance metrics available for this model configuration.",
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        return fig

    # Sort ascending for horizontal bar chart (highest at top)
    sorted_df = df.sort_values(by="score", ascending=True)

    metric_type = importance_dict.get("metric_type", "")
    is_signed = "Coefficients" in metric_type

    if is_signed:
        colors = ["#10b981" if s >= 0 else "#ef4444" for s in sorted_df["score"]]
    else:
        colors = ["#0284c7"] * len(sorted_df)

    fig = go.Figure(
        go.Bar(
            x=sorted_df["score"],
            y=sorted_df["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{s:+.3f}" if is_signed else f"{s:.3f}" for s in sorted_df["score"]],
            textposition="auto",
        )
    )

    fig.update_layout(
        title={"text": f"<b>{title}</b><br><sub>{metric_type}</sub>", "x": 0.5},
        xaxis_title="<b>Impact Score</b>",
        yaxis_title="<b>Feature / Characteristic</b>",
        margin=dict(l=150, r=40, t=70, b=40),
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Roboto, sans-serif", size=11),
    )

    return fig


def audit_demographic_fairness(
    pipeline: Pipeline,
    df: pd.DataFrame,
    sensitive_column: str,
    target_column: str = "class",
    favorable_class_code: int = 1,
) -> Dict[str, Any]:
    """Conduct fairness audit on sensitive demographic attributes (e.g. sex, age bracket).

    Calculates demographic parity ratio and favorable outcome approval rate by group.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline.
    df : pd.DataFrame
        Dataset containing sensitive attribute and features.
    sensitive_column : str
        Column name for sensitive attribute (e.g., 'personal_status' or 'age').
    target_column : str, default='class'
        Target column name to exclude from inference.
    favorable_class_code : int, default=1
        Binary code representing favorable outcome (Good Credit Risk = 1).

    Returns
    -------
    Dict[str, Any]
        Audit summary containing approval rates, disparate impact ratio, and warning flags.
    """
    if sensitive_column not in df.columns:
        return {
            "is_evaluated": False,
            "reason": f"Column '{sensitive_column}' not found in dataset.",
        }

    X = df.drop(columns=[target_column], errors="ignore")
    preds = pipeline.predict(X)

    eval_df = pd.DataFrame({
        "group": df[sensitive_column].astype(str),
        "prediction": preds,
    })

    group_stats = []
    approval_rates = {}

    for group_name, sub in eval_df.groupby("group"):
        n_total = len(sub)
        n_approved = int((sub["prediction"] == favorable_class_code).sum())
        rate = n_approved / max(n_total, 1)
        approval_rates[group_name] = rate
        group_stats.append({
            "group": group_name,
            "total_applicants": n_total,
            "approved_as_good_risk": n_approved,
            "approval_rate": round(rate * 100.0, 1),
        })

    # Disparate impact ratio: min_rate / max_rate
    rates = list(approval_rates.values())
    min_rate = min(rates) if rates else 0.0
    max_rate = max(rates) if rates else 1.0
    disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 1.0

    # 80% rule (EEOC standard rule of thumb)
    passes_four_fifths_rule = disparate_impact_ratio >= 0.80

    return {
        "is_evaluated": True,
        "sensitive_column": sensitive_column,
        "group_statistics": group_stats,
        "disparate_impact_ratio": round(disparate_impact_ratio, 3),
        "passes_four_fifths_rule": passes_four_fifths_rule,
        "disclaimer": (
            "Statistical association within historical training datasets does not imply "
            "inherent creditworthiness differentials. Fair lending regulations mandate "
            "removing or controlling for legally protected demographic characteristics."
        ),
    }
