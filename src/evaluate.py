"""Model evaluation, metrics computation, and financial risk visualization module.

Computes classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
confusion matrix with lending cost analysis, and generates interactive
Plotly visualization charts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_model_performance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_split_name: str = "Test",
    cost_fp: float = 5.0,
    cost_fn: float = 1.0,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics, reports, and lending risk costs.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Binary ground truth labels (0 = Poor Credit Risk, 1 = Good Credit Risk).
    dataset_split_name : str, default='Test'
        Label identifying the split ('Train' or 'Test').
    cost_fp : float, default=5.0
        Cost weight for False Positive (Actual Poor Risk approved as Good -> Default).
    cost_fn : float, default=1.0
        Cost weight for False Negative (Actual Good Risk rejected as Poor -> Opportunity loss).

    Returns
    -------
    Dict[str, Any]
        Dictionary with all numerical metrics, confusion matrix, and reports.
    """
    y_pred = pipeline.predict(X)

    # Probabilities for positive class (1 = Good Credit Risk)
    y_proba = None
    roc_auc = None
    roc_fpr, roc_tpr, roc_thresholds = None, None, None

    if hasattr(pipeline, "predict_proba"):
        try:
            probas = pipeline.predict_proba(X)
            # Ensure probabilities match binary layout
            if probas.shape[1] == 2:
                y_proba = probas[:, 1]
            elif probas.shape[1] == 1:
                y_proba = probas[:, 0]
        except Exception:
            y_proba = None

    # Check validity for ROC-AUC
    n_unique_y = len(np.unique(y))
    if y_proba is not None and n_unique_y >= 2:
        try:
            roc_auc = float(roc_auc_score(y, y_proba))
            fpr, tpr, thresh = roc_curve(y, y_proba)
            roc_fpr = fpr.tolist()
            roc_tpr = tpr.tolist()
            roc_thresholds = thresh.tolist()
        except Exception:
            roc_auc = None

    # Base metrics
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    # Recall on Class 0 (Poor Credit Risk): Crucial in credit underwriting!
    # Class 0: y == 0
    rec_poor_risk = float(recall_score(y, y_pred, pos_label=0, zero_division=0))
    prec_poor_risk = float(precision_score(y, y_pred, pos_label=0, zero_division=0))

    # Confusion matrix
    # Format: [[TN, FP], [FN, TP]]
    # TN: Actual 0 (Poor Risk), Predicted 0 (Poor Risk) - Correctly flagged risky!
    # FP: Actual 0 (Poor Risk), Predicted 1 (Good Risk) - Default disaster!
    # FN: Actual 1 (Good Risk), Predicted 0 (Poor Risk) - Opportunity rejection
    # TP: Actual 1 (Good Risk), Predicted 1 (Good Risk) - Profitable loan approved
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Asymmetric lending financial cost calculation
    total_financial_loss_score = (fp * cost_fp) + (fn * cost_fn)

    # Detailed report dict
    report = classification_report(
        y,
        y_pred,
        labels=[0, 1],
        target_names=["Poor Credit Risk (0)", "Good Credit Risk (1)"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "split": dataset_split_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "recall_poor_credit_risk": rec_poor_risk,
        "precision_poor_credit_risk": prec_poor_risk,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": cm.tolist(),
        },
        "financial_cost_score": total_financial_loss_score,
        "cost_assumptions": {
            "cost_fp_default": cost_fp,
            "cost_fn_opportunity": cost_fn,
        },
        "roc_curve": {
            "fpr": roc_fpr,
            "tpr": roc_tpr,
            "thresholds": roc_thresholds,
        },
        "classification_report": report,
        "total_instances": len(y),
    }


def create_confusion_matrix_figure(
    cm_dict: Dict[str, Any],
    title: str = "Confusion Matrix & Underwriting Outcomes"
) -> go.Figure:
    """Generate an annotated interactive Plotly heatmap for credit underwriting outcomes.

    Parameters
    ----------
    cm_dict : Dict[str, Any]
        Dictionary containing tn, fp, fn, tp counts.
    title : str, default='Confusion Matrix & Underwriting Outcomes'

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]
    total = max(tn + fp + fn + tp, 1)

    z = [[tn, fp], [fn, tp]]

    annotations_text = [
        [
            f"<b>TN (True Negative)</b><br>{tn} applicants<br>({(tn/total)*100:.1f}%)<br><i>Risky applicant flagged</i>",
            f"<b>FP (False Positive)</b><br>{fp} applicants<br>({(fp/total)*100:.1f}%)<br>⚠️ <i>Default loss exposure</i>",
        ],
        [
            f"<b>FN (False Negative)</b><br>{fn} applicants<br>({(fn/total)*100:.1f}%)<br><i>Opportunity lost</i>",
            f"<b>TP (True Positive)</b><br>{tp} applicants<br>({(tp/total)*100:.1f}%)<br>✓ <i>Sound loan approved</i>",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted: Poor Risk (0)", "Predicted: Good Risk (1)"],
            y=["Actual: Poor Risk (0)", "Actual: Good Risk (1)"],
            text=annotations_text,
            texttemplate="%{text}",
            textfont={"size": 13, "color": "white"},
            colorscale=[[0, "#1e293b"], [0.5, "#334155"], [1, "#0284c7"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.5},
        xaxis_title="<b>Model Prediction</b>",
        yaxis_title="<b>Ground Truth Risk Category</b>",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Roboto, sans-serif", size=12),
    )

    return fig


def create_roc_curves_figure(
    models_evaluation: Dict[str, Dict[str, Any]],
    title: str = "Receiver Operating Characteristic (ROC) Comparison"
) -> go.Figure:
    """Generate multi-model interactive ROC curves overlaid with diagonal random line.

    Parameters
    ----------
    models_evaluation : Dict[str, Dict[str, Any]]
        Evaluation results for each model containing 'roc_curve' and 'roc_auc'.
    title : str, default='Receiver Operating Characteristic (ROC) Comparison'

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    fig = go.Figure()

    # Diagonal no-discrimination reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#94a3b8", width=2, dash="dash"),
            name="Random Guess (AUC = 0.50)",
        )
    )

    colors = ["#0284c7", "#10b981", "#f59e0b", "#8b5cf6"]

    for idx, (name, eval_dict) in enumerate(models_evaluation.items()):
        roc_data = eval_dict.get("roc_curve", {})
        auc_val = eval_dict.get("roc_auc")
        fpr = roc_data.get("fpr")
        tpr = roc_data.get("tpr")

        if fpr and tpr:
            auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color=color, width=3),
                    name=f"<b>{name}</b> (AUC = {auc_str})",
                )
            )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.5},
        xaxis_title="<b>False Positive Rate (1 - Specificity)</b>",
        yaxis_title="<b>True Positive Rate (Sensitivity / Recall)</b>",
        xaxis=dict(range=[0.0, 1.0], gridcolor="#e2e8f0"),
        yaxis=dict(range=[0.0, 1.05], gridcolor="#e2e8f0"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(255,255,255,0.85)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Roboto, sans-serif", size=12),
    )

    return fig


def create_model_comparison_bar_chart(
    models_evaluation: Dict[str, Dict[str, Any]],
    title: str = "Multi-Metric Model Comparison (Test Split)"
) -> go.Figure:
    """Generate grouped bar chart comparing Accuracy, Precision, Recall, F1, and ROC-AUC.

    Parameters
    ----------
    models_evaluation : Dict[str, Dict[str, Any]]
        Evaluation dictionaries keyed by model name.
    title : str, default='Multi-Metric Model Comparison (Test Split)'

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Poor Risk Recall"]
    metric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "recall_poor_credit_risk",
    ]

    colors = ["#0284c7", "#10b981", "#f59e0b", "#8b5cf6"]
    fig = go.Figure()

    for idx, (model_name, eval_dict) in enumerate(models_evaluation.items()):
        values = []
        for key in metric_keys:
            val = eval_dict.get(key)
            values.append(val if val is not None else 0.0)

        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=f"<b>{model_name}</b>",
                marker_color=colors[idx % len(colors)],
                text=[f"{v:.1%}" if v > 0 else "N/A" for v in values],
                textposition="auto",
            )
        )

    fig.update_layout(
        title={"text": f"<b>{title}</b>", "x": 0.5},
        yaxis_title="<b>Score</b>",
        yaxis=dict(range=[0.0, 1.1], gridcolor="#e2e8f0"),
        barmode="group",
        margin=dict(l=40, r=40, t=50, b=40),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Roboto, sans-serif", size=12),
    )

    return fig


def get_metric_educational_definitions() -> Dict[str, Dict[str, str]]:
    """Return plain-language definitions and financial credit underwriting contexts.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Dictionary explaining each metric.
    """
    return {
        "Accuracy": {
            "definition": "The fraction of total credit applicants correctly categorized as Good or Poor credit risk.",
            "financial_context": (
                "Can be misleading when credit default rates are low. If 90% of applicants are creditworthy, "
                "a naive model predicting everyone as good achieves 90% accuracy while failing to prevent a single loan default."
            ),
        },
        "Precision": {
            "definition": "Out of all applicants predicted to be Good Credit Risk, the proportion who were genuinely creditworthy.",
            "financial_context": (
                "High precision ensures that approved loans are sound. Low precision implies the bank is approving risky applicants "
                "who will subsequently default on debt obligations."
            ),
        },
        "Recall (Sensitivity)": {
            "definition": "Out of all genuinely creditworthy applicants, the proportion successfully identified and approved.",
            "financial_context": (
                "High recall ensures that qualified applicants are not erroneously turned away. Low recall forfeits loan interest revenue."
            ),
        },
        "F1-Score": {
            "definition": "The harmonic mean of Precision and Recall, providing a balanced evaluation.",
            "financial_context": (
                "Essential when finding an optimal trade-off between aggressive risk aversion and customer acquisition."
            ),
        },
        "ROC-AUC": {
            "definition": "Area Under the Receiver Operating Characteristic curve. Measures the model's ability to rank risk properly.",
            "financial_context": (
                "An AUC of 0.80 means there is an 80% chance the model assigns a higher creditworthiness score to a creditworthy "
                "borrower than to a defaulting borrower across all possible score thresholds."
            ),
        },
        "Poor Credit Risk Recall": {
            "definition": "The percentage of actually risky/defaulting applicants correctly flagged as Poor Credit Risk.",
            "financial_context": (
                "Because default losses typically cost banks 4x to 5x more than the margin gained from a performing loan, "
                "maximizing recall on the high-risk category is often the paramount regulatory and solvency objective."
            ),
        },
    }
