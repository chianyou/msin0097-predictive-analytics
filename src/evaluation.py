"""Evaluation utilities for churn classification with strict split discipline.

Workflow supported:
1) Tune LogisticRegression hyperparameters with CV on TRAIN only.
2) Tune decision threshold on VALIDATION only.
3) Refit final model on TRAIN+VAL and evaluate once on TEST.
4) Produce calibration diagnostics and save figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data_prep import PreparedData


RANDOM_STATE = 42


@dataclass
class TuningResult:
    """Container for CV hyperparameter tuning results."""

    best_estimator: LogisticRegression
    best_params: Dict[str, object]
    best_pr_auc: float
    cv_results: pd.DataFrame


@dataclass
class ThresholdResult:
    """Container for threshold search results on validation."""

    best_threshold: float
    best_row: pd.Series
    threshold_table: pd.DataFrame
    objective: str


def build_logistic_model(
    C: float = 1.0,
    class_weight: Optional[str] = None,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Construct LogisticRegression with reproducible settings."""
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=3000,
        random_state=random_state,
    )


def tune_logistic_hyperparameters(
    X_train_processed,
    y_train,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> TuningResult:
    """Tune LogisticRegression using PR-AUC via Stratified K-Fold on TRAIN only."""
    base_model = build_logistic_model(random_state=random_state)
    param_grid = {
        "C": [0.1, 0.3, 1, 3, 10],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="average_precision",  # PR-AUC primary metric
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train_processed, y_train)

    cv_df = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    cv_df = cv_df.rename(
        columns={
            "mean_test_score": "mean_pr_auc",
            "std_test_score": "std_pr_auc",
        }
    )

    return TuningResult(
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_pr_auc=float(search.best_score_),
        cv_results=cv_df,
    )


def _classification_metrics_at_threshold(
    y_true,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def search_thresholds(
    y_true,
    y_prob: np.ndarray,
    objective: str = "f1",
    min_recall: float = 0.75,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> ThresholdResult:
    """Search threshold on VALIDATION probabilities.

    objective:
    - "f1": maximize F1.
    - "constrained_precision": maximize precision subject to recall >= min_recall.
    """
    thresholds = np.round(
        np.arange(threshold_min, threshold_max + threshold_step, threshold_step), 2
    )
    records = [
        _classification_metrics_at_threshold(y_true=y_true, y_prob=y_prob, threshold=t)
        for t in thresholds
    ]
    table = pd.DataFrame(records)

    if objective == "f1":
        # Tie-breaks: higher recall then lower threshold.
        best = table.sort_values(
            by=["f1", "recall", "threshold"], ascending=[False, False, True]
        ).iloc[0]
    elif objective == "constrained_precision":
        feasible = table[table["recall"] >= min_recall]
        if feasible.empty:
            # Fallback: no threshold satisfies recall constraint.
            best = table.sort_values(by=["recall", "threshold"], ascending=[False, True]).iloc[0]
        else:
            best = feasible.sort_values(
                by=["precision", "f1", "threshold"], ascending=[False, False, True]
            ).iloc[0]
    else:
        raise ValueError("objective must be 'f1' or 'constrained_precision'.")

    return ThresholdResult(
        best_threshold=float(best["threshold"]),
        best_row=best,
        threshold_table=table,
        objective=objective,
    )


def evaluate_probabilities(y_true, y_prob: np.ndarray, threshold: float) -> Dict[str, object]:
    """Compute full metric set at a fixed threshold, plus confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "confusion_matrix": cm,
    }


def plot_pr_curve(y_true, y_prob: np.ndarray, save_path: Optional[str] = None, title: str = "Precision-Recall Curve"):
    """Plot PR curve and optionally save to disk."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_confusion(y_true, y_prob: np.ndarray, threshold: float, save_path: Optional[str] = None, title: str = "Confusion Matrix"):
    """Plot confusion matrix at selected threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_calibration_curve(
    y_true,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Calibration Curve",
):
    """Plot calibration curve and annotate Brier score."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_pred, frac_pos, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(f"{title} (Brier={brier:.4f})")
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax, float(brier)


def fit_model_for_validation(
    prepared: PreparedData,
    best_params: Dict[str, object],
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Fit best logistic model on TRAIN only (for validation threshold tuning)."""
    model = build_logistic_model(
        C=float(best_params["C"]),
        class_weight=best_params["class_weight"],
        random_state=random_state,
    )
    # Uses train-only transformed features, so validation remains unseen for fitting.
    model.fit(prepared.X_train_processed, prepared.y_train)
    return model


def fit_final_model_trainval(
    prepared: PreparedData,
    best_params: Dict[str, object],
    random_state: int = RANDOM_STATE,
):
    """Refit preprocessor + model on TRAIN+VAL, then transform TEST once.

    Leakage control:
    - TEST is never used in fitting preprocessor, model, or threshold selection.
    """
    X_trainval_raw = pd.concat([prepared.X_train, prepared.X_val], axis=0)
    y_trainval = pd.concat([prepared.y_train, prepared.y_val], axis=0)

    # Refit preprocessing on Train+Val for final model capacity.
    preprocessor_final = clone(prepared.preprocessor)
    X_trainval_processed = preprocessor_final.fit_transform(X_trainval_raw, y_trainval)
    X_test_processed = preprocessor_final.transform(prepared.X_test)

    model_final = build_logistic_model(
        C=float(best_params["C"]),
        class_weight=best_params["class_weight"],
        random_state=random_state,
    )
    model_final.fit(X_trainval_processed, y_trainval)
    return model_final, preprocessor_final, X_test_processed


def run_step5_evaluation(
    prepared: PreparedData,
    threshold_objective: str = "f1",
    min_recall: float = 0.75,
    figures_dir: str = "reports/figures",
    random_state: int = RANDOM_STATE,
) -> Dict[str, object]:
    """End-to-end Step 5 evaluation with strict split usage."""
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    # 1) Hyperparameter tuning on TRAIN only.
    tuning = tune_logistic_hyperparameters(
        X_train_processed=prepared.X_train_processed,
        y_train=prepared.y_train,
        n_splits=5,
        random_state=random_state,
    )

    # 2) Threshold tuning on VALIDATION only (using model fit only on TRAIN).
    val_model = fit_model_for_validation(prepared, tuning.best_params, random_state=random_state)
    val_prob = val_model.predict_proba(prepared.X_val_processed)[:, 1]

    threshold_result = search_thresholds(
        y_true=prepared.y_val,
        y_prob=val_prob,
        objective=threshold_objective,
        min_recall=min_recall,
    )

    # Calibration on validation (tuning phase diagnostics).
    plot_calibration_curve(
        y_true=prepared.y_val,
        y_prob=val_prob,
        save_path=str(figures / "calibration_validation.png"),
        title="Validation Calibration Curve",
    )

    # 3) Final fit on Train+Val, then single evaluation on TEST.
    final_model, final_preprocessor, X_test_processed_final = fit_final_model_trainval(
        prepared=prepared,
        best_params=tuning.best_params,
        random_state=random_state,
    )
    test_prob = final_model.predict_proba(X_test_processed_final)[:, 1]
    test_metrics = evaluate_probabilities(
        y_true=prepared.y_test,
        y_prob=test_prob,
        threshold=threshold_result.best_threshold,
    )

    # 4) Save required figures from final test evaluation.
    plot_pr_curve(
        y_true=prepared.y_test,
        y_prob=test_prob,
        save_path=str(figures / "pr_curve_test.png"),
        title="Test Precision-Recall Curve",
    )
    plot_confusion(
        y_true=prepared.y_test,
        y_prob=test_prob,
        threshold=threshold_result.best_threshold,
        save_path=str(figures / "confusion_matrix_test.png"),
        title=f"Test Confusion Matrix (threshold={threshold_result.best_threshold:.2f})",
    )
    plot_calibration_curve(
        y_true=prepared.y_test,
        y_prob=test_prob,
        save_path=str(figures / "calibration_test.png"),
        title="Test Calibration Curve",
    )

    # Tabular summary for reporting.
    final_report = pd.DataFrame(
        [
            {
                "pr_auc": test_metrics["pr_auc"],
                "roc_auc": test_metrics["roc_auc"],
                "f1": test_metrics["f1"],
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "brier_score": test_metrics["brier_score"],
                "threshold": test_metrics["threshold"],
            }
        ]
    )

    return {
        "tuning": tuning,
        "threshold_result": threshold_result,
        "validation_probabilities": val_prob,
        "final_model": final_model,
        "final_preprocessor": final_preprocessor,
        "test_probabilities": test_prob,
        "test_metrics": test_metrics,
        "final_report_df": final_report,
        "figures_dir": str(figures.resolve()),
    }
