"""Model experiment harness for Telco churn classification.

This module runs model comparison on TRAIN data only using Stratified K-Fold CV.
Primary metric: PR-AUC (average precision), suitable for churn imbalance.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate


RANDOM_STATE = 42


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, object]:
    """Return baseline and candidate models for churn classification.

    Model choices:
    - DummyClassifier: lower-bound benchmark.
    - LogisticRegression (plain + class_weight='balanced'): linear baseline for interpretability.
    - RandomForest: captures non-linear interactions from EDA (e.g., Contract x tenure).
    - XGBoost/LightGBM if installed; otherwise HistGradientBoosting as modern tabular booster.
    """
    models: Dict[str, object] = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=random_state,
        ),
        "logreg_balanced": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    # Preferred "modern" boosters for tabular data if available.
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        return models
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        return models
    except Exception:
        pass

    # Fallback modern boosting model in pure scikit-learn.
    models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        random_state=random_state,
    )
    return models


def get_scoring() -> Dict[str, object]:
    """Scoring metrics: PR-AUC primary + secondary metrics."""
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "balanced_accuracy": "balanced_accuracy",
    }


def _coerce_features_for_model(X, model):
    """Convert sparse to dense only for models that do not support sparse input."""
    if sparse.issparse(X) and isinstance(model, HistGradientBoostingClassifier):
        return X.toarray()
    return X


def _summarize_cv_scores(model_name: str, cv_results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Aggregate fold-level metrics into mean/std summary row."""
    summary: Dict[str, float] = {"model": model_name}
    metric_names = ["pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_accuracy"]
    for metric in metric_names:
        values = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))
    return summary


def train_and_cv_models(
    X_train_processed,
    y_train,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Run Stratified K-Fold CV model comparison on TRAIN only.

    Leakage prevention:
    - This function must receive train-only data.
    - CV folds are generated only within training data.
    - No validation/test split is referenced or touched.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = get_scoring()
    models = get_candidate_models(random_state=random_state)

    all_raw_results: Dict[str, Dict[str, np.ndarray]] = {}
    rows = []

    for model_name, model in models.items():
        X_model = _coerce_features_for_model(X_train_processed, model)
        model_clone = clone(model)

        cv_results = cross_validate(
            estimator=model_clone,
            X=X_model,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        all_raw_results[model_name] = cv_results
        rows.append(_summarize_cv_scores(model_name, cv_results))

    comparison_df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

    # Human-readable mean ± std columns for report tables.
    for metric in ["pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_accuracy"]:
        comparison_df[f"{metric}_cv"] = comparison_df.apply(
            lambda r: f"{r[f'{metric}_mean']:.4f} ± {r[f'{metric}_std']:.4f}",
            axis=1,
        )

    return comparison_df, all_raw_results


def save_comparison_table(results_df: pd.DataFrame, output_path: str) -> None:
    """Persist model comparison table for reporting."""
    results_df.to_csv(output_path, index=False)

