"""Data preparation utilities for Telco Customer Churn classification.

This module implements:
1) Strict stratified train/validation/test splits.
2) Leakage-safe preprocessing with scikit-learn Pipeline + ColumnTransformer.
3) Validation checks for transformed outputs.

All stochastic operations use random_state=42 for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


RANDOM_STATE = 42
TARGET_COL = "Churn"
ID_COL = "customerID"


@dataclass
class PreparedData:
    """Container for raw splits, transformed arrays, and fitted preprocessor."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    X_train_processed: sparse.spmatrix | np.ndarray
    X_val_processed: sparse.spmatrix | np.ndarray
    X_test_processed: sparse.spmatrix | np.ndarray
    preprocessor: Pipeline


def load_telco_data(csv_path: str) -> pd.DataFrame:
    """Load dataset without fitting any transformations.

    Path resolution is robust to common notebook working directories:
    - absolute path
    - current working directory
    - project root (one level above this src/ folder)
    - project data folders: data/, data/raw/, datasets/
    """
    resolved_path = _resolve_dataset_path(csv_path)
    df = pd.read_csv(resolved_path)
    required_cols = {TARGET_COL, ID_COL, "TotalCharges"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    return df


def _resolve_dataset_path(csv_path: str) -> Path:
    """Resolve dataset path across common project layouts."""
    requested = Path(csv_path).expanduser()
    if requested.is_absolute() and requested.exists():
        return requested

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path.cwd() / requested,
        project_root / requested,
        project_root / "data" / requested.name,
        project_root / "data" / "raw" / requested.name,
        project_root / "datasets" / requested.name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Could not find dataset file. "
        "Provide an absolute path or place the file in a standard project location.\n"
        f"Requested: {csv_path}\n"
        f"Searched:\n{searched}"
    )


def encode_target(y: pd.Series) -> pd.Series:
    """Encode target: Yes -> 1, No -> 0."""
    valid_labels = {"Yes", "No"}
    observed = set(y.dropna().unique())
    if not observed.issubset(valid_labels):
        raise ValueError(f"Unexpected target labels: {sorted(observed - valid_labels)}")
    return y.map({"No": 0, "Yes": 1}).astype("int64")


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = RANDOM_STATE,
    test_size: float = 0.20,
    val_size_within_trainval: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train/val/test split with stratification.

    With defaults:
    - test = 20%
    - val  = 20% of full data (25% of remaining 80%)
    - train = 60%
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    # Split first on raw data. No global fitting/statistics are computed here.
    X = df.drop(columns=[target_col])
    y = encode_target(df[target_col])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size_within_trainval,
        stratify=y_trainval,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _basic_cleaning(X: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, leakage-safe cleaning on feature matrix only."""
    X_clean = X.copy()

    # Prevent identifier leakage by removing customer-level unique ID.
    if ID_COL in X_clean.columns:
        X_clean = X_clean.drop(columns=[ID_COL])

    # Safe numeric conversion: blanks become NaN, then coerced to numeric.
    # Imputation of NaN happens later inside the training-fitted numeric pipeline.
    if "TotalCharges" in X_clean.columns:
        X_clean["TotalCharges"] = (
            X_clean["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
        )
        X_clean["TotalCharges"] = pd.to_numeric(
            X_clean["TotalCharges"], errors="coerce"
        )

    return X_clean


def build_preprocessor() -> Pipeline:
    """Build preprocessing pipeline to be fit on training data only."""
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    # Order matters:
    # 1) deterministic cleaning (no learned parameters)
    # 2) learned transforms fit only on train split
    preprocessor = Pipeline(
        steps=[
            ("basic_cleaning", FunctionTransformer(_basic_cleaning, validate=False)),
            ("feature_transform", column_transformer),
        ]
    )
    return preprocessor


def class_distribution(y: pd.Series) -> pd.DataFrame:
    """Return class counts and rates for reporting."""
    counts = y.value_counts().sort_index()
    rates = y.value_counts(normalize=True).sort_index()
    return pd.DataFrame(
        {
            "count": counts,
            "proportion": rates,
        }
    )


def validate_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    """Validate split integrity and stratification behavior."""
    if len(X_train) != len(y_train) or len(X_val) != len(y_val) or len(X_test) != len(y_test):
        raise ValueError("Feature/target row count mismatch in one or more splits.")

    split_names = ["train", "val", "test"]
    y_splits = [y_train, y_val, y_test]
    for name, y_split in zip(split_names, y_splits):
        unique = set(y_split.unique())
        if unique - {0, 1}:
            raise ValueError(f"Unexpected target values in {name}: {sorted(unique)}")


def _has_nan(matrix: sparse.spmatrix | np.ndarray) -> bool:
    """Check NaN presence for dense or sparse matrices."""
    if sparse.issparse(matrix):
        return bool(np.isnan(matrix.data).any())
    return bool(np.isnan(matrix).any())


def validate_transformed_outputs(
    X_train_processed: sparse.spmatrix | np.ndarray,
    X_val_processed: sparse.spmatrix | np.ndarray,
    X_test_processed: sparse.spmatrix | np.ndarray,
    expected_train_rows: int,
    expected_val_rows: int,
    expected_test_rows: int,
) -> None:
    """Validate matrix shapes, nulls, and dtype consistency after preprocessing."""
    if X_train_processed.shape[0] != expected_train_rows:
        raise ValueError("Train transformed row count mismatch.")
    if X_val_processed.shape[0] != expected_val_rows:
        raise ValueError("Validation transformed row count mismatch.")
    if X_test_processed.shape[0] != expected_test_rows:
        raise ValueError("Test transformed row count mismatch.")

    train_cols = X_train_processed.shape[1]
    if X_val_processed.shape[1] != train_cols or X_test_processed.shape[1] != train_cols:
        raise ValueError("Feature dimension mismatch across transformed splits.")

    if _has_nan(X_train_processed) or _has_nan(X_val_processed) or _has_nan(X_test_processed):
        raise ValueError("NaN values remain after preprocessing.")

    # OneHotEncoder + numeric imputers should output numeric matrix only.
    for matrix, name in [
        (X_train_processed, "train"),
        (X_val_processed, "val"),
        (X_test_processed, "test"),
    ]:
        if sparse.issparse(matrix):
            dtype = matrix.dtype
        else:
            dtype = matrix.dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"Unexpected non-numeric dtype after transform in {name}: {dtype}")


def prepare_telco_data(
    csv_path: str,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """End-to-end data preparation for reusable modeling workflows.

    Leakage prevention:
    - Split data before fitting any imputation/encoding.
    - Fit preprocessor only on X_train.
    - Apply learned transformations to validation/test via transform only.
    """
    df = load_telco_data(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        df,
        target_col=TARGET_COL,
        random_state=random_state,
    )
    validate_splits(X_train, X_val, X_test, y_train, y_val, y_test)

    preprocessor = build_preprocessor()

    # Fit only on training data to avoid leakage from val/test distributions.
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    validate_transformed_outputs(
        X_train_processed=X_train_processed,
        X_val_processed=X_val_processed,
        X_test_processed=X_test_processed,
        expected_train_rows=len(X_train),
        expected_val_rows=len(X_val),
        expected_test_rows=len(X_test),
    )

    return PreparedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        X_train_processed=X_train_processed,
        X_val_processed=X_val_processed,
        X_test_processed=X_test_processed,
        preprocessor=preprocessor,
    )


def print_split_summary(prepared: PreparedData) -> None:
    """Convenience report for split sizes and churn prevalence."""
    print("Split shapes:")
    print(f"  Train: {prepared.X_train.shape}, y: {prepared.y_train.shape}")
    print(f"  Val:   {prepared.X_val.shape}, y: {prepared.y_val.shape}")
    print(f"  Test:  {prepared.X_test.shape}, y: {prepared.y_test.shape}")

    print("\nClass distribution by split (0=No churn, 1=Churn):")
    print("Train:")
    print(class_distribution(prepared.y_train))
    print("Val:")
    print(class_distribution(prepared.y_val))
    print("Test:")
    print(class_distribution(prepared.y_test))

    print("\nTransformed matrix shapes:")
    print(f"  Train processed: {prepared.X_train_processed.shape}")
    print(f"  Val processed:   {prepared.X_val_processed.shape}")
    print(f"  Test processed:  {prepared.X_test_processed.shape}")


def _save_matrix(
    matrix: sparse.spmatrix | np.ndarray,
    output_stem: Path,
) -> Path:
    """Save sparse matrices as .npz and dense matrices as .npy."""
    if sparse.issparse(matrix):
        out_path = output_stem.with_suffix(".npz")
        sparse.save_npz(str(out_path), matrix)
        return out_path

    out_path = output_stem.with_suffix(".npy")
    np.save(out_path, matrix)
    return out_path


def save_prepared_data(
    prepared: PreparedData,
    output_dir: str = "data/processed",
) -> Dict[str, str]:
    """Persist prepared splits and transformed matrices for reuse."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = prepared.X_train.copy()
    train_df[TARGET_COL] = prepared.y_train.values
    val_df = prepared.X_val.copy()
    val_df[TARGET_COL] = prepared.y_val.values
    test_df = prepared.X_test.copy()
    test_df[TARGET_COL] = prepared.y_test.values

    train_csv = out_dir / "train_raw_with_target.csv"
    val_csv = out_dir / "val_raw_with_target.csv"
    test_csv = out_dir / "test_raw_with_target.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    train_processed = _save_matrix(prepared.X_train_processed, out_dir / "X_train_processed")
    val_processed = _save_matrix(prepared.X_val_processed, out_dir / "X_val_processed")
    test_processed = _save_matrix(prepared.X_test_processed, out_dir / "X_test_processed")

    y_train_csv = out_dir / "y_train.csv"
    y_val_csv = out_dir / "y_val.csv"
    y_test_csv = out_dir / "y_test.csv"
    prepared.y_train.to_csv(y_train_csv, index=False, header=True)
    prepared.y_val.to_csv(y_val_csv, index=False, header=True)
    prepared.y_test.to_csv(y_test_csv, index=False, header=True)

    feature_names: list[str] = []
    try:
        feature_transform = prepared.preprocessor.named_steps["feature_transform"]
        feature_names = feature_transform.get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    metadata = {
        "target_column": TARGET_COL,
        "n_features_processed": int(prepared.X_train_processed.shape[1]),
        "train_rows": int(len(prepared.X_train)),
        "val_rows": int(len(prepared.X_val)),
        "test_rows": int(len(prepared.X_test)),
        "feature_names": feature_names,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    saved_paths = {
        "train_raw_with_target": str(train_csv.resolve()),
        "val_raw_with_target": str(val_csv.resolve()),
        "test_raw_with_target": str(test_csv.resolve()),
        "X_train_processed": str(train_processed.resolve()),
        "X_val_processed": str(val_processed.resolve()),
        "X_test_processed": str(test_processed.resolve()),
        "y_train": str(y_train_csv.resolve()),
        "y_val": str(y_val_csv.resolve()),
        "y_test": str(y_test_csv.resolve()),
        "metadata": str(metadata_path.resolve()),
    }
    return saved_paths
