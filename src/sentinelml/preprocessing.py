
"""
SentinelML - Data Preprocessing Module

Responsibilities:
1. Define expected model features.
2. Separate features and target.
3. Validate incoming feature columns.
4. Build numerical and categorical preprocessors.
5. Fit preprocessing on training data only.
6. Transform validation, test and inference data.
7. Save and load preprocessing artifacts.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.sentinelml.config import ARTIFACTS_DIR

from src.sentinelml.data_validation import (
    validate_dataset,
    EXPECTED_COLUMNS,
)


# ============================================================
# 1. FEATURE CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attack_type"

CONSTANT_FEATURES = [
    "bwd_URG_flag_count",
]

MODEL_FEATURE_COLUMNS = [
    col
    for col in EXPECTED_COLUMNS
    if col not in [TARGET_COLUMN] + CONSTANT_FEATURES
]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
]

NUMERICAL_FEATURES = [
    col
    for col in MODEL_FEATURE_COLUMNS
    if col not in CATEGORICAL_FEATURES
]

PREPROCESSOR_PATH = (
    ARTIFACTS_DIR / "preprocessor.joblib"
)


# ============================================================
# 2. VALIDATE FEATURE COLUMNS
# ============================================================

def validate_feature_columns(X):
    """
    Validate incoming model feature columns.

    Checks:
    - Input is a pandas DataFrame.
    - No expected features are missing.
    - No unexpected features are present.
    - Feature ordering is correct.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    expected_columns = list(MODEL_FEATURE_COLUMNS)
    actual_columns = list(X.columns)

    missing_columns = [
        col for col in expected_columns
        if col not in actual_columns
    ]

    extra_columns = [
        col for col in actual_columns
        if col not in expected_columns
    ]

    if missing_columns or extra_columns:
        raise ValueError(
            "Feature schema mismatch. "
            f"Missing columns: {missing_columns}. "
            f"Unexpected columns: {extra_columns}."
        )

    if actual_columns != expected_columns:
        raise ValueError(
            "Feature columns are not in the expected order."
        )

    return X


# ============================================================
# 3. PREPARE LABELED DATA
# ============================================================

def prepare_labeled_data(df):
    """
    Validate labeled data and separate features and target.

    Returns
    -------
    X : pd.DataFrame
        82 model features.

    y : pd.Series
        Target labels.
    """

    validated_df = validate_dataset(
        df,
        labeled=True,
    )

    y = validated_df[TARGET_COLUMN].copy()

    X = validated_df.drop(
        columns=[TARGET_COLUMN] + CONSTANT_FEATURES
    )

    X = validate_feature_columns(X)

    return X, y


# ============================================================
# 4. PREPARE INFERENCE DATA
# ============================================================

def prepare_inference_data(df):
    """
    Validate unlabeled inference data.

    Returns
    -------
    pd.DataFrame
        Validated model features.
    """

    validated_df = validate_dataset(
        df,
        labeled=False,
    )

    X = validated_df.drop(
        columns=CONSTANT_FEATURES,
        errors="ignore",
    )

    X = validate_feature_columns(X)

    return X


# ============================================================
# 5. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(scaled=True):
    """
    Build numerical and categorical preprocessing pipelines.

    Numerical:
        - Median imputation.
        - Optional StandardScaler.

    Categorical:
        - Most frequent imputation.
        - OneHotEncoder.

    Parameters
    ----------
    scaled : bool
        Whether to apply StandardScaler.

    Returns
    -------
    ColumnTransformer
    """

    numerical_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]

    if scaled:
        numerical_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# 6. FIT PREPROCESSOR
# ============================================================

def fit_preprocessor(X_train, scaled=True):
    """
    Fit preprocessing using training data only.

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted preprocessor.

    X_train_transformed : array or sparse matrix
        Transformed training features.
    """

    X_train = validate_feature_columns(
        X_train
    )

    preprocessor = build_preprocessor(
        scaled=scaled
    )

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    return preprocessor, X_train_transformed


# ============================================================
# 7. TRANSFORM FEATURES
# ============================================================

def transform_features(preprocessor, X):
    """
    Transform features using an already fitted preprocessor.

    Does not refit the preprocessor.
    """

    X = validate_feature_columns(X)

    return preprocessor.transform(X)


# ============================================================
# 8. RESOLVE ARTIFACT PATH
# ============================================================

def _resolve_artifact_path(path):
    """
    Resolve relative artifact paths inside ARTIFACTS_DIR.

    Absolute paths are preserved.
    """

    path = Path(path)

    if not path.is_absolute():
        path = ARTIFACTS_DIR / path

    return path


# ============================================================
# 9. SAVE PREPROCESSOR
# ============================================================

def save_preprocessor(
    preprocessor,
    path=PREPROCESSOR_PATH,
):
    """
    Save fitted preprocessing artifact.

    Returns
    -------
    Path
        Actual saved artifact path.
    """

    path = _resolve_artifact_path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        path,
    )

    print(
        f"Preprocessor saved successfully: {path}"
    )

    return path


# ============================================================
# 10. LOAD PREPROCESSOR
# ============================================================

def load_preprocessor(
    path=PREPROCESSOR_PATH,
):
    """
    Load saved preprocessing artifact.
    """

    path = _resolve_artifact_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: {path}"
        )

    return joblib.load(path)
