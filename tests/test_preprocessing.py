
"""
SentinelML - Preprocessing Unit Tests

Tests feature preparation, preprocessing,
transformation and artifact persistence.
"""

import numpy as np
import pandas as pd
import pytest

from scipy import sparse

from src.sentinelml.config import MODELING_DATA_PATH

from src.sentinelml.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    prepare_labeled_data,
    prepare_inference_data,
    build_preprocessor,
    fit_preprocessor,
    transform_features,
    save_preprocessor,
    load_preprocessor,
)


# ============================================================
# 1. TEST FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def dataset():

    return pd.read_csv(
        MODELING_DATA_PATH
    )


@pytest.fixture(scope="module")
def prepared_data(dataset):

    return prepare_labeled_data(
        dataset
    )


# ============================================================
# 2. FEATURE AND TARGET SEPARATION
# ============================================================

def test_feature_target_separation(prepared_data):

    X, y = prepared_data

    assert X.shape[1] == 82

    assert len(X) == len(y)

    assert "Attack_type" not in X.columns

    assert "bwd_URG_flag_count" not in X.columns


# ============================================================
# 3. FEATURE COLUMN ORDER
# ============================================================

def test_feature_columns_match_expected(prepared_data):

    X, _ = prepared_data

    assert list(X.columns) == MODEL_FEATURE_COLUMNS


# ============================================================
# 4. BUILD SCALED PREPROCESSOR
# ============================================================

def test_build_scaled_preprocessor():

    preprocessor = build_preprocessor(
        scaled=True
    )

    assert preprocessor is not None

    assert len(preprocessor.transformers) == 2


# ============================================================
# 5. BUILD TREE PREPROCESSOR
# ============================================================

def test_build_tree_preprocessor():

    preprocessor = build_preprocessor(
        scaled=False
    )

    assert preprocessor is not None

    assert len(preprocessor.transformers) == 2


# ============================================================
# 6. FIT SCALED PREPROCESSOR
# ============================================================

def test_scaled_preprocessor(prepared_data):

    X, _ = prepared_data

    X_train = X.iloc[:1000]

    preprocessor, X_transformed = fit_preprocessor(
        X_train,
        scaled=True,
    )

    assert preprocessor is not None

    assert X_transformed.shape[0] == 1000

    assert X_transformed.shape[1] > X_train.shape[1]


# ============================================================
# 7. FIT TREE PREPROCESSOR
# ============================================================

def test_tree_preprocessor(prepared_data):

    X, _ = prepared_data

    X_train = X.iloc[:1000]

    preprocessor, X_transformed = fit_preprocessor(
        X_train,
        scaled=False,
    )

    assert preprocessor is not None

    assert X_transformed.shape[0] == 1000

    assert X_transformed.shape[1] > X_train.shape[1]


# ============================================================
# 8. TRANSFORM VALIDATION DATA
# ============================================================

def test_transform_features(prepared_data):

    X, _ = prepared_data

    X_train = X.iloc[:1000]

    X_val = X.iloc[1000:1200]

    preprocessor, X_train_transformed = fit_preprocessor(
        X_train,
        scaled=True,
    )

    X_val_transformed = transform_features(
        preprocessor,
        X_val,
    )

    assert X_val_transformed.shape[0] == 200

    assert (
        X_val_transformed.shape[1]
        == X_train_transformed.shape[1]
    )


# ============================================================
# 9. HANDLE UNSEEN CATEGORICAL VALUES
# ============================================================

def test_unseen_categories_are_handled(prepared_data):

    X, _ = prepared_data

    X_train = X.iloc[:1000].copy()

    X_test = X.iloc[1000:1010].copy()

    preprocessor, _ = fit_preprocessor(
        X_train,
        scaled=True,
    )

    # Introduce a category not seen during training.
    X_test.loc[
        X_test.index[0],
        "service",
    ] = "unknown_service"

    transformed = transform_features(
        preprocessor,
        X_test,
    )

    assert transformed.shape[0] == 10


# ============================================================
# 10. INFERENCE DATA PREPARATION
# ============================================================

def test_inference_data_preparation(dataset):

    inference_df = dataset.drop(
        columns=["Attack_type"]
    )

    X = prepare_inference_data(
        inference_df
    )

    assert X.shape[1] == 82

    assert "Attack_type" not in X.columns

    assert "bwd_URG_flag_count" not in X.columns


# ============================================================
# 11. SAVE AND LOAD PREPROCESSOR
# ============================================================

def test_save_and_load_preprocessor(prepared_data):

    X, _ = prepared_data

    X_train = X.iloc[:1000]

    X_test = X.iloc[1000:1010]

    preprocessor, _ = fit_preprocessor(
        X_train,
        scaled=True,
    )

    filename = "test_preprocessor.joblib"

    save_path = save_preprocessor(
        preprocessor,
        filename,
    )

    try:

        assert save_path.exists()

        loaded_preprocessor = load_preprocessor(
            filename
        )

        original_output = preprocessor.transform(
            X_test
        )

        loaded_output = loaded_preprocessor.transform(
            X_test
        )

        if sparse.issparse(original_output):

            assert (
                original_output != loaded_output
            ).nnz == 0

        else:

            np.testing.assert_allclose(
                original_output,
                loaded_output,
            )

    finally:

        if save_path.exists():
            save_path.unlink()
