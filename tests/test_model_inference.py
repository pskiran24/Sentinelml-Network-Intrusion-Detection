
"""
SentinelML - Model Inference Tests

Tests:
1. Model artifact existence
2. Model loading
3. Input feature validation
4. Binary model predictions
5. Multiclass model predictions
6. Probability validation
7. Invalid input rejection
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.sentinelml.config import (
    MODELING_DATA_PATH,
    MULTICLASS_MODEL_PATH,
    MULTICLASS_ENCODER_PATH,
)

from src.sentinelml.train import BINARY_MODEL_PATH

from src.sentinelml.preprocessing import (
    prepare_labeled_data,
    validate_feature_columns,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture(scope="module")
def sample_features():
    """
    Load real dataset records for inference testing.

    Returns 10 samples with the expected 82 features.
    """

    df = pd.read_csv(MODELING_DATA_PATH)

    X, y = prepare_labeled_data(df)

    return X.iloc[:10].copy()


@pytest.fixture(scope="module")
def binary_model():
    """Load the saved binary classification pipeline."""

    return joblib.load(BINARY_MODEL_PATH)


@pytest.fixture(scope="module")
def multiclass_model():
    """Load the saved multiclass classification pipeline."""

    return joblib.load(MULTICLASS_MODEL_PATH)


@pytest.fixture(scope="module")
def label_encoder():
    """Load the saved multiclass label encoder."""

    return joblib.load(MULTICLASS_ENCODER_PATH)


# ============================================================
# 1. ARTIFACT EXISTENCE TESTS
# ============================================================

def test_binary_model_exists():

    assert BINARY_MODEL_PATH.exists()


def test_multiclass_model_exists():

    assert MULTICLASS_MODEL_PATH.exists()


def test_label_encoder_exists():

    assert MULTICLASS_ENCODER_PATH.exists()


# ============================================================
# 2. MODEL LOADING TESTS
# ============================================================

def test_binary_model_loads(binary_model):

    assert binary_model is not None


def test_multiclass_model_loads(multiclass_model):

    assert multiclass_model is not None


# ============================================================
# 3. INPUT FEATURE TESTS
# ============================================================

def test_sample_has_expected_feature_count(sample_features):

    assert sample_features.shape[1] == 82


def test_sample_contains_no_missing_values(sample_features):

    assert not sample_features.isnull().any().any()


def test_valid_features_pass_schema_validation(sample_features):

    validated_features = validate_feature_columns(
        sample_features
    )

    assert validated_features.shape == sample_features.shape


# ============================================================
# 4. BINARY MODEL INFERENCE TESTS
# ============================================================

def test_binary_prediction_shape(
    binary_model,
    sample_features,
):

    predictions = binary_model.predict(
        sample_features
    )

    assert predictions.shape == (10,)


def test_binary_predictions_are_valid(
    binary_model,
    sample_features,
):

    predictions = binary_model.predict(
        sample_features
    )

    assert set(predictions).issubset({
        "Normal",
        "Attack",
    })


def test_binary_probabilities_are_valid(
    binary_model,
    sample_features,
):

    probabilities = binary_model.predict_proba(
        sample_features
    )

    assert probabilities.shape == (10, 2)

    assert np.all(
        (probabilities >= 0) &
        (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


# ============================================================
# 5. MULTICLASS MODEL INFERENCE TESTS
# ============================================================

def test_multiclass_prediction_shape(
    multiclass_model,
    sample_features,
):

    predictions = multiclass_model.predict(
        sample_features
    )

    assert predictions.shape == (10,)


def test_multiclass_predictions_are_valid(
    multiclass_model,
    label_encoder,
    sample_features,
):

    predictions = multiclass_model.predict(
        sample_features
    )

    decoded_predictions = label_encoder.inverse_transform(
        predictions.astype(int)
    )

    assert set(decoded_predictions).issubset(
        set(label_encoder.classes_)
    )


def test_multiclass_probabilities_are_valid(
    multiclass_model,
    sample_features,
):

    probabilities = multiclass_model.predict_proba(
        sample_features
    )

    assert probabilities.shape == (10, 12)

    assert np.all(
        (probabilities >= 0) &
        (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    )


# ============================================================
# 6. INVALID INPUT TESTS
# ============================================================

def test_binary_model_rejects_missing_feature(
    sample_features,
):

    invalid_input = sample_features.drop(
        columns=[sample_features.columns[0]]
    )

    with pytest.raises(ValueError):

        validate_feature_columns(invalid_input)


def test_multiclass_model_rejects_missing_feature(
    sample_features,
):

    invalid_input = sample_features.drop(
        columns=[sample_features.columns[0]]
    )

    with pytest.raises(ValueError):

        validate_feature_columns(invalid_input)


def test_binary_model_rejects_extra_feature(
    sample_features,
):

    invalid_input = sample_features.copy()

    invalid_input["unexpected_feature"] = 1

    with pytest.raises(ValueError):

        validate_feature_columns(invalid_input)


def test_multiclass_model_rejects_extra_feature(
    sample_features,
):

    invalid_input = sample_features.copy()

    invalid_input["unexpected_feature"] = 1

    with pytest.raises(ValueError):

        validate_feature_columns(invalid_input)
