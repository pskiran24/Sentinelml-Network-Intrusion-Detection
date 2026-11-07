
"""
Unit tests for SentinelML data validation.
"""

import numpy as np
import pandas as pd
import pytest

from pandera.errors import SchemaErrors

from src.sentinelml.config import MODELING_DATA_PATH

from src.sentinelml.data_validation import validate_dataset


# ============================================================
# 1. FIXTURE
# ============================================================

@pytest.fixture
def dataset():

    return pd.read_csv(MODELING_DATA_PATH)


# ============================================================
# 2. VALID DATASET
# ============================================================

def test_valid_dataset_passes(dataset):

    validated_df = validate_dataset(dataset)

    assert validated_df.shape == dataset.shape

    assert "Attack_type" in validated_df.columns

    assert validated_df.isna().sum().sum() == 0


# ============================================================
# 3. MISSING COLUMN
# ============================================================

def test_missing_column_fails(dataset):

    invalid_df = dataset.drop(
        columns=["flow_duration"]
    )

    with pytest.raises(SchemaErrors):

        validate_dataset(invalid_df)


# ============================================================
# 4. INVALID TARGET LABEL
# ============================================================

def test_invalid_target_fails(dataset):

    invalid_df = dataset.copy()

    invalid_df.loc[
        invalid_df.index[0],
        "Attack_type"
    ] = "INVALID_ATTACK"

    with pytest.raises(SchemaErrors):

        validate_dataset(invalid_df)


# ============================================================
# 5. INFERENCE DATA
# ============================================================

def test_inference_dataset_passes(dataset):

    X = dataset.drop(
        columns=["Attack_type"]
    )

    validated_df = validate_dataset(
        X,
        labeled=False,
    )

    assert validated_df.shape == X.shape

    assert "Attack_type" not in validated_df.columns


# ============================================================
# 6. INFINITE VALUES
# ============================================================

def test_infinite_values_fail(dataset):

    invalid_df = dataset.copy()

    invalid_df.loc[
        invalid_df.index[0],
        "flow_duration"
    ] = np.inf

    with pytest.raises(SchemaErrors):

        validate_dataset(invalid_df)


# ============================================================
# 7. MISSING VALUES
# ============================================================

def test_missing_values_fail(dataset):

    invalid_df = dataset.copy()

    invalid_df.loc[
        invalid_df.index[0],
        "flow_duration"
    ] = np.nan

    with pytest.raises(SchemaErrors):

        validate_dataset(invalid_df)


# ============================================================
# 8. UNEXPECTED COLUMN
# ============================================================

def test_unexpected_column_fails(dataset):

    invalid_df = dataset.copy()

    invalid_df["unexpected_feature"] = 1

    with pytest.raises(SchemaErrors):

        validate_dataset(invalid_df)


# ============================================================
# 9. EMPTY DATAFRAME
# ============================================================

def test_empty_dataframe_fails():

    empty_df = pd.DataFrame()

    with pytest.raises(ValueError):

        validate_dataset(empty_df)


# ============================================================
# 10. INVALID INPUT TYPE
# ============================================================

def test_invalid_input_type_fails():

    invalid_input = [
        "not",
        "a",
        "dataframe",
    ]

    with pytest.raises(TypeError):

        validate_dataset(invalid_input)
