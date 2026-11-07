
"""
SentinelML - FastAPI Automated Tests

Tests:
1. API homepage
2. Model health
3. Binary prediction
4. Multiclass prediction
5. Invalid input handling
"""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.sentinelml.api import app
from src.sentinelml.config import (
    MODELING_DATA_PATH,
    SPLIT_MANIFEST_PATH,
)
from src.sentinelml.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    CONSTANT_FEATURES,
)


# ---------------------------------------------------------
# 1. Load API client and initialize application lifespan
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------
# 2. Prepare a real test-dataset sample
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def sample_payload():
    df = pd.read_csv(MODELING_DATA_PATH)
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    test_positions = manifest.loc[
        manifest["split"] == "test", "original_index"
    ].tolist()

    sample_position = int(test_positions[0])
    sample = df.iloc[sample_position]

    features = (
        sample
        .drop(labels=["Attack_type"] + CONSTANT_FEATURES)
        .reindex(MODEL_FEATURE_COLUMNS)
    )

    payload = {
        "features": {
            column: (
                value.item() if hasattr(value, "item") else value
            )
            for column, value in features.items()
        }
    }

    return payload


# ---------------------------------------------------------
# 3. Homepage test
# ---------------------------------------------------------

def test_api_root(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "SentinelML"
    assert data["status"] == "running"


# ---------------------------------------------------------
# 4. Health-check test
# ---------------------------------------------------------

def test_api_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["binary_model_loaded"] is True
    assert data["multiclass_model_loaded"] is True
    assert data["label_encoder_loaded"] is True
    assert data["expected_features"] == 82


# ---------------------------------------------------------
# 5. Binary prediction test
# ---------------------------------------------------------

def test_binary_prediction(client, sample_payload):
    response = client.post(
        "/predict/binary",
        json=sample_payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["task"] == "binary_classification"
    assert data["prediction"] in ["Normal", "Attack"]

    probabilities = data["probabilities"]

    assert set(probabilities.keys()) == {"Normal", "Attack"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


# ---------------------------------------------------------
# 6. Multiclass prediction test
# ---------------------------------------------------------

def test_multiclass_prediction(client, sample_payload):
    response = client.post(
        "/predict/multiclass",
        json=sample_payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["task"] == "multiclass_classification"

    probabilities = data["probabilities"]

    assert len(probabilities) == 12
    assert data["prediction"] in probabilities

    assert all(0 <= p <= 1 for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


# ---------------------------------------------------------
# 7. Missing feature test
# ---------------------------------------------------------

def test_api_rejects_missing_feature(client, sample_payload):
    invalid_payload = json.loads(json.dumps(sample_payload))

    invalid_payload["features"].pop("flow_duration")

    response = client.post(
        "/predict/binary",
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "Missing features" in response.json()["detail"]


# ---------------------------------------------------------
# 8. Unexpected feature test
# ---------------------------------------------------------

def test_api_rejects_extra_feature(client, sample_payload):
    invalid_payload = json.loads(json.dumps(sample_payload))

    invalid_payload["features"]["unexpected_feature"] = 123

    response = client.post(
        "/predict/multiclass",
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "Unexpected features" in response.json()["detail"]


# ---------------------------------------------------------
# 9. Invalid numerical feature test
# ---------------------------------------------------------

def test_api_rejects_invalid_numeric_value(client, sample_payload):
    invalid_payload = json.loads(json.dumps(sample_payload))

    invalid_payload["features"]["flow_duration"] = "invalid"

    response = client.post(
        "/predict/binary",
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "must contain a numeric value" in response.json()["detail"]


# ---------------------------------------------------------
# 10. Empty categorical feature test
# ---------------------------------------------------------

def test_api_rejects_empty_categorical_value(client, sample_payload):
    invalid_payload = json.loads(json.dumps(sample_payload))

    invalid_payload["features"]["proto"] = ""

    response = client.post(
        "/predict/multiclass",
        json=invalid_payload,
    )

    assert response.status_code == 422
    assert "non-empty string" in response.json()["detail"]
