
"""
SentinelML - FastAPI Model Serving Module

Responsibilities:
1. Load trained binary and multiclass models.
2. Load the multiclass label encoder.
3. Validate incoming network-flow features.
4. Generate binary and multiclass predictions.
5. Return prediction probabilities through REST endpoints.
"""

from contextlib import asynccontextmanager
from math import isfinite
from numbers import Real
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.sentinelml.config import (
    ARTIFACTS_DIR,
    MULTICLASS_MODEL_PATH,
    MULTICLASS_ENCODER_PATH,
)
from src.sentinelml.train import BINARY_MODEL_PATH
from src.sentinelml.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    validate_feature_columns,
)


# ---------------------------------------------------------
# 1. Artifact paths
# ---------------------------------------------------------

BINARY_ARTIFACT_PATH = BINARY_MODEL_PATH
MULTICLASS_ARTIFACT_PATH = MULTICLASS_MODEL_PATH
ENCODER_ARTIFACT_PATH = MULTICLASS_ENCODER_PATH


# ---------------------------------------------------------
# 2. Global model references
# ---------------------------------------------------------

binary_model = None
multiclass_model = None
label_encoder = None


# ---------------------------------------------------------
# 3. Load models when the API starts
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global binary_model, multiclass_model, label_encoder

    required_artifacts = [
        BINARY_ARTIFACT_PATH,
        MULTICLASS_ARTIFACT_PATH,
        ENCODER_ARTIFACT_PATH,
    ]

    missing_artifacts = [
        str(path) for path in required_artifacts if not path.exists()
    ]

    if missing_artifacts:
        raise RuntimeError(
            f"Required model artifacts are missing: {missing_artifacts}"
        )

    binary_model = joblib.load(BINARY_ARTIFACT_PATH)
    multiclass_model = joblib.load(MULTICLASS_ARTIFACT_PATH)
    label_encoder = joblib.load(ENCODER_ARTIFACT_PATH)

    print("SentinelML models loaded successfully.")

    yield

    binary_model = None
    multiclass_model = None
    label_encoder = None

    print("SentinelML API shutting down.")


# ---------------------------------------------------------
# 4. Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="SentinelML - Network Intrusion Detection API",
    description=(
        "REST API for binary and multiclass network intrusion "
        "detection using the RT-IoT2022 dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# 5. Request schema
# ---------------------------------------------------------

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ...,
        description=(
            "Dictionary containing all 82 model features. "
            "Use the original feature names."
        ),
    )


# ---------------------------------------------------------
# 6. Validate and prepare incoming features
# ---------------------------------------------------------

def prepare_api_input(features: dict[str, Any]) -> pd.DataFrame:
    """
    Validate incoming features and convert them into
    a DataFrame with the expected feature order.
    """

    if not isinstance(features, dict):
        raise ValueError("Features must be provided as a JSON object.")

    expected = set(MODEL_FEATURE_COLUMNS)
    received = set(features.keys())

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing or extra:
        raise ValueError(
            f"Feature schema mismatch. "
            f"Missing features: {missing}. "
            f"Unexpected features: {extra}."
        )

    # Validate numerical features.
    for column in NUMERICAL_FEATURES:
        value = features[column]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"Feature '{column}' must contain a numeric value."
            )

        if not isfinite(float(value)):
            raise ValueError(
                f"Feature '{column}' must contain a finite value."
            )

    # Validate categorical features.
    for column in CATEGORICAL_FEATURES:
        value = features[column]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Feature '{column}' must contain a non-empty string."
            )

    # Preserve the exact feature order expected by the model.
    input_df = pd.DataFrame(
        [[features[column] for column in MODEL_FEATURE_COLUMNS]],
        columns=MODEL_FEATURE_COLUMNS,
    )

    validate_feature_columns(input_df)

    return input_df


# ---------------------------------------------------------
# 7. Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "project": "SentinelML",
        "description": "Network Intrusion Detection API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
    }


# ---------------------------------------------------------
# 8. Health-check endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "binary_model_loaded": binary_model is not None,
        "multiclass_model_loaded": multiclass_model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "expected_features": len(MODEL_FEATURE_COLUMNS),
    }


# ---------------------------------------------------------
# 9. Binary classification endpoint
# ---------------------------------------------------------

@app.post("/predict/binary")
def predict_binary(request: PredictionRequest):

    if binary_model is None:
        raise HTTPException(
            status_code=503,
            detail="Binary model is not loaded.",
        )

    try:
        input_df = prepare_api_input(request.features)

        prediction = binary_model.predict(input_df)[0]

        probabilities = binary_model.predict_proba(input_df)[0]

        class_probabilities = {
            str(class_label): float(probability)
            for class_label, probability in zip(
                binary_model.classes_,
                probabilities,
            )
        }

        return {
            "task": "binary_classification",
            "prediction": str(prediction),
            "probabilities": class_probabilities,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Binary prediction failed: {str(error)}",
        ) from error


# ---------------------------------------------------------
# 10. Multiclass classification endpoint
# ---------------------------------------------------------

@app.post("/predict/multiclass")
def predict_multiclass(request: PredictionRequest):

    if multiclass_model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Multiclass model or label encoder is not loaded.",
        )

    try:
        input_df = prepare_api_input(request.features)

        encoded_prediction = multiclass_model.predict(input_df)[0]

        prediction = label_encoder.inverse_transform(
            [int(encoded_prediction)]
        )[0]

        probabilities = multiclass_model.predict_proba(input_df)[0]

        class_probabilities = {
            str(class_label): float(probability)
            for class_label, probability in zip(
                label_encoder.classes_,
                probabilities,
            )
        }

        return {
            "task": "multiclass_classification",
            "prediction": str(prediction),
            "probabilities": class_probabilities,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Multiclass prediction failed: {str(error)}",
        ) from error
