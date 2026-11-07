
"""
SentinelML - Model Training Module

Supports:
1. Binary classification: Normal vs Attack.
2. Multiclass classification: 12 attack categories.

Both workflows:
- Reconstruct the existing train/validation/test splits.
- Fit preprocessing using training data only.
- Keep validation data separate from training.
- Save the complete fitted pipeline.
- Track parameters and validation metrics with MLflow.
- Do not log duplicate model artifacts to MLflow.
- Do not use the test set during training.
"""

import argparse
from functools import wraps

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.sentinelml.config import (
    MODELING_DATA_PATH,
    SPLIT_MANIFEST_PATH,
    ARTIFACTS_DIR,
    BINARY_MODEL_PATH,
    MULTICLASS_MODEL_PATH,
    MULTICLASS_ENCODER_PATH,
    MLFLOW_TRACKING_URI,
    BINARY_EXPERIMENT_NAME,
    MULTICLASS_EXPERIMENT_NAME,
)

from src.sentinelml.preprocessing import (
    build_preprocessor,
    prepare_labeled_data,
)


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

NORMAL_CLASSES = {
    "MQTT_Publish",
    "Thing_Speak",
    "Wipro_bulb",
}


BINARY_RF_PARAMS = {
    "n_estimators": 314,
    "max_depth": 25,
    "max_features": 0.5,
    "min_samples_leaf": 2,
    "min_samples_split": 4,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


MULTICLASS_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 12,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# 2. MLFLOW EXPERIMENT TRACKING
# ============================================================

def track_experiment(
    experiment_name,
    run_name,
    model_params,
    tags,
):
    """
    Decorator that tracks a model-training function in MLflow.

    Logs:
    - Model hyperparameters.
    - Task and model metadata.
    - Validation metrics.

    Does not log model files or duplicate artifacts.
    """

    def decorator(training_function):

        @wraps(training_function)
        def wrapper(*args, **kwargs):

            # Configure local MLflow tracking.
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Select or create the experiment.
            mlflow.set_experiment(experiment_name)

            print("\n" + "=" * 60)
            print("MLFLOW TRACKING")
            print("=" * 60)

            print("Experiment:", experiment_name)
            print("Run name:", run_name)
            print("Tracking URI:", MLFLOW_TRACKING_URI)

            # Start a new run.
            with mlflow.start_run(run_name=run_name):

                # Log model hyperparameters and metadata.
                params_to_log = {
                    **model_params,
                    "feature_count": 82,
                    "preprocessing_scaled": False,
                }

                mlflow.log_params(params_to_log)

                # Log descriptive tags.
                mlflow.set_tags(tags)

                # Execute the original training function.
                result = training_function(*args, **kwargs)

                # Both training functions return metrics last.
                metrics = result[-1]

                # Log validation metrics.
                mlflow.log_metrics({
                    metric_name: float(metric_value)
                    for metric_name, metric_value in metrics.items()
                })

                mlflow.set_tag(
                    "training_status",
                    "completed",
                )

                print("\nMLflow run completed.")
                print("Run ID:", mlflow.active_run().info.run_id)
                print("Experiment:", experiment_name)

                return result

        return wrapper

    return decorator


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_dataset():
    """
    Load the modeling dataset and split manifest.

    Returns
    -------
    X : pd.DataFrame
        82 model features.

    y : pd.Series
        Original 12-class target.

    manifest : pd.DataFrame
        Original train/validation/test assignments.
    """

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {MODELING_DATA_PATH}"
        )

    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_MANIFEST_PATH}"
        )

    df = pd.read_csv(MODELING_DATA_PATH)

    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    required_manifest_columns = {
        "original_index",
        "split",
    }

    if not required_manifest_columns.issubset(manifest.columns):
        raise ValueError(
            "Split manifest must contain "
            "'original_index' and 'split'."
        )

    if manifest["original_index"].duplicated().any():
        raise ValueError(
            "Duplicate original_index values found."
        )

    if set(manifest["split"].unique()) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(
            "Unexpected split names in manifest."
        )

    indices = manifest["original_index"]

    if not indices.between(0, len(df) - 1).all():
        raise ValueError(
            "Split manifest contains out-of-range indices."
        )

    if len(manifest) != len(df):
        raise ValueError(
            "Split manifest and dataset have different row counts."
        )

    if set(indices) != set(range(len(df))):
        raise ValueError(
            "Split manifest does not cover every dataset row exactly."
        )

    X, y = prepare_labeled_data(df)

    print("\nDataset loaded successfully.")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nSplit distribution:")
    print(manifest["split"].value_counts())

    return X, y, manifest


# ============================================================
# 4. RECONSTRUCT TRAIN / VALIDATION / TEST SPLITS
# ============================================================

def create_splits(X, y, manifest):
    """
    Reconstruct the original splits using positional indices.

    Uses .iloc(), not .loc(), because original_index
    represents the row position in the modeling dataset.
    """

    train_indices = manifest.loc[
        manifest["split"] == "train",
        "original_index",
    ].to_numpy()

    val_indices = manifest.loc[
        manifest["split"] == "validation",
        "original_index",
    ].to_numpy()

    test_indices = manifest.loc[
        manifest["split"] == "test",
        "original_index",
    ].to_numpy()

    X_train = X.iloc[train_indices].copy()
    X_val = X.iloc[val_indices].copy()
    X_test = X.iloc[test_indices].copy()

    y_train = y.iloc[train_indices].copy()
    y_val = y.iloc[val_indices].copy()
    y_test = y.iloc[test_indices].copy()

    print("\nReconstructed split sizes:")
    print("Training:", len(X_train))
    print("Validation:", len(X_val))
    print("Test:", len(X_test))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# 5. BUILD BINARY TARGET
# ============================================================

def create_binary_target(y):
    """
    Convert the 12-class target into binary labels.

    Normal:
        MQTT_Publish
        Thing_Speak
        Wipro_bulb

    Attack:
        All remaining classes.
    """

    binary_y = np.where(
        y.isin(NORMAL_CLASSES),
        "Normal",
        "Attack",
    )

    return pd.Series(
        binary_y,
        index=y.index,
        name="Binary_Target",
    )


# ============================================================
# 6. EVALUATE VALIDATION PERFORMANCE
# ============================================================

def print_validation_metrics(
    y_true,
    y_pred,
    *,
    task_name,
):
    """
    Print validation metrics.

    The test partition is deliberately not used here.
    """

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"{task_name} - VALIDATION RESULTS")
    print("=" * 60)

    print(f"Accuracy:          {accuracy:.6f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.6f}")
    print(f"Macro F1:          {macro_f1:.6f}")
    print(f"Weighted F1:       {weighted_f1:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


# ============================================================
# 7. TRAIN BINARY RANDOM FOREST
# ============================================================

@track_experiment(
    experiment_name=BINARY_EXPERIMENT_NAME,
    run_name="binary_random_forest",
    model_params=BINARY_RF_PARAMS,
    tags={
        "task": "binary_classification",
        "model_type": "RandomForestClassifier",
        "dataset": "RT-IoT2022",
        "target": "Normal_vs_Attack",
    },
)
def train_binary_model():
    """
    Train the tuned binary Random Forest.

    Preprocessing and model are fitted together
    in one sklearn Pipeline.

    Returns
    -------
    Pipeline
        Fitted binary classification pipeline.
    """

    print("\n" + "=" * 60)
    print("BINARY CLASSIFICATION TRAINING")
    print("=" * 60)

    X, y, manifest = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_splits(X, y, manifest)

    y_train_binary = create_binary_target(y_train)
    y_val_binary = create_binary_target(y_val)

    print("\nTraining target distribution:")
    print(y_train_binary.value_counts())

    preprocessor = build_preprocessor(
        scaled=False
    )

    classifier = RandomForestClassifier(
        **BINARY_RF_PARAMS
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    print("\nFitting binary Random Forest...")

    model_pipeline.fit(
        X_train,
        y_train_binary,
    )

    y_val_pred = model_pipeline.predict(X_val)

    metrics = print_validation_metrics(
        y_val_binary,
        y_val_pred,
        task_name="Binary Random Forest",
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_pipeline,
        BINARY_MODEL_PATH,
    )

    print("\nBinary model saved:")
    print(BINARY_MODEL_PATH)

    print(
        "\nNote: The test partition was not used "
        "for training or model selection."
    )

    return model_pipeline, metrics


# ============================================================
# 8. TRAIN MULTICLASS WEIGHTED XGBOOST
# ============================================================

@track_experiment(
    experiment_name=MULTICLASS_EXPERIMENT_NAME,
    run_name="multiclass_weighted_xgboost",
    model_params=MULTICLASS_XGB_PARAMS,
    tags={
        "task": "multiclass_classification",
        "model_type": "XGBClassifier",
        "dataset": "RT-IoT2022",
        "target": "Attack_type",
        "class_weighting": "balanced_sample_weights",
    },
)
def train_multiclass_model():
    """
    Train weighted multiclass XGBoost.

    LabelEncoder is fitted using training labels only.
    Class weights are calculated from training labels only.

    Returns
    -------
    model_pipeline : Pipeline
        Fitted preprocessing + XGBoost pipeline.

    label_encoder : LabelEncoder
        Fitted target encoder.

    metrics : dict
        Validation metrics.
    """

    print("\n" + "=" * 60)
    print("MULTICLASS CLASSIFICATION TRAINING")
    print("=" * 60)

    X, y, manifest = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_splits(X, y, manifest)

    # Encode target labels using training labels only.
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(
        y_train
    )

    y_val_encoded = label_encoder.transform(
        y_val
    )

    print("\nTarget classes:")
    print(label_encoder.classes_)

    # Calculate class weights from training labels only.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    print("\nTraining sample weights calculated.")

    preprocessor = build_preprocessor(
        scaled=False
    )

    xgb_classifier = XGBClassifier(
        **MULTICLASS_XGB_PARAMS
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb_classifier),
        ]
    )

    print("\nFitting weighted XGBoost...")

    model_pipeline.fit(
        X_train,
        y_train_encoded,
        model__sample_weight=sample_weights,
    )

    y_val_pred_encoded = model_pipeline.predict(X_val)

    y_val_pred = label_encoder.inverse_transform(
        y_val_pred_encoded.astype(int)
    )

    metrics = print_validation_metrics(
        y_val,
        y_val_pred,
        task_name="Weighted Multiclass XGBoost",
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model_pipeline,
        MULTICLASS_MODEL_PATH,
    )

    joblib.dump(
        label_encoder,
        MULTICLASS_ENCODER_PATH,
    )

    print("\nMulticlass model saved:")
    print(MULTICLASS_MODEL_PATH)

    print("\nLabel encoder saved:")
    print(MULTICLASS_ENCODER_PATH)

    print(
        "\nNote: The test partition was not used "
        "for training or model selection."
    )

    return model_pipeline, label_encoder, metrics


# ============================================================
# 9. COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SentinelML model training"
    )

    parser.add_argument(
        "task",
        choices=[
            "binary",
            "multiclass",
            "all",
        ],
        help="Choose which training workflow to execute.",
    )

    args = parser.parse_args()

    if args.task in ["binary", "all"]:
        train_binary_model()

    if args.task in ["multiclass", "all"]:
        train_multiclass_model()


if __name__ == "__main__":
    main()
