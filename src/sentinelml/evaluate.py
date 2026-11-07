
"""
SentinelML - Model Evaluation Module

Evaluates saved binary and multiclass pipelines
on the reserved test partition.

Outputs:
- JSON evaluation metrics
- CSV classification reports
- CSV confusion matrices
"""

import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.sentinelml.config import (
    ARTIFACTS_DIR,
    METRICS_DIR,
    MULTICLASS_MODEL_PATH,
    MULTICLASS_ENCODER_PATH,
)

from src.sentinelml.train import (
    BINARY_MODEL_PATH,
    create_binary_target,
    load_dataset,
    create_splits,
)


# ============================================================
# 1. EVALUATION HELPER
# ============================================================

def calculate_metrics(y_true, y_pred):
    """Calculate the main classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_evaluation_outputs(
    task_name,
    y_true,
    y_pred,
    labels,
):
    """Save metrics, classification report and confusion matrix."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    # Save metrics as JSON.
    metrics_path = (
        METRICS_DIR / f"{task_name}_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # Save classification report.
    report_path = (
        METRICS_DIR / f"{task_name}_classification_report.csv"
    )

    pd.DataFrame(report).transpose().to_csv(
        report_path,
        index=True,
    )

    # Save confusion matrix.
    matrix_path = (
        METRICS_DIR / f"{task_name}_confusion_matrix.csv"
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(matrix_path)

    print("\n" + "=" * 65)
    print(f"{task_name.upper()} - TEST RESULTS")
    print("=" * 65)

    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.6f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )
    )

    print("\nSaved evaluation files:")
    print(metrics_path)
    print(report_path)
    print(matrix_path)

    return metrics


# ============================================================
# 2. BINARY MODEL EVALUATION
# ============================================================

def evaluate_binary_model(X_test, y_test):
    """Evaluate the saved binary Random Forest pipeline."""

    print("\nLoading binary model...")

    if not BINARY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Binary model not found: {BINARY_MODEL_PATH}"
        )

    model = joblib.load(
        BINARY_MODEL_PATH
    )

    y_test_binary = create_binary_target(
        y_test
    )

    y_pred = model.predict(
        X_test
    )

    labels = [
        "Normal",
        "Attack",
    ]

    return save_evaluation_outputs(
        task_name="binary",
        y_true=y_test_binary,
        y_pred=y_pred,
        labels=labels,
    )


# ============================================================
# 3. MULTICLASS MODEL EVALUATION
# ============================================================

def evaluate_multiclass_model(X_test, y_test):
    """Evaluate the saved weighted XGBoost pipeline."""

    print("\nLoading multiclass model...")

    if not MULTICLASS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Multiclass model not found: {MULTICLASS_MODEL_PATH}"
        )

    if not MULTICLASS_ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {MULTICLASS_ENCODER_PATH}"
        )

    model = joblib.load(
        MULTICLASS_MODEL_PATH
    )

    label_encoder = joblib.load(
        MULTICLASS_ENCODER_PATH
    )

    y_pred_encoded = model.predict(
        X_test
    )

    y_pred = label_encoder.inverse_transform(
        y_pred_encoded.astype(int)
    )

    labels = label_encoder.classes_.tolist()

    return save_evaluation_outputs(
        task_name="multiclass",
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
    )


# ============================================================
# 4. MAIN EVALUATION WORKFLOW
# ============================================================

def evaluate_all():
    """Evaluate both saved models on the same test partition."""

    print("=" * 65)
    print("SENTINELML - MODEL EVALUATION")
    print("=" * 65)

    X, y, manifest = load_dataset()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_splits(
        X,
        y,
        manifest,
    )

    print("\nTest partition size:", len(X_test))

    binary_metrics = evaluate_binary_model(
        X_test,
        y_test,
    )

    multiclass_metrics = evaluate_multiclass_model(
        X_test,
        y_test,
    )

    # Combined summary.
    summary = {
        "binary": binary_metrics,
        "multiclass": multiclass_metrics,
        "evaluation_partition": "test",
        "test_samples": len(X_test),
        "test_partition_limitation": (
            "Test results were inspected in earlier notebook "
            "experiments; this is not a pristine first-use holdout."
        ),
    }

    summary_path = (
        METRICS_DIR / "evaluation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print("\nCombined summary saved:")
    print(summary_path)

    return summary


if __name__ == "__main__":
    evaluate_all()
