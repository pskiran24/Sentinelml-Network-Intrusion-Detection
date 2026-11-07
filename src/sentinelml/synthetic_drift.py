
"""
SentinelML - Controlled Synthetic Data Drift

Creates an artificial change in selected network traffic
features and uses Evidently to detect the introduced drift.

Reference data:
    Original training partition.

Current data:
    Test partition with deliberately modified features.

This is a controlled simulation, not live network traffic.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

from src.sentinelml.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    prepare_labeled_data,
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELING_DATA_PATH = (
    PROJECT_ROOT / "data" / "interim" / "rt_iot2022_modeling.csv"
)

SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
)

MONITORING_DIR = (
    PROJECT_ROOT / "reports" / "monitoring"
)

REPORT_PATH = (
    MONITORING_DIR / "synthetic_data_drift_report.html"
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

RANDOM_STATE = 42

SAMPLE_SIZE = 10000

NUMERICAL_FEATURE = "fwd_iat.min"

CATEGORICAL_FEATURE = "proto"

NUMERICAL_MULTIPLIER = 5

CATEGORICAL_CHANGE_RATIO = 0.30


# ---------------------------------------------------------
# LOAD TRAINING AND TEST DATA
# ---------------------------------------------------------

def load_monitoring_data():
    """Load and reconstruct the original training and test splits."""

    print("\nLoading dataset and split manifest...")

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {MODELING_DATA_PATH}"
        )

    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {SPLIT_MANIFEST_PATH}"
        )

    dataset = pd.read_csv(MODELING_DATA_PATH)

    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    if len(dataset) != len(manifest):
        raise ValueError(
            "Dataset and split manifest have different row counts."
        )

    if manifest["original_index"].duplicated().any():
        raise ValueError(
            "Duplicate original indices found in split manifest."
        )

    if not manifest["original_index"].between(
        0, len(dataset) - 1
    ).all():
        raise ValueError(
            "Invalid positional indices in split manifest."
        )

    train_positions = manifest.loc[
        manifest["split"] == "train",
        "original_index"
    ].to_numpy()

    test_positions = manifest.loc[
        manifest["split"] == "test",
        "original_index"
    ].to_numpy()

    train_data = dataset.iloc[
        train_positions
    ].reset_index(drop=True)

    test_data = dataset.iloc[
        test_positions
    ].reset_index(drop=True)

    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")

    return train_data, test_data


# ---------------------------------------------------------
# PREPARE FEATURE DATA
# ---------------------------------------------------------

def prepare_feature_data(train_data, test_data):
    """Separate features and prepare reference/current samples."""

    print("\nPreparing reference and current datasets...")

    X_train, _ = prepare_labeled_data(train_data)

    X_test, _ = prepare_labeled_data(test_data)

    reference_data = X_train[
        MODEL_FEATURE_COLUMNS
    ].copy()

    current_data = X_test[
        MODEL_FEATURE_COLUMNS
    ].copy()

    reference_data = reference_data.sample(
        n=min(SAMPLE_SIZE, len(reference_data)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    current_data = current_data.sample(
        n=min(SAMPLE_SIZE, len(current_data)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Reference shape: {reference_data.shape}")
    print(f"Current shape: {current_data.shape}")

    return reference_data, current_data


# ---------------------------------------------------------
# INTRODUCE SYNTHETIC DRIFT
# ---------------------------------------------------------

def introduce_synthetic_drift(current_data):
    """
    Artificially modify selected features.

    Numerical drift:
        Multiply non-zero fwd_iat.min values by 5.

    Categorical drift:
        Replace 30% of proto values with the most frequent
        protocol category already present in the data.
    """

    print("\nIntroducing synthetic drift...")

    synthetic_data = current_data.copy()

    rng = np.random.default_rng(RANDOM_STATE)

    # Validate selected features.
    required_features = [
        NUMERICAL_FEATURE,
        CATEGORICAL_FEATURE,
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in synthetic_data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Required features are missing: {missing_features}"
        )

    # -----------------------------------------------------
    # 1. Numerical drift
    # -----------------------------------------------------

    before_numeric = synthetic_data[
        NUMERICAL_FEATURE
    ].copy()

    nonzero_mask = synthetic_data[
        NUMERICAL_FEATURE
    ] > 0

    synthetic_data.loc[
        nonzero_mask,
        NUMERICAL_FEATURE
    ] *= NUMERICAL_MULTIPLIER

    print(f"\nNumerical feature: {NUMERICAL_FEATURE}")
    print(f"Multiplier: {NUMERICAL_MULTIPLIER}")
    print(
        "Non-zero records modified:",
        int(nonzero_mask.sum())
    )
    print(
        "Original mean:",
        round(before_numeric.mean(), 4)
    )
    print(
        "Modified mean:",
        round(
            synthetic_data[NUMERICAL_FEATURE].mean(),
            4
        )
    )

    # -----------------------------------------------------
    # 2. Categorical drift
    # -----------------------------------------------------

    before_categorical = synthetic_data[
        CATEGORICAL_FEATURE
    ].copy()

    most_frequent_category = (
        before_categorical.mode().iloc[0]
    )

    number_to_modify = int(
        len(synthetic_data) * CATEGORICAL_CHANGE_RATIO
    )

    selected_indices = rng.choice(
        synthetic_data.index.to_numpy(),
        size=number_to_modify,
        replace=False,
    )

    synthetic_data.loc[
        selected_indices,
        CATEGORICAL_FEATURE
    ] = most_frequent_category

    print(f"\nCategorical feature: {CATEGORICAL_FEATURE}")
    print(f"Replacement category: {most_frequent_category}")
    print(f"Records modified: {number_to_modify}")

    print("\nOriginal category distribution:")
    print(
        before_categorical.value_counts(normalize=True).head()
    )

    print("\nModified category distribution:")
    print(
        synthetic_data[
            CATEGORICAL_FEATURE
        ].value_counts(normalize=True).head()
    )

    return synthetic_data


# ---------------------------------------------------------
# GENERATE EVIDENTLY REPORT
# ---------------------------------------------------------

def generate_drift_report(reference_data, current_data):
    """Generate and save the synthetic drift report."""

    print("\nGenerating synthetic drift report...")

    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    MONITORING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(str(REPORT_PATH))

    print("\nSynthetic drift report generated successfully.")
    print(f"Report location: {REPORT_PATH}")

    return snapshot


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("SENTINELML - SYNTHETIC DATA DRIFT EXPERIMENT")
    print("=" * 65)

    # Step 1: Load original partitions.
    train_data, test_data = load_monitoring_data()

    # Step 2: Prepare reference and current samples.
    reference_data, current_data = prepare_feature_data(
        train_data,
        test_data,
    )

    # Step 3: Introduce controlled drift.
    synthetic_current_data = introduce_synthetic_drift(
        current_data
    )

    # Step 4: Generate report.
    generate_drift_report(
        reference_data,
        synthetic_current_data,
    )

    print("\n" + "=" * 65)
    print("SYNTHETIC DRIFT EXPERIMENT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
