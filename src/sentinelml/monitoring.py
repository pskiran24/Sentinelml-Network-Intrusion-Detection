
"""
SentinelML - Data Drift Monitoring

Uses Evidently to compare reference network traffic
against current network traffic.

Reference data: Training partition
Current data: Test partition

This is an offline monitoring demonstration using
RT-IoT2022, not live network traffic.
"""

from pathlib import Path

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
    MONITORING_DIR / "data_drift_report.html"
)


# ---------------------------------------------------------
# MONITORING CONFIGURATION
# ---------------------------------------------------------

RANDOM_STATE = 42

REFERENCE_SAMPLE_SIZE = 10000
CURRENT_SAMPLE_SIZE = 10000


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_monitoring_data():
    """
    Load the modeling dataset and split manifest.

    Reconstruct the original training and test partitions
    using positional indices.
    """

    print("\nLoading monitoring datasets...")

    if not MODELING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {MODELING_DATA_PATH}"
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
            "Split manifest contains duplicate row indices."
        )

    if not manifest["original_index"].between(
        0, len(dataset) - 1
    ).all():
        raise ValueError(
            "Split manifest contains invalid positional indices."
        )

    # Reconstruct partitions using positional indexing.
    train_positions = manifest.loc[
        manifest["split"] == "train",
        "original_index"
    ].to_numpy()

    test_positions = manifest.loc[
        manifest["split"] == "test",
        "original_index"
    ].to_numpy()

    train_data = dataset.iloc[train_positions].reset_index(drop=True)

    test_data = dataset.iloc[test_positions].reset_index(drop=True)

    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")

    return train_data, test_data


# ---------------------------------------------------------
# PREPARE REFERENCE AND CURRENT DATA
# ---------------------------------------------------------

def prepare_monitoring_data(train_data, test_data):
    """
    Prepare feature-only datasets for drift detection.

    Reference:
        Training partition

    Current:
        Test partition

    The target column is excluded because this report
    focuses on feature distribution drift.
    """

    print("\nPreparing reference and current data...")

    # Use the existing project validation and preprocessing
    # preparation logic to maintain the expected feature schema.
    X_train, _ = prepare_labeled_data(train_data)

    X_test, _ = prepare_labeled_data(test_data)

    # Ensure exact feature order.
    reference_data = X_train[MODEL_FEATURE_COLUMNS].copy()

    current_data = X_test[MODEL_FEATURE_COLUMNS].copy()

    # Sample records for manageable monitoring execution.
    reference_data = reference_data.sample(
        n=min(REFERENCE_SAMPLE_SIZE, len(reference_data)),
        random_state=RANDOM_STATE,
    )

    current_data = current_data.sample(
        n=min(CURRENT_SAMPLE_SIZE, len(current_data)),
        random_state=RANDOM_STATE,
    )

    reference_data = reference_data.reset_index(drop=True)

    current_data = current_data.reset_index(drop=True)

    print(f"Reference sample shape: {reference_data.shape}")
    print(f"Current sample shape: {current_data.shape}")

    return reference_data, current_data


# ---------------------------------------------------------
# GENERATE DATA DRIFT REPORT
# ---------------------------------------------------------

def generate_drift_report(reference_data, current_data):
    """
    Generate an Evidently data drift report.

    The report compares numerical and categorical
    feature distributions.
    """

    print("\nGenerating Evidently data drift report...")

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

    print("\nData drift report generated successfully.")

    print(f"Report location: {REPORT_PATH}")

    return snapshot


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("SENTINELML - DATA DRIFT MONITORING")
    print("=" * 65)

    # Step 1: Load datasets.
    train_data, test_data = load_monitoring_data()

    # Step 2: Prepare reference and current datasets.
    reference_data, current_data = prepare_monitoring_data(
        train_data,
        test_data,
    )

    # Step 3: Generate monitoring report.
    generate_drift_report(
        reference_data,
        current_data,
    )

    print("\n" + "=" * 65)
    print("MONITORING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
