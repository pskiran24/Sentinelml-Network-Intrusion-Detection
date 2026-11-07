
"""
SentinelML project configuration.

Centralized paths for datasets, artifacts,
reports, MLflow tracking, and project directories.
"""

from pathlib import Path


# ============================================================
# 1. PROJECT ROOT
# ============================================================

def find_project_root() -> Path:
    """
    Locate the SentinelML project root.

    Works independently of the current working directory.
    """

    current_file = Path(__file__).resolve()

    for parent in current_file.parents:
        if (
            (parent / "src" / "sentinelml").is_dir()
            and (parent / "notebooks").is_dir()
            and (parent / "data").is_dir()
        ):
            return parent

    raise FileNotFoundError(
        "Could not locate SentinelML project root."
    )


PROJECT_ROOT = find_project_root()


# ============================================================
# 2. MAIN PROJECT DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_DIR = REPORTS_DIR / "metrics"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

TESTS_DIR = PROJECT_ROOT / "tests"


# ============================================================
# 3. MLFLOW TRACKING CONFIGURATION
# ============================================================

# Use SQLite instead of the deprecated filesystem backend.
# No separate database server is required.

MLFLOW_DATABASE_PATH = PROJECT_ROOT / "mlflow.db"

# SQLAlchemy SQLite URI.
# as_posix() ensures forward slashes work on Windows.
MLFLOW_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DATABASE_PATH.resolve().as_posix()}"
)

BINARY_EXPERIMENT_NAME = "SentinelML_Binary_Classification"

MULTICLASS_EXPERIMENT_NAME = "SentinelML_Multiclass_Classification"


# ============================================================
# 4. DATASET FILE PATHS
# ============================================================

RAW_DATA_PATH = (
    RAW_DATA_DIR / "rt_iot2022.csv"
)

AMBIGUOUS_DATA_PATH = (
    INTERIM_DATA_DIR / "ambiguous_records.csv"
)

MODELING_DATA_PATH = (
    INTERIM_DATA_DIR / "rt_iot2022_modeling.csv"
)

SPLIT_MANIFEST_PATH = (
    PROCESSED_DATA_DIR / "split_manifest.csv"
)


# ============================================================
# 5. MODEL ARTIFACT PATHS
# ============================================================

BINARY_MODEL_PATH = (
    ARTIFACTS_DIR / "binary_tuned_random_forest.joblib"
)

MULTICLASS_MODEL_PATH = (
    ARTIFACTS_DIR / "multiclass_weighted_xgboost.joblib"
)

MULTICLASS_ENCODER_PATH = (
    ARTIFACTS_DIR / "multiclass_label_encoder.joblib"
)


# ============================================================
# 6. DIRECTORY INITIALIZATION
# ============================================================

def create_project_directories() -> None:
    """
    Create required project directories if they do not exist.
    """

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# 7. CONFIGURATION CHECK
# ============================================================

if __name__ == "__main__":

    create_project_directories()

    print("SentinelML project configuration loaded.")

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset paths:")
    print("Raw:", RAW_DATA_PATH)
    print("Modeling:", MODELING_DATA_PATH)
    print("Split manifest:", SPLIT_MANIFEST_PATH)

    print("\nModel artifact paths:")
    print("Binary:", BINARY_MODEL_PATH)
    print("Multiclass:", MULTICLASS_MODEL_PATH)
    print("Encoder:", MULTICLASS_ENCODER_PATH)

    print("\nMLflow configuration:")
    print("Database:", MLFLOW_DATABASE_PATH)
    print("Tracking URI:", MLFLOW_TRACKING_URI)
    print("Binary experiment:", BINARY_EXPERIMENT_NAME)
    print("Multiclass experiment:", MULTICLASS_EXPERIMENT_NAME)
