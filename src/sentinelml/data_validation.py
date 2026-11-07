
"""
SentinelML - RT-IoT2022 Data Validation

Validates incoming datasets before model training,
evaluation, and inference.
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa

from pandera import Check


# ============================================================
# 1. DATASET CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attack_type"

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
]


# ============================================================
# 2. EXPECTED DATASET COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "id.orig_p",
    "id.resp_p",
    "proto",
    "service",
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_data_pkts_tot",
    "bwd_data_pkts_tot",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "down_up_ratio",
    "fwd_header_size_tot",
    "fwd_header_size_min",
    "fwd_header_size_max",
    "bwd_header_size_tot",
    "bwd_header_size_min",
    "bwd_header_size_max",
    "flow_FIN_flag_count",
    "flow_SYN_flag_count",
    "flow_RST_flag_count",
    "fwd_PSH_flag_count",
    "bwd_PSH_flag_count",
    "flow_ACK_flag_count",
    "fwd_URG_flag_count",
    "bwd_URG_flag_count",
    "flow_CWR_flag_count",
    "flow_ECE_flag_count",
    "fwd_pkts_payload.min",
    "fwd_pkts_payload.max",
    "fwd_pkts_payload.tot",
    "fwd_pkts_payload.avg",
    "fwd_pkts_payload.std",
    "bwd_pkts_payload.min",
    "bwd_pkts_payload.max",
    "bwd_pkts_payload.tot",
    "bwd_pkts_payload.avg",
    "bwd_pkts_payload.std",
    "flow_pkts_payload.min",
    "flow_pkts_payload.max",
    "flow_pkts_payload.tot",
    "flow_pkts_payload.avg",
    "flow_pkts_payload.std",
    "fwd_iat.min",
    "fwd_iat.max",
    "fwd_iat.tot",
    "fwd_iat.avg",
    "fwd_iat.std",
    "bwd_iat.min",
    "bwd_iat.max",
    "bwd_iat.tot",
    "bwd_iat.avg",
    "bwd_iat.std",
    "flow_iat.min",
    "flow_iat.max",
    "flow_iat.tot",
    "flow_iat.avg",
    "flow_iat.std",
    "payload_bytes_per_second",
    "fwd_subflow_pkts",
    "bwd_subflow_pkts",
    "fwd_subflow_bytes",
    "bwd_subflow_bytes",
    "fwd_bulk_bytes",
    "bwd_bulk_bytes",
    "fwd_bulk_packets",
    "bwd_bulk_packets",
    "fwd_bulk_rate",
    "bwd_bulk_rate",
    "active.min",
    "active.max",
    "active.tot",
    "active.avg",
    "active.std",
    "idle.min",
    "idle.max",
    "idle.tot",
    "idle.avg",
    "idle.std",
    "fwd_init_window_size",
    "bwd_init_window_size",
    "fwd_last_window_size",
    TARGET_COLUMN,
]


# Derive numerical feature names automatically.
NUMERIC_COLUMNS = [
    col
    for col in EXPECTED_COLUMNS
    if col not in CATEGORICAL_COLUMNS + [TARGET_COLUMN]
]


# ============================================================
# 3. EXPECTED TARGET LABELS
# ============================================================

EXPECTED_TARGET_LABELS = {
    "ARP_poisioning",
    "DDOS_Slowloris",
    "DOS_SYN_Hping",
    "MQTT_Publish",
    "Metasploit_Brute_Force_SSH",
    "NMAP_FIN_SCAN",
    "NMAP_OS_DETECTION",
    "NMAP_TCP_scan",
    "NMAP_UDP_SCAN",
    "NMAP_XMAS_TREE_SCAN",
    "Thing_Speak",
    "Wipro_bulb",
}


# ============================================================
# 4. REUSABLE VALIDATION CHECKS
# ============================================================

def is_numeric(series: pd.Series) -> bool:
    """Check whether a column contains numerical values."""
    return pd.api.types.is_numeric_dtype(series)


def contains_only_finite_values(series: pd.Series) -> bool:
    """Reject NaN and positive/negative infinity."""
    return np.isfinite(series).all()


def contains_nonempty_strings(series: pd.Series) -> bool:
    """Reject empty or whitespace-only categorical values."""
    return series.str.strip().ne("").all()


# ============================================================
# 5. DEFINE TRAINING / EVALUATION SCHEMA
# ============================================================

columns = {}

# Numerical feature validation
for col in NUMERIC_COLUMNS:

    columns[col] = pa.Column(
        checks=[
            Check(
                is_numeric,
                element_wise=False,
                error="Column must contain numerical values.",
            ),
            Check(
                contains_only_finite_values,
                element_wise=False,
                error="Column must contain only finite values.",
            ),
        ],
        nullable=False,
        coerce=False,
    )


# Categorical feature validation
for col in CATEGORICAL_COLUMNS:

    columns[col] = pa.Column(
        str,
        checks=Check(
            contains_nonempty_strings,
            element_wise=False,
            error="Categorical values cannot be empty.",
        ),
        nullable=False,
        coerce=False,
    )


# Target validation
columns[TARGET_COLUMN] = pa.Column(
    str,
    checks=Check.isin(EXPECTED_TARGET_LABELS),
    nullable=False,
    coerce=False,
)


# Complete training schema
RT_IOT_SCHEMA = pa.DataFrameSchema(
    columns=columns,
    checks=[
        Check(
            lambda df: set(df.columns) == set(EXPECTED_COLUMNS),
            error="Dataset contains missing or unexpected columns.",
        )
    ],
    strict=True,
    coerce=False,
)


# ============================================================
# 6. DEFINE INFERENCE SCHEMA
# ============================================================

# Inference data does not require Attack_type.

INFERENCE_COLUMNS = {
    col: schema
    for col, schema in RT_IOT_SCHEMA.columns.items()
    if col != TARGET_COLUMN
}


INFERENCE_SCHEMA = pa.DataFrameSchema(
    columns=INFERENCE_COLUMNS,
    strict=True,
    coerce=False,
)


# ============================================================
# 7. MAIN VALIDATION FUNCTION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    *,
    labeled: bool = True,
) -> pd.DataFrame:
    """
    Validate an RT-IoT2022 dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    labeled : bool, default=True
        True:
            Validate training/evaluation data.
            Attack_type must be present.

        False:
            Validate inference data.
            Attack_type is optional and removed if present.

    Returns
    -------
    pd.DataFrame
        Validated copy of the input dataframe.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    pandera.errors.SchemaErrors
        If one or more schema checks fail.
    """

    # Check input type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # Work on a copy to avoid modifying original data
    data = df.copy()

    # Check empty dataframe
    if data.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    # Training / evaluation validation
    if labeled:

        validated_data = RT_IOT_SCHEMA.validate(
            data,
            lazy=True,
        )

    # Inference validation
    else:

        # Remove target if it is accidentally supplied
        if TARGET_COLUMN in data.columns:
            data = data.drop(
                columns=[TARGET_COLUMN]
            )

        validated_data = INFERENCE_SCHEMA.validate(
            data,
            lazy=True,
        )

    return validated_data

