import json
import pandas as pd

from src.sentinelml.config import (
    MODELING_DATA_PATH,
    SPLIT_MANIFEST_PATH,
)
from src.sentinelml.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    CONSTANT_FEATURES,
)

# 1. Load the modeling dataset and split manifest
df = pd.read_csv(MODELING_DATA_PATH)
manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

# 2. Select the first row belonging to the test split
test_positions = manifest.loc[
    manifest["split"] == "test", "original_index"
].tolist()

sample_position = int(test_positions[0])
sample = df.iloc[sample_position]

# 3. Separate the true label and model features
actual_label = sample["Attack_type"]

features = (
    sample
    .drop(labels=["Attack_type"] + CONSTANT_FEATURES)
    .reindex(MODEL_FEATURE_COLUMNS)
)

# 4. Build the API request payload
payload = {
    "features": {
        column: (
            value.item() if hasattr(value, "item") else value
        )
        for column, value in features.items()
    }
}

# 5. Save the payload
output_path = "data/processed/api_test_sample.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=4)

print("API test sample created successfully.")
print("Sample position:", sample_position)
print("Actual multiclass label:", actual_label)
print("Number of features:", len(payload["features"]))
print("Saved to:", output_path)