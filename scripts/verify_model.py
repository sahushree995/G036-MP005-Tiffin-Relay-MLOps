from pathlib import Path
import hashlib

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"
HOLDOUT_PATH = ROOT / "data/raw/holdout_features.csv"
PREDICTIONS_PATH = ROOT / "predictions/holdout_predictions.csv"

FEATURES = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
    "transport",
    "route_zone",
]


def verify_model():
    for path in [MODEL_PATH, HOLDOUT_PATH, PREDICTIONS_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    model = joblib.load(MODEL_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    saved = pd.read_csv(PREDICTIONS_PATH)

    if len(holdout) != 250 or len(saved) != 250:
        raise ValueError("Unexpected holdout record count")

    if saved["record_id"].isna().any():
        raise ValueError("Missing record IDs")

    if not saved["record_id"].is_unique:
        raise ValueError("Duplicate record IDs")

    if saved["record_id"].tolist() != holdout["record_id"].tolist():
        raise ValueError("Record ID mismatch")

    actual = np.asarray(model.predict(holdout[FEATURES]))

    if not np.isfinite(actual).all():
        raise ValueError("Invalid model predictions")

    np.testing.assert_allclose(
        actual,
        saved["prediction"].to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    checksum = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    print("MODEL VERIFICATION SUCCESSFUL")
    print(f"Verified predictions: {len(actual)}")
    print(f"Minimum prediction: {actual.min():.2f} minutes")
    print(f"Maximum prediction: {actual.max():.2f} minutes")
    print(f"Mean prediction: {actual.mean():.2f} minutes")
    print(f"Model SHA256: {checksum}")
    print("All saved predictions match the champion model.")


if __name__ == "__main__":
    verify_model()
