"""Calculate feature drift and evaluate model promotion on later-period data."""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.tiffin_mlops.train import FEATURES, NUMERIC_FEATURES, TARGET

PROMOTION_RATIO = 0.92
REGISTERED_MODEL_NAME = "MP005_Tiffin_Relay_Model"


def population_stability_index(reference, current, bins=10):
    """Return PSI using reference quantile bins and a small zero-count floor."""
    reference = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if not len(reference) or not len(current):
        raise ValueError("PSI requires nonempty numeric samples")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        # Constant reference values still need two finite bounds.
        edges = np.array([reference[0] - 0.5, reference[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    floor = 1e-6
    expected = np.clip(expected, floor, None)
    actual = np.clip(actual, floor, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def evaluate(reference_path, later_path, v1_model, v2_model, output_path):
    reference = pd.read_csv(reference_path)
    later = pd.read_csv(later_path)
    required = set(FEATURES + [TARGET, "route_zone", "date"])
    missing = sorted(required - set(later.columns))
    if missing:
        raise ValueError(f"later dataset is missing required columns: {missing}")

    later["date"] = pd.to_datetime(later["date"], errors="raise")
    later = later.sort_values(["date", "record_id"], kind="stable").reset_index(drop=True)
    if len(later) < 2:
        raise ValueError("later dataset needs at least two rows")

    psi = {
        feature: population_stability_index(reference[feature], later[feature])
        for feature in NUMERIC_FEATURES
    }
    v1_all_predictions = v1_model.predict(later[FEATURES])
    zone_mae = {}
    for zone, group in later.assign(_v1_prediction=v1_all_predictions).groupby("route_zone", dropna=False):
        key = "<missing>" if pd.isna(zone) else str(zone)
        zone_mae[key] = float(mean_absolute_error(group[TARGET], group["_v1_prediction"]))

    holdout_start = int(len(later) * 0.70)
    holdout = later.iloc[holdout_start:]
    if holdout.empty:
        raise ValueError("last 30% holdout is empty")
    y_true = holdout[TARGET]
    v1_mae = float(mean_absolute_error(y_true, v1_model.predict(holdout[FEATURES])))
    v2_mae = float(mean_absolute_error(y_true, v2_model.predict(holdout[FEATURES])))
    required_v2_mae = PROMOTION_RATIO * v1_mae
    improvement_pct = ((v1_mae - v2_mae) / v1_mae * 100) if v1_mae else 0.0
    approved = v2_mae <= required_v2_mae

    report = {
        "data": {
            "reference_file": str(reference_path),
            "later_file": str(later_path),
            "later_rows": len(later),
            "holdout_rows": len(holdout),
            "holdout_start_date": holdout["date"].min().date().isoformat(),
            "holdout_end_date": holdout["date"].max().date().isoformat(),
        },
        "psi": psi,
        "v1_mae_by_route_zone_minutes": zone_mae,
        "promotion_evaluation": {
            "v1_last_30pct_mae_minutes": v1_mae,
            "v2_last_30pct_mae_minutes": v2_mae,
            "required_improvement_pct": 8.0,
            "observed_improvement_pct": improvement_pct,
            "maximum_v2_mae_minutes": required_v2_mae,
            "promotion_approved": approved,
            "decision": "promote_v2" if approved else "retain_v1",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/processed/train_cleaned.csv")
    parser.add_argument("--later", default="data/raw/later.csv")
    parser.add_argument("--v1-model", help="V1 joblib path; defaults to MLflow version 1")
    parser.add_argument("--v2-model", default="models/model.joblib")
    parser.add_argument("--output", default="reports/ops_monitoring.json")
    args = parser.parse_args()

    if not Path(args.later).exists():
        raise FileNotFoundError(
            f"{args.later} is unavailable. Restore it with DVC after configuring the dataset remote."
        )
    v2_model = joblib.load(args.v2_model)
    if args.v1_model:
        v1_model = joblib.load(args.v1_model)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        try:
            v1_model = mlflow.sklearn.load_model(
                f"models:/{REGISTERED_MODEL_NAME}/1"
            )
        except Exception:
            # MLflow 3 model versions may point to logged-model IDs. In a
            # checked-out local tracking store, load that model's artifact dir.
            version = MlflowClient().get_model_version(
                REGISTERED_MODEL_NAME, "1"
            )
            model_id = version.source.removeprefix("models:/")
            local_artifacts = Path("mlruns/1/models") / model_id / "artifacts"
            if not (local_artifacts / "MLmodel").exists():
                raise FileNotFoundError(
                    "Could not find the registered v1 model locally; pass --v1-model."
                )
            v1_model = mlflow.sklearn.load_model(str(local_artifacts))

    report = evaluate(args.reference, args.later, v1_model, v2_model, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
