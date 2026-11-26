from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
PREDICTIONS_PATH = ROOT / "predictions" / "holdout_predictions.csv"

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


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), "Champion model file missing"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def holdout():
    assert HOLDOUT_PATH.exists(), "Holdout input file missing"
    return pd.read_csv(HOLDOUT_PATH)


def test_model_loads(model):
    assert hasattr(model, "predict")


def test_required_features_exist(holdout):
    assert all(col in holdout.columns for col in FEATURES)


def test_valid_prediction(model, holdout):
    X = holdout[FEATURES].head(5)
    predictions = np.asarray(model.predict(X))

    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_missing_numeric_values(model, holdout):
    X = holdout[FEATURES].head(1).copy()

    X.loc[X.index[0], "rider_age"] = np.nan
    X.loc[X.index[0], "wait_at_station_min"] = np.nan

    predictions = np.asarray(model.predict(X))

    assert len(predictions) == 1
    assert np.isfinite(predictions).all()


def test_unknown_category(model, holdout):
    X = holdout[FEATURES].head(1).copy()

    X.loc[X.index[0], "transport"] = "UNKNOWN_TRANSPORT"

    predictions = np.asarray(model.predict(X))

    assert len(predictions) == 1
    assert np.isfinite(predictions).all()


def test_holdout_predictions_file_exists():
    assert PREDICTIONS_PATH.exists()


def test_holdout_predictions_schema():
    df = pd.read_csv(PREDICTIONS_PATH)

    assert list(df.columns) == ["record_id", "prediction"]
    assert len(df) == 250
    assert df["record_id"].is_unique
    assert df["record_id"].notna().all()
    assert df["prediction"].notna().all()
    assert np.isfinite(df["prediction"].to_numpy()).all()


def test_holdout_prediction_ids_match(holdout):
    predictions = pd.read_csv(PREDICTIONS_PATH)

    assert predictions["record_id"].tolist() == (
        holdout["record_id"].tolist()
    )


def test_exported_model_matches_predictions(model, holdout):
    saved = pd.read_csv(PREDICTIONS_PATH)["prediction"].to_numpy()
    actual = np.asarray(model.predict(holdout[FEATURES]))

    np.testing.assert_allclose(actual, saved, rtol=1e-5, atol=1e-5)