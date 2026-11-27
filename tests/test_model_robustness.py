from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

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
    return joblib.load(ROOT / "models" / "model.joblib")


@pytest.fixture(scope="module")
def samples():
    df = pd.read_csv(
        ROOT / "data" / "raw" / "holdout_features.csv"
    )
    return df[FEATURES].head(10).copy()


def test_prediction_consistency(model, samples):
    first = model.predict(samples)
    second = model.predict(samples)

    np.testing.assert_allclose(first, second)


def test_column_order_independence(model, samples):
    original = model.predict(samples)
    reordered = model.predict(samples[FEATURES[::-1]])

    np.testing.assert_allclose(original, reordered)


def test_batch_prediction_consistency(model, samples):
    batch = model.predict(samples)

    individual = np.array([
        model.predict(samples.iloc[[i]])[0]
        for i in range(len(samples))
    ])

    np.testing.assert_allclose(batch, individual)


def test_missing_categorical_values(model, samples):
    modified = samples.head(1).copy()
    modified["transport"] = np.nan
    modified["route_zone"] = np.nan

    predictions = model.predict(modified)

    assert len(predictions) == 1
    assert np.isfinite(predictions).all()


def test_extra_column_handling(model, samples):
    modified = samples.copy()
    modified["unused_feature"] = 999

    original = model.predict(samples)
    actual = model.predict(modified)

    np.testing.assert_allclose(original, actual)
