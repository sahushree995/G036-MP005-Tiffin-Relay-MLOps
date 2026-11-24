from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
VALID_PAYLOAD = {
    "distance_km": 6.9,
    "tiffins": 71,
    "handovers": 3,
    "wait_at_station_min": 22.1,
    "rider_age": 46,
    "temp_c": 22.4,
    "transport": "bicycle",
    "route_zone": "Suburb",
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_predict_returns_finite_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert set(response.json()) == {"prediction_minutes"}
    prediction = response.json()["prediction_minutes"]
    assert isinstance(prediction, float)
    assert prediction == round(prediction, 1)


def test_predict_rejects_too_many_tiffins():
    payload = {**VALID_PAYLOAD, "tiffins": 101}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_temperature_below_five():
    payload = {**VALID_PAYLOAD, "temp_c": 4.9}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_accepts_exact_validation_limits():
    payload = {**VALID_PAYLOAD, "tiffins": 100, "temp_c": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction_minutes" in response.json()


def test_predict_requires_exact_feature_set():
    payload = {key: value for key, value in VALID_PAYLOAD.items() if key != "route_zone"}
    assert client.post("/predict", json=payload).status_code == 422
