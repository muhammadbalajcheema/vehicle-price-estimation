from fastapi.testclient import TestClient
from app.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

def test_predict_endpoint():
    payload = {
        "year": 2019,
        "present_price": 8.5,
        "kms_driven": 32000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "predicted_price_lakhs" in body
        assert body["predicted_price_lakhs"] > 0

def test_predict_validation_error():
    bad_payload = {"year": 1800, "present_price": -5}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422