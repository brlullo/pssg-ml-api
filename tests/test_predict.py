VALID_FEATURES = {
    "Post-Index_Inferior_Attachments_Spine_Based": 1.0,
    "Pre-Index_Height_cm": 110.0,
    "Post-Index_Device_Type_VEPTR": 0.0,
    "Post-Index_Device_Type_TGR": 1.0,
    "Pre-Index_Kyphosis": 45.0,
    "age_at_surgery": 6.5,
    "Post-Index_Number_of_Superior_Anchors": 4.0,
    "Gender_Male": 1.0,
}


def test_predict_valid(client):
    response = client.post(
        "/predict/upror_growing_rod", json=VALID_FEATURES
    )
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_name"] == "upror_growing_rod"
    assert data["prediction"] in (0, 1)


def test_predict_missing_features(client):
    incomplete = {"Post-Index_Inferior_Attachments_Spine_Based": 1.0}
    response = client.post(
        "/predict/upror_growing_rod", json=incomplete
    )
    assert response.status_code == 422


def test_predict_invalid_model(client):
    response = client.post(
        "/predict/nonexistent_model", json=VALID_FEATURES
    )
    assert response.status_code == 404


def test_predict_probability_range(client):
    """Test that different inputs produce different probabilities."""
    high_risk = VALID_FEATURES.copy()
    high_risk["age_at_surgery"] = 3.0
    high_risk["Pre-Index_Kyphosis"] = 80.0

    resp1 = client.post("/predict/upror_growing_rod", json=VALID_FEATURES)
    resp2 = client.post("/predict/upror_growing_rod", json=high_risk)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["probability"] != resp2.json()["probability"]
