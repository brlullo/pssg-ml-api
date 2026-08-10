VALID_PAYLOAD = {
    "coronal_deformity": 73.0,
    "max_kyphosis": 52.0,
    "age_years": 6.5,
    "gender": "Male",
    "race": "White",
    "weight_kg": 18.3,
    "etiology": "Idiopathic",
    "ambulatory_status": "Ambulatory",
    "pelvic_attachments": "No",
    "rod_number": "Bilateral",
    "device_type": "VEPTR/TGR",
}


def test_predict_hgt_valid(client):
    response = client.post("/predict/hgt_complication", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "hgt_complication"
    labels = [s["label"] for s in data["scenarios"]]
    assert labels == ["Without HGT", "With HGT"]
    for s in data["scenarios"]:
        assert 0.0 < s["probability"] < 1.0
        assert s["ci_lower"] <= s["probability"] <= s["ci_upper"]


def test_predict_hgt_bad_categorical(client):
    payload = dict(VALID_PAYLOAD, gender="Malee")
    response = client.post("/predict/hgt_complication", json=payload)
    assert response.status_code == 422


def test_predict_hgt_missing_field(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age_years"}
    response = client.post("/predict/hgt_complication", json=payload)
    assert response.status_code == 422


def test_both_models_listed(client):
    names = {m["name"] for m in client.get("/models").json()}
    assert {"upror_growing_rod", "hgt_complication"} <= names


def test_hgt_detail_has_prediction_type_and_options(client):
    data = client.get("/models/hgt_complication").json()
    assert data["prediction_type"] == "scenario_comparison"
    gender = next(f for f in data["features"] if f["name"] == "gender")
    assert gender["options"] == ["Male", "Female"]


def test_upror_detail_defaults_binary_threshold(client):
    data = client.get("/models/upror_growing_rod").json()
    assert data["prediction_type"] == "binary_threshold"
