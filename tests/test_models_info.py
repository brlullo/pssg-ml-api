def test_list_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    assert any(m["name"] == "upror_growing_rod" for m in models)


def test_get_model_detail(client):
    response = client.get("/models/upror_growing_rod")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "upror_growing_rod"
    assert data["algorithm"] == "Logistic Regression"
    assert len(data["features"]) == 8
    assert "test_auc" in data["metrics"]


def test_get_model_not_found(client):
    response = client.get("/models/nonexistent_model")
    assert response.status_code == 404


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["models_loaded"] >= 1


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "PSSG ML API"
    assert "docs_url" in data
