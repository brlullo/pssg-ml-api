import sqlite3

from app.config import settings


def test_usage_open_when_no_key_configured(client):
    response = client.get("/admin/usage")
    assert response.status_code == 200
    assert "total_requests" in response.json()


def test_usage_requires_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_API_KEY", "secret123")

    assert client.get("/admin/usage").status_code == 401
    assert (
        client.get("/admin/usage", headers={"X-API-Key": "wrong"}).status_code == 401
    )

    response = client.get("/admin/usage", headers={"X-API-Key": "secret123"})
    assert response.status_code == 200
    assert "total_requests" in response.json()


def test_failed_predict_requests_are_logged(client):
    """The middleware must log non-200 responses with their real status code."""
    response = client.post("/predict/nonexistent_model", json={})
    assert response.status_code == 404

    conn = sqlite3.connect(settings.LOG_DB_PATH)
    try:
        rows = conn.execute(
            "SELECT model_name, endpoint, status_code FROM request_logs"
        ).fetchall()
    finally:
        conn.close()

    assert ("nonexistent_model", "/predict/nonexistent_model", 404) in rows


def test_successful_predict_requests_are_logged(client):
    features = {
        "Post-Index_Inferior_Attachments_Spine_Based": 1,
        "Pre-Index_Height_cm": 110.0,
        "Post-Index_Device_Type_VEPTR": 0,
        "Post-Index_Device_Type_TGR": 1,
        "Pre-Index_Kyphosis": 45.0,
        "age_at_surgery": 5.2,
        "Post-Index_Number_of_Superior_Anchors": 4,
        "Gender_Male": 1,
    }
    response = client.post("/predict/upror_growing_rod", json=features)
    assert response.status_code == 200

    conn = sqlite3.connect(settings.LOG_DB_PATH)
    try:
        row = conn.execute(
            "SELECT model_name, status_code, response_time_ms FROM request_logs"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "upror_growing_rod"
    assert row[1] == 200
    assert row[2] >= 0
