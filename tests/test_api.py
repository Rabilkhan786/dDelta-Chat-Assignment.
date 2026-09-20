"""A smoke test for the optional local HTTP interface."""

from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint_is_available() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
