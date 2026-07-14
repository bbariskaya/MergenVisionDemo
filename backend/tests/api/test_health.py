import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def test_live_returns_alive(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_returns_200_when_healthy(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    names = {c["name"] for c in data["components"]}
    assert names == {"postgres", "minio", "qdrant"}
    for c in data["components"]:
        assert c["status"] == "ok"


def test_ready_returns_503_on_failure(monkeypatch, client: TestClient) -> None:
    from app.services.readiness import ComponentStatus, ReadinessService

    service: ReadinessService = client.app.state.readiness_service

    async def failing_check():
        return (
            [
                ComponentStatus(name="postgres", status="ok"),
                ComponentStatus(name="minio", status="unavailable"),
                ComponentStatus(name="qdrant", status="ok"),
            ],
            False,
        )

    monkeypatch.setattr(service, "check", failing_check)

    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    minio_status = next(c for c in data["components"] if c["name"] == "minio")
    assert minio_status["status"] == "unavailable"
    for key in ("traceback", "exception", "password", "secret", "endpoint"):
        assert key not in data


def test_ready_adapter_level_minio_unavailable(monkeypatch, client: TestClient) -> None:
    service = client.app.state.readiness_service

    async def unhealthy():
        return False

    monkeypatch.setattr(service._storage, "health_check", unhealthy)

    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    minio = next(c for c in data["components"] if c["name"] == "minio")
    assert minio["status"] == "unavailable"
    for key in ("traceback", "exception", "password", "secret", "endpoint"):
        assert key not in data


def test_ready_adapter_level_qdrant_unavailable(
    monkeypatch, client: TestClient
) -> None:
    service = client.app.state.readiness_service

    async def unhealthy():
        return False

    monkeypatch.setattr(service._vector_store, "health_check", unhealthy)

    response = client.get("/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    qdrant = next(c for c in data["components"] if c["name"] == "qdrant")
    assert qdrant["status"] == "unavailable"
