from __future__ import annotations

from fastapi.testclient import TestClient


def test_api_catalogs_and_dataset_upload(db, demo_path):
    from app.main import app

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        pipelines = client.get("/api/v1/pipelines").json()
        models = client.get("/api/v1/models/catalog").json()
        assert any(item["id"] == "tabular_basic" for item in pipelines)
        assert len(models) == 5
        with demo_path.open("rb") as source:
            response = client.post(
                "/api/v1/datasets",
                data={"name": "Iris API", "data_type": "tabular"},
                files={"file": ("iris.csv", source, "text/csv")},
            )
        assert response.status_code == 201, response.text
        dataset_id = response.json()["id"]
        assert client.get(f"/api/v1/datasets/{dataset_id}").json()["versions"][0]["row_count"] == 30


def test_openapi_contains_required_routes():
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/api/v1/experiments/{experiment_id}/rerun" in paths
    assert "/api/v1/experiments/{experiment_id}/tracking" in paths
    assert "/api/v1/experiments/tracking-summary" in paths
    assert "/api/v1/datasets/{dataset_id}/versions" in paths
    assert "/api/v1/datasets/mounted" in paths
    assert "/api/v1/pipelines/{pipeline_id}/install" not in paths
    assert "/api/v1/registry/models/{name}/aliases" in paths
    assert "/api/v1/registry/models/{name}/versions/{version_or_alias}/input-schema" in paths
    assert "/api/v1/predictions/{prediction_id}" in paths
    assert "/api/v1/predictions/{prediction_id}/tracking" in paths
    assert "/api/v1/predictions/tracking-summary" in paths


def test_experiment_endpoint_awaits_prefect_dispatch(db, demo_dataset, monkeypatch):
    from app.main import app

    dispatched = []

    async def submit(experiment_id):
        dispatched.append(experiment_id)
        return "prefect-run"

    monkeypatch.setattr("app.api.v1.router.submit_experiment", submit)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/experiments",
            json={
                "name": "API dispatch",
                "dataset_id": demo_dataset.id,
                "pipeline_id": "tabular_basic",
                "pipeline_config": {"target_column": "species"},
                "models": [{"model_id": "logistic_regression", "parameters": {}}],
                "validation_strategy": "train_test_split",
                "validation_config": {"test_size": 0.2},
                "primary_metric": "f1_macro",
                "random_seed": 42,
            },
        )
    assert response.status_code == 202, response.text
    assert dispatched == [response.json()["id"]]
