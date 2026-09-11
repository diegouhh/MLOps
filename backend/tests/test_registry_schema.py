from __future__ import annotations

from fastapi.testclient import TestClient

from app.db.models import ModelCandidate, RegisteredModelReference
from app.schemas.api import ExperimentCreate, WinnerRequest
from app.services.experiments import create_experiment
from app.services.registry import get_model_reference, reference_details, reference_input_schema


def test_registered_tabular_model_exposes_prediction_fields(db, demo_dataset):
    experiment = create_experiment(
        db,
        ExperimentCreate(
            name="Iris schema",
            dataset_id=demo_dataset.id,
            pipeline_id="tabular_basic",
            pipeline_config={"target_column": "species"},
            models=[{"model_id": "logistic_regression", "parameters": {}}],
        ),
    )
    run = experiment.runs[0]
    run.status = "completed"
    run.mlflow_run_id = "child-run"
    run.metrics = {"f1_macro": 0.96}
    run.candidate = ModelCandidate(
        model_id=run.model_id,
        artifact_uri="models:/m-test",
        is_winner=True,
    )
    db.commit()
    reference = RegisteredModelReference(
        name="iris-classifier",
        version="1",
        alias="champion",
        experiment_id=experiment.id,
        candidate_id=run.candidate.id,
        model_uri="models:/iris-classifier/1",
        metrics=run.metrics,
    )
    db.add(reference)
    db.commit()

    resolved = get_model_reference(db, "iris-classifier", "champion")
    schema = reference_input_schema(db, resolved)
    details = reference_details(db, resolved)

    assert [field["name"] for field in schema["fields"]] == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    assert all(field["type"] == "number" for field in schema["fields"])
    assert schema["target_column"] == "species"
    assert details["model_id"] == "logistic_regression"
    assert details["dataset_name"] == "Iris demo"

    from app.main import app

    with TestClient(app) as client:
    response = client.get(
        "/api/v1/registry/models/iris-classifier/versions/champion/input-schema"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["example"] == {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    assert payload["examples"]
    assert payload["examples"][0] == payload["example"]


def test_registered_model_name_accepts_readable_names():
    payload = WinnerRequest(
        training_run_id="run-id",
        alias="champion",
        model_name="iris-classifier v1",
    )
    assert payload.model_name == "iris-classifier v1"
