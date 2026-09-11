from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.core.errors import ValidationError
from app.db.models import ModelCandidate, RegisteredModelReference
from app.schemas.api import ExperimentCreate
from app.services.datasets import create_dataset_version
from app.services.experiments import create_experiment, register_candidate, rerun_experiment
from app.services.jobs import submit_experiment
from app.services.predictions import create_prediction_job, execute_prediction


def _experiment(db, dataset_id: str):
    return create_experiment(
        db,
        ExperimentCreate(
            name="Regression",
            dataset_id=dataset_id,
            pipeline_id="tabular_basic",
            registered_model_name="iris-classifier",
            pipeline_config={"target_column": "species"},
            models=[{"model_id": "logistic_regression", "parameters": {}}],
        ),
    )


def _reference(db, experiment):
    run = experiment.runs[0]
    run.status = "completed"
    run.mlflow_run_id = "run-id"
    run.metrics = {"f1_macro": 0.9}
    run.candidate = ModelCandidate(model_id=run.model_id, artifact_uri="models:/m-test")
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
    return run, reference


def test_dataset_versions_are_real_and_experiments_stay_pinned(db, settings, demo_dataset):
    experiment = _experiment(db, demo_dataset.id)
    content = (
        b"sepal_length,sepal_width,petal_length,petal_width,species\n"
        b"5.1,3.5,1.4,0.2,setosa\n"
        b"6.7,3.0,5.2,2.3,virginica\n"
    )
    updated = create_dataset_version(
        db,
        settings,
        demo_dataset.id,
        filename="iris-v2.csv",
        source=io.BytesIO(content),
    )
    assert updated.current_version == 2
    assert [item.version for item in updated.versions] == [1, 2]
    assert experiment.dataset_version == 1
    assert rerun_experiment(db, experiment.id).dataset_version == 1


def test_prediction_payload_is_validated_before_queueing(db, demo_dataset):
    experiment = _experiment(db, demo_dataset.id)
    _run, reference = _reference(db, experiment)
    valid = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    job = create_prediction_job(
        db,
        model_name=reference.name,
        version_or_alias="champion",
        records=[valid],
    )
    assert job.status == "queued"
    assert job.resolved_model_version == "1"
    with pytest.raises(ValidationError, match="faltan campos"):
        create_prediction_job(
            db,
            model_name=reference.name,
            version_or_alias="champion",
            records=[{"sepal_length": 5.1}],
        )
    with pytest.raises(ValidationError, match="campos no esperados"):
        create_prediction_job(
            db,
            model_name=reference.name,
            version_or_alias="champion",
            records=[{**valid, "species": "setosa"}],
        )
    with pytest.raises(ValidationError, match="debe ser de tipo number"):
        create_prediction_job(
            db,
            model_name=reference.name,
            version_or_alias="champion",
            records=[{**valid, "sepal_length": "5.1"}],
        )


def test_failed_prediction_keeps_mlflow_run_id(
    db, settings, demo_dataset, monkeypatch
):
    experiment = _experiment(db, demo_dataset.id)
    _run, reference = _reference(db, experiment)
    job = create_prediction_job(
        db,
        model_name=reference.name,
        version_or_alias="champion",
        records=[
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        ],
    )

    class Active:
        class Info:
            run_id = "prediction-run"

        info = Info()

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

        def create_run(self, experiment_id, tags):
            return Active()

        def log_param(self, run_id, key, value):
            pass

        def set_tag(self, run_id, key, value):
            pass

        def set_terminated(self, run_id, status):
            pass

    monkeypatch.setattr("app.services.predictions._configure_mlflow", lambda settings: Client())
    monkeypatch.setattr(
        "app.services.predictions.mlflow.sklearn.load_model",
        lambda selector: (_ for _ in ()).throw(RuntimeError("modelo dañado")),
    )
    with pytest.raises(RuntimeError, match="modelo dañado"):
        execute_prediction(db, settings, job.id)
    db.refresh(job)
    assert job.status == "failed"
    assert job.mlflow_run_id == "prediction-run"
    assert "modelo dañado" in job.error_message


def test_registering_same_candidate_is_idempotent(
    db, settings, demo_dataset, monkeypatch
):
    experiment = _experiment(db, demo_dataset.id)
    run, reference = _reference(db, experiment)

    class Client:
        def set_registered_model_alias(self, name, alias, version):
            assert (name, alias, version) == ("iris-classifier", "challenger", "1")

        def create_model_version(self, **kwargs):
            raise AssertionError(f"No debe crear otra versión: {kwargs}")

    monkeypatch.setattr("app.services.experiments._configure_mlflow", lambda _settings: Client())
    result = register_candidate(
        db,
        settings,
        experiment,
        run,
        "challenger",
        "iris-classifier",
    )
    assert result.id == reference.id
    assert result.alias == "challenger"
    count = db.scalar(select(func.count()).select_from(RegisteredModelReference))
    assert count == 1


def test_service_health_rejects_non_success_status(monkeypatch):
    from app.api.v1.router import _service_status

    class Response:
        status_code = 404

    monkeypatch.setattr("app.api.v1.router.httpx.get", lambda *args, **kwargs: Response())
    result = _service_status("http://service")
    assert result["status"] == "unhealthy"
    assert result["status_code"] == 404


def test_experiment_is_dispatched_to_prefect_deployment(db, demo_dataset, monkeypatch):
    experiment = _experiment(db, demo_dataset.id)
    prefect_id = uuid4()

    async def dispatch(name, **kwargs):
        assert name == "NeuroOps experiment/neuroops-experiments"
        assert kwargs["parameters"] == {"experiment_id": experiment.id}
        assert kwargs["flow_run_name"] == experiment.name
        assert f"neuroops-experiment-id:{experiment.id}" in kwargs["tags"]
        assert kwargs["timeout"] == 0
        assert kwargs["idempotency_key"] == f"neuroops:experiment:{experiment.id}"
        return SimpleNamespace(id=prefect_id)

    monkeypatch.setattr("app.services.jobs.run_deployment", dispatch)
    assert asyncio.run(submit_experiment(experiment.id)) == str(prefect_id)
    db.expire_all()
    assert db.get(type(experiment), experiment.id).prefect_flow_run_id == str(prefect_id)
