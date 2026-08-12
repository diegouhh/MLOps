from __future__ import annotations

from sqlalchemy import select

from app.db.models import RegisteredModelReference
from app.schemas.api import ExperimentCreate
from app.services.experiments import create_experiment, execute_experiment, get_experiment
from app.services.predictions import create_prediction_job, execute_prediction


def test_end_to_end_three_models_registry_and_prediction(db, settings, demo_dataset):
    payload = ExperimentCreate(
        name="E2E Iris",
        dataset_id=demo_dataset.id,
        pipeline_id="tabular_basic",
        pipeline_config={"target_column": "species"},
        models=[
            {"model_id": "logistic_regression", "parameters": {"max_iter": 500}},
            {"model_id": "random_forest", "parameters": {"n_estimators": 30}},
            {"model_id": "svm", "parameters": {"C": 1.0}},
        ],
        validation_strategy="train_test_split",
        validation_config={"test_size": 0.3},
        primary_metric="f1_macro",
        random_seed=42,
    )
    experiment = create_experiment(db, payload)
    result = execute_experiment(db, settings, experiment.id)
    assert result.status == "completed"
    assert len(result.runs) == 3
    assert all(run.status == "completed" and "f1_macro" in run.metrics for run in result.runs)
    assert result.winner_candidate_id
    reference = db.scalar(
        select(RegisteredModelReference).where(RegisteredModelReference.experiment_id == result.id)
    )
    assert reference and reference.alias == "champion"
    job = create_prediction_job(
        db,
        model_name=reference.name,
        version_or_alias="champion",
        records=[
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
        ],
    )
    prediction = execute_prediction(db, settings, job.id)
    assert prediction.status == "completed"
    assert len(prediction.result_payload["predictions"]) == 2
    persisted = get_experiment(db, experiment.id)
    assert persisted.mlflow_parent_run_id
