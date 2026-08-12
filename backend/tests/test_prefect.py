from __future__ import annotations

from app.orchestration.flows import (
    build_training_data_task,
    extract_features_task,
    load_data_task,
    preprocess_task,
    run_experiment_flow,
    train_candidate_task,
    validate_dataset_task,
)
from app.schemas.api import ExperimentCreate
from app.services.experiments import create_experiment, get_experiment


def test_prefect_flow_persists_visible_stages(db, demo_dataset):
    assert [
        validate_dataset_task.name,
        load_data_task.name,
        preprocess_task.name,
        extract_features_task.name,
        build_training_data_task.name,
        train_candidate_task.name,
    ] == [
        "Validar dataset",
        "Cargar datos",
        "Preprocesar",
        "Extraer características",
        "Preparar validación",
        "Entrenar candidato",
    ]
    experiment = create_experiment(
        db,
        ExperimentCreate(
            name="Prefect integration",
            dataset_id=demo_dataset.id,
            pipeline_id="tabular_basic",
            pipeline_config={"target_column": "species"},
            models=[{"model_id": "logistic_regression", "parameters": {"max_iter": 300}}],
            validation_strategy="stratified_kfold",
            validation_config={"n_splits": 3},
            primary_metric="balanced_accuracy",
            random_seed=7,
        ),
    )
    try:
        assert run_experiment_flow(experiment.id) == experiment.id
        persisted = get_experiment(db, experiment.id)
        assert persisted.status == "completed"
        assert persisted.runs[0].stage == "completed"
        assert persisted.prefect_flow_run_id is not None
    finally:
        from prefect.server.api.server import SubprocessASGIServer

        for server in list(SubprocessASGIServer._instances.values()):
            server.stop()
