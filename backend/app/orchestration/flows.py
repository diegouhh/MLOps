from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task
from prefect.runtime import flow_run as runtime_flow_run

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.services.experiments import (
    build_experiment_training_data,
    execute_candidate,
    extract_experiment_features,
    fail_experiment_execution,
    finalize_experiment_execution,
    initialize_experiment_execution,
    load_experiment_data,
    preprocess_experiment_data,
    validate_experiment_dataset,
)
from app.services.predictions import (
    fail_prediction_execution,
    initialize_prediction_execution,
    load_prediction_model,
    perform_prediction,
    validate_prediction_job,
)


@task(name="Inicializar seguimiento", persist_result=False)
def initialize_experiment_task(experiment_id: str, prefect_flow_run_id: str) -> dict[str, Any]:
    with get_session_factory()() as session:
        return initialize_experiment_execution(
            session,
            get_settings(),
            experiment_id,
            prefect_flow_run_id,
        )


@task(name="Validar dataset", persist_result=False)
def validate_dataset_task(experiment_id: str) -> dict[str, Any]:
    with get_session_factory()() as session:
        return validate_experiment_dataset(session, experiment_id)


@task(name="Cargar datos", persist_result=False)
def load_data_task(experiment_id: str) -> Any:
    with get_session_factory()() as session:
        return load_experiment_data(session, experiment_id)


@task(name="Preprocesar", persist_result=False)
def preprocess_task(experiment_id: str, data: Any) -> Any:
    with get_session_factory()() as session:
        return preprocess_experiment_data(session, experiment_id, data)


@task(name="Extraer características", persist_result=False)
def extract_features_task(experiment_id: str, data: Any) -> Any:
    with get_session_factory()() as session:
        return extract_experiment_features(session, experiment_id, data)


@task(name="Preparar validación", persist_result=False)
def build_training_data_task(experiment_id: str, data: Any) -> dict[str, Any]:
    with get_session_factory()() as session:
        return build_experiment_training_data(session, experiment_id, data)


@task(name="Entrenar candidato", persist_result=False)
def train_candidate_task(
    experiment_id: str,
    training_run_id: str,
    prepared: dict[str, Any],
    dataset_summary: dict[str, Any],
    context: dict[str, Any],
) -> str:
    with get_session_factory()() as session:
        return execute_candidate(
            session,
            get_settings(),
            experiment_id,
            training_run_id,
            prepared,
            dataset_summary,
            context,
        )


@task(name="Seleccionar y registrar champion", persist_result=False)
def finalize_experiment_task(experiment_id: str, context: dict[str, Any]) -> str:
    with get_session_factory()() as session:
        finalize_experiment_execution(session, get_settings(), experiment_id, context)
    return experiment_id


@task(name="Persistir fallo del experimento", persist_result=False)
def fail_experiment_task(experiment_id: str, error_message: str) -> None:
    with get_session_factory()() as session:
        fail_experiment_execution(
            session,
            get_settings(),
            experiment_id,
            RuntimeError(error_message),
        )


@flow(
    name="NeuroOps experiment",
    persist_result=False,
)
def run_experiment_flow(experiment_id: str) -> str:
    try:
        context = initialize_experiment_task(experiment_id, str(runtime_flow_run.id))
        dataset_summary = validate_dataset_task(experiment_id)
        data = load_data_task(experiment_id)
        data = preprocess_task(experiment_id, data)
        data = extract_features_task(experiment_id, data)
        prepared = build_training_data_task(experiment_id, data)
        futures = [
            train_candidate_task.with_options(
                name=f"Entrenar y evaluar - {candidate['model_id']}"
            ).submit(
                experiment_id,
                candidate["training_run_id"],
                prepared,
                dataset_summary,
                context,
            )
            for candidate in context["candidates"]
        ]
        logger = get_run_logger()
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                logger.error("Un candidato falló: %s", exc)
        return finalize_experiment_task(experiment_id, context)
    except Exception as exc:
        fail_experiment_task(experiment_id, str(exc))
        raise


@task(name="Inicializar predicción", persist_result=False)
def initialize_prediction_task(job_id: str, prefect_flow_run_id: str) -> dict[str, Any]:
    with get_session_factory()() as session:
        return initialize_prediction_execution(
            session,
            get_settings(),
            job_id,
            prefect_flow_run_id,
        )


@task(name="Validar registros", persist_result=False)
def validate_prediction_task(job_id: str) -> str:
    with get_session_factory()() as session:
        return validate_prediction_job(session, job_id)


@task(name="Cargar modelo registrado", persist_result=False)
def load_model_task(selector: str) -> Any:
    return load_prediction_model(get_settings(), selector)


@task(name="Ejecutar inferencia", persist_result=False)
def perform_prediction_task(job_id: str, model: Any) -> str:
    with get_session_factory()() as session:
        perform_prediction(session, get_settings(), job_id, model)
    return job_id


@task(name="Persistir fallo de predicción", persist_result=False)
def fail_prediction_task(job_id: str, error_message: str) -> None:
    with get_session_factory()() as session:
        fail_prediction_execution(
            session,
            get_settings(),
            job_id,
            RuntimeError(error_message),
        )


@flow(
    name="NeuroOps prediction",
    persist_result=False,
)
def run_prediction_flow(job_id: str) -> str:
    try:
        context = initialize_prediction_task(job_id, str(runtime_flow_run.id))
        validate_prediction_task(job_id)
        model = load_model_task(context["selector"])
        return perform_prediction_task(job_id, model)
    except Exception as exc:
        fail_prediction_task(job_id, str(exc))
        raise
