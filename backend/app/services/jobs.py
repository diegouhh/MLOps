from __future__ import annotations

from uuid import UUID

from prefect.client.orchestration import get_client
from prefect.deployments import run_deployment
from prefect.states import State, StateType

from app.core.config import get_settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.core.traceability import (
    experiment_run_name,
    prediction_run_name,
    prefect_trace_tags,
)
from app.db.models import Experiment, PredictionJob, utcnow
from app.db.session import get_session_factory
from app.services.registry import get_model_reference


async def _dispatch(
    deployment: str,
    parameter_name: str,
    resource_id: str,
    kind: str,
    flow_run_name: str,
) -> str:
    try:
        flow_run = await run_deployment(
            deployment,
            parameters={parameter_name: resource_id},
            flow_run_name=flow_run_name,
            timeout=0,
            idempotency_key=f"neuroops:{kind}:{resource_id}",
            tags=prefect_trace_tags(kind, resource_id),
            as_subflow=False,
        )
        return str(flow_run.id)
    except Exception as exc:
        raise ValidationError(
            "Prefect no pudo aceptar el trabajo. Verifica el deployment y el worker: "
            f"{exc}"
        ) from exc


async def submit_experiment(experiment_id: str) -> str:
    settings = get_settings()
    with get_session_factory()() as session:
        experiment = session.get(Experiment, experiment_id)
        if not experiment:
            raise ResourceNotFoundError("Experimento no encontrado")
        display_name = experiment_run_name(experiment.name, experiment.id)

    flow_run_id = await _dispatch(
        settings.prefect_experiment_deployment,
        "experiment_id",
        experiment_id,
        "experiment",
        display_name,
    )
    with get_session_factory()() as session:
        experiment = session.get(Experiment, experiment_id)
        if not experiment:
            raise ResourceNotFoundError("Experimento no encontrado")
        experiment.prefect_flow_run_id = flow_run_id
        session.commit()
    return flow_run_id


async def submit_prediction(job_id: str) -> str:
    settings = get_settings()
    with get_session_factory()() as session:
        job = session.get(PredictionJob, job_id)
        if not job:
            raise ResourceNotFoundError("Predicción no encontrada")
        if not job.resolved_model_version:
            reference = get_model_reference(
                session,
                job.registered_model_name,
                job.version_or_alias,
            )
            job.resolved_model_version = reference.version
            session.commit()
        display_name = prediction_run_name(
            job.registered_model_name,
            job.resolved_model_version,
            job.id,
        )

    flow_run_id = await _dispatch(
        settings.prefect_prediction_deployment,
        "job_id",
        job_id,
        "prediction",
        display_name,
    )
    with get_session_factory()() as session:
        job = session.get(PredictionJob, job_id)
        if not job:
            raise ResourceNotFoundError("Predicción no encontrada")
        job.prefect_flow_run_id = flow_run_id
        session.commit()
    return flow_run_id


async def cancel_flow_run(flow_run_id: str) -> None:
    async with get_client() as client:
        result = await client.set_flow_run_state(
            flow_run_id=UUID(flow_run_id),
            state=State(type=StateType.CANCELLING),
        )
    if result.status.name == "ABORT":
        raise ValidationError(result.details.reason or "Prefect rechazó la cancelación")


def mark_submission_failed(kind: str, resource_id: str, error: Exception) -> None:
    message = f"No se pudo enviar a Prefect: {error}"[:4000]
    with get_session_factory()() as session:
        if kind == "experiment":
            resource = session.get(Experiment, resource_id)
            if resource:
                resource.status = "failed"
                resource.error_message = message
                resource.completed_at = utcnow()
                for run in resource.runs:
                    run.status = "failed"
                    run.stage = "failed"
                    run.error_message = message[:2000]
        elif kind == "prediction":
            resource = session.get(PredictionJob, resource_id)
            if resource:
                resource.status = "failed"
                resource.error_message = message[:2000]
        session.commit()
