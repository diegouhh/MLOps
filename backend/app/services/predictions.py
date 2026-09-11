from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.core.traceability import prediction_run_name
from app.db.models import PredictionJob, RegisteredModelReference
from app.services.registry import get_model_reference, validate_prediction_records


def _resolve_reference(db: Session, job: PredictionJob) -> RegisteredModelReference:
    selector = job.resolved_model_version or job.version_or_alias
    reference = get_model_reference(db, job.registered_model_name, selector)
    if job.resolved_model_version != reference.version:
        job.resolved_model_version = reference.version
        db.commit()
    return reference


def create_prediction_job(
    db: Session,
    *,
    model_name: str,
    version_or_alias: str,
    records: list[dict[str, Any]],
) -> PredictionJob:
    reference = get_model_reference(db, model_name, version_or_alias)
    validate_prediction_records(db, reference, records)
    job = PredictionJob(
        registered_model_name=model_name,
        version_or_alias=version_or_alias,
        resolved_model_version=reference.version,
        status="queued",
        input_payload=records,
    )
    db.add(job)
    db.commit()
    return job


def get_prediction_job(db: Session, job_id: str) -> PredictionJob:
    job = db.get(PredictionJob, job_id)
    if not job:
        raise ResourceNotFoundError("Predicción no encontrada")
    return job


def _configure_mlflow(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri,
    )


def initialize_prediction_execution(
    db: Session,
    settings: Settings,
    job_id: str,
    prefect_flow_run_id: str | None = None,
) -> dict[str, Any]:
    job = get_prediction_job(db, job_id)
    if job.status == "running":
        raise ValidationError("La predicción ya está en ejecución")
    if job.status == "completed":
        raise ValidationError("La predicción ya terminó")
    reference = _resolve_reference(db, job)
    job.status = "running"
    job.error_message = None
    job.prefect_flow_run_id = prefect_flow_run_id or job.prefect_flow_run_id
    db.commit()
    client = _configure_mlflow(settings)
    tracking_experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if not tracking_experiment:
        raise ValidationError("MLflow no pudo crear el experimento de seguimiento")
    display_name = prediction_run_name(
        job.registered_model_name,
        reference.version,
        job.id,
    )
    active_run = client.create_run(
        tracking_experiment.experiment_id,
        tags={
            "mlflow.runName": display_name,
            "neuroops.run_type": "prediction",
            "neuroops.prediction_job_id": job.id,
            "neuroops.model_name": job.registered_model_name,
            "neuroops.model_version": reference.version,
        },
    )
    job.mlflow_run_id = active_run.info.run_id
    db.commit()
    for key, value in {
        "prediction_job_id": job.id,
        "model_name": job.registered_model_name,
        "requested_version_or_alias": job.version_or_alias,
        "resolved_model_version": reference.version,
        "record_count": len(job.input_payload),
    }.items():
        client.log_param(active_run.info.run_id, key, value)
    selector = f"models:/{job.registered_model_name}/{reference.version}"
    return {
        "job_id": job.id,
        "selector": selector,
        "records": job.input_payload,
        "mlflow_run_id": active_run.info.run_id,
        "resolved_model_version": reference.version,
    }


def validate_prediction_job(db: Session, job_id: str) -> str:
    job = get_prediction_job(db, job_id)
    reference = _resolve_reference(db, job)
    validate_prediction_records(db, reference, job.input_payload)
    return job.id


def load_prediction_model(settings: Settings, selector: str) -> Any:
    _configure_mlflow(settings)
    return mlflow.sklearn.load_model(selector)


def perform_prediction(
    db: Session,
    settings: Settings,
    job_id: str,
    model: Any,
) -> PredictionJob:
    job = get_prediction_job(db, job_id)
    frame = pd.DataFrame(job.input_payload)
    predictions = model.predict(frame)
    payload: dict[str, Any] = {"predictions": predictions.tolist()}
    if hasattr(model, "predict_proba"):
        try:
            payload["probabilities"] = model.predict_proba(frame).tolist()
            classes = getattr(model, "classes_", None)
            if classes is not None:
                payload["classes"] = [str(value) for value in classes]
        except AttributeError:
            pass
    job.result_payload = payload
    job.status = "completed"
    db.commit()
    if job.mlflow_run_id:
        client = _configure_mlflow(settings)
        client.set_tag(job.mlflow_run_id, "neuroops.status", "completed")
        client.set_terminated(job.mlflow_run_id, status="FINISHED")
    return job


def fail_prediction_execution(
    db: Session,
    settings: Settings,
    job_id: str,
    error: Exception,
) -> None:
    job = get_prediction_job(db, job_id)
    job.status = "failed"
    job.error_message = str(error)[:2000]
    db.commit()
    if job.mlflow_run_id:
        try:
            client = _configure_mlflow(settings)
            client.set_tag(job.mlflow_run_id, "neuroops.status", "failed")
            client.set_terminated(job.mlflow_run_id, status="FAILED")
        except Exception:
            pass


def execute_prediction(db: Session, settings: Settings, job_id: str) -> PredictionJob:
    try:
        context = initialize_prediction_execution(db, settings, job_id)
        validate_prediction_job(db, job_id)
        model = load_prediction_model(settings, context["selector"])
        return perform_prediction(db, settings, job_id, model)
    except Exception as exc:
        fail_prediction_execution(db, settings, job_id, exc)
        raise
