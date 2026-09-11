from __future__ import annotations

from collections import Counter
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.core.traceability import prediction_run_name
from app.db.models import Dataset, PredictionJob, RegisteredModelReference
from app.ml.pipelines.registry import pipeline_registry
from app.services.experiments import get_experiment
from app.services.registry import get_model_reference, validate_prediction_records


def _resolve_reference(db: Session, job: PredictionJob) -> RegisteredModelReference:
    selector = job.resolved_model_version or job.version_or_alias
    reference = get_model_reference(db, job.registered_model_name, selector)
    if job.resolved_model_version != reference.version:
        job.resolved_model_version = reference.version
        db.commit()
    return reference


def _dataset_version(db: Session, reference: RegisteredModelReference):
    experiment = get_experiment(db, reference.experiment_id)
    dataset = db.get(Dataset, experiment.dataset_id)
    if not dataset:
        raise ResourceNotFoundError("Dataset no encontrado")
    version = next(
        (item for item in dataset.versions if item.version == experiment.dataset_version),
        None,
    )
    if not version:
        raise ValidationError("La versión del dataset usada para entrenar ya no está disponible")
    return experiment, dataset, version


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
    experiment, dataset, _version = _dataset_version(db, reference)
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
            "neuroops.experiment_id": experiment.id,
            "neuroops.dataset_id": dataset.id,
        },
    )
    job.mlflow_run_id = active_run.info.run_id
    db.commit()
    for key, value in {
        "prediction_job_id": job.id,
        "model_name": job.registered_model_name,
        "requested_version_or_alias": job.version_or_alias,
        "resolved_model_version": reference.version,
        "experiment_id": experiment.id,
        "dataset_id": dataset.id,
        "dataset_version": experiment.dataset_version,
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


def _python_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _align_model_features(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return frame
    expected_names = [str(value) for value in expected]
    missing = [column for column in expected_names if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected_names]
    if missing or extra:
        details = []
        if missing:
            details.append(f"faltan {len(missing)} variables")
        if extra:
            details.append(f"sobran {len(extra)} variables")
        raise ValidationError(
            "El registro EEG no produce el mismo conjunto de características usado durante "
            f"el entrenamiento ({', '.join(details)}). Revisa canales y configuración."
        )
    return frame[expected_names]


def _aggregate_eeg_result(
    recording: str,
    predictions: Any,
    probabilities: Any,
    classes: Any,
) -> dict[str, Any]:
    values = [_python_value(value) for value in np.asarray(predictions).tolist()]
    if not values:
        raise ValidationError("El registro EEG no produjo predicciones")
    counts = Counter(str(value) for value in values)
    winner = counts.most_common(1)[0][0]
    result: dict[str, Any] = {
        "recording": recording,
        "prediction": winner,
        "epochs_analyzed": len(values),
        "class_distribution": dict(counts),
        "aggregation": "majority_vote",
    }
    if probabilities is not None and classes is not None:
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim == 2 and matrix.shape[0] == len(values):
            means = matrix.mean(axis=0)
            result["mean_probabilities"] = {
                str(label): float(score)
                for label, score in zip(classes, means, strict=False)
            }
    return result


def _perform_eeg_prediction(
    db: Session,
    reference: RegisteredModelReference,
    job: PredictionJob,
    model: Any,
) -> dict[str, Any]:
    experiment, _dataset, version = _dataset_version(db, reference)
    pipeline = pipeline_registry.get(experiment.pipeline_id)
    builder = getattr(pipeline, "build_prediction_features", None)
    if builder is None:
        raise ValidationError("El pipeline EEG no implementa inferencia desde registros BIDS")

    results = []
    for record in job.input_payload:
        recording = record["recording"]
        frame = builder(version.storage_path, experiment.pipeline_config, recording)
        frame = _align_model_features(model, frame)
        predictions = model.predict(frame)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(frame)
            except (AttributeError, ValueError):
                probabilities = None
        results.append(
            _aggregate_eeg_result(
                recording,
                predictions,
                probabilities,
                getattr(model, "classes_", None),
            )
        )
    payload: dict[str, Any] = {"mode": "eeg_recording", "results": results}
    if len(results) == 1:
        payload.update(
            {
                "prediction": results[0]["prediction"],
                "epochs_analyzed": results[0]["epochs_analyzed"],
                "class_distribution": results[0]["class_distribution"],
            }
        )
        if "mean_probabilities" in results[0]:
            payload["mean_probabilities"] = results[0]["mean_probabilities"]
    return payload


def perform_prediction(
    db: Session,
    settings: Settings,
    job_id: str,
    model: Any,
) -> PredictionJob:
    job = get_prediction_job(db, job_id)
    reference = _resolve_reference(db, job)
    _experiment, dataset, _version = _dataset_version(db, reference)

    if dataset.data_type == "eeg_bids":
        payload = _perform_eeg_prediction(db, reference, job, model)
    else:
        frame = pd.DataFrame(job.input_payload)
        predictions = model.predict(frame)
        payload = {"mode": "tabular", "predictions": predictions.tolist()}
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
