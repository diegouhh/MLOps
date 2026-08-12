from __future__ import annotations

import importlib.metadata
from typing import Annotated, Any

import httpx
import mlflow
from fastapi import APIRouter, Depends, File, Form, Response, UploadFile, status
from mlflow import MlflowClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.db.models import (
    ArtifactReference,
    PredictionJob,
    RegisteredModelReference,
    TrainingRun,
)
from app.db.session import get_db
from app.ml.models.registry import model_registry
from app.ml.pipelines.registry import pipeline_registry
from app.schemas.api import (
    AliasRequest,
    DatasetRead,
    ExperimentCreate,
    ExperimentRead,
    JobAccepted,
    MountedDatasetCandidate,
    MountedDatasetCreate,
    PredictionCreate,
    PredictionRead,
    TrainingRunRead,
    WinnerRequest,
)
from app.services.datasets import (
    create_dataset,
    create_dataset_version,
    delete_dataset,
    discover_mounted_datasets,
    get_dataset,
    list_datasets,
    register_mounted_dataset,
)
from app.services.experiments import (
    cancel_experiment,
    create_experiment,
    get_experiment,
    list_experiments,
    register_candidate,
    rerun_experiment,
)
from app.services.jobs import (
    cancel_flow_run,
    mark_submission_failed,
    submit_experiment,
    submit_prediction,
)
from app.services.predictions import create_prediction_job, get_prediction_job
from app.services.registry import (
    get_model_reference,
    reference_details,
    reference_input_schema,
)
from app.services.tracking import mlflow_run_tracking, prefect_flow_tracking

router = APIRouter()
Db = Annotated[Session, Depends(get_db)]
Config = Annotated[Settings, Depends(get_settings)]


def _service_status(url: str | None, health_path: str = "/health") -> dict[str, Any]:
    if not url or url.startswith("sqlite") or url.startswith("file"):
        return {"status": "embedded", "url": url}
    try:
        response = httpx.get(url.rstrip("/") + health_path, timeout=1.5)
        return {
            "status": "healthy" if 200 <= response.status_code < 300 else "unhealthy",
            "url": url,
            "status_code": response.status_code,
        }
    except Exception:
        return {"status": "unreachable", "url": url}


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "neuroops-api"}


@router.get("/system/dependencies", tags=["system"])
def dependencies(settings: Config) -> dict[str, Any]:
    packages = {}
    for package in [
        "fastapi",
        "mlflow",
        "prefect",
        "scikit-learn",
        "mne",
        "mne-bids",
        "sovaharmony",
    ]:
        try:
            packages[package] = {"installed": True, "version": importlib.metadata.version(package)}
        except importlib.metadata.PackageNotFoundError:
            packages[package] = {"installed": False, "version": None}
    return {
        "packages": packages,
        "services": {
            "api": {"status": "healthy"},
            "mlflow": _service_status(settings.mlflow_tracking_uri),
            "prefect": _service_status(
                settings.prefect_api_url.removesuffix("/api") if settings.prefect_api_url else None,
                "/api/health",
            ),
            "prefect_worker": _service_status(settings.prefect_worker_health_url),
        },
    }


@router.post(
    "/datasets", response_model=DatasetRead, status_code=status.HTTP_201_CREATED, tags=["datasets"]
)
def upload_dataset(
    db: Db,
    settings: Config,
    file: Annotated[UploadFile, File(description="CSV tabular o ZIP con raíz BIDS")],
    name: Annotated[str, Form(min_length=1, max_length=200)],
    data_type: Annotated[str, Form(pattern="^(tabular|eeg_bids)$")],
    description: Annotated[str | None, Form()] = None,
) -> DatasetRead:
    if not file.filename:
        raise ValidationError("El archivo necesita un nombre")
    dataset = create_dataset(
        db,
        settings,
        name=name,
        data_type=data_type,
        description=description,
        filename=file.filename,
        source=file.file,
    )
    return DatasetRead.model_validate(dataset)


@router.get("/datasets", response_model=list[DatasetRead], tags=["datasets"])
def datasets(db: Db) -> list[DatasetRead]:
    return [DatasetRead.model_validate(item) for item in list_datasets(db)]


@router.get(
    "/datasets/mounted",
    response_model=list[MountedDatasetCandidate],
    tags=["datasets"],
)
def mounted_datasets(db: Db, settings: Config) -> list[MountedDatasetCandidate]:
    return [
        MountedDatasetCandidate.model_validate(item)
        for item in discover_mounted_datasets(db, settings)
    ]


@router.post(
    "/datasets/mounted",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    tags=["datasets"],
)
def mounted_dataset_create(
    payload: MountedDatasetCreate,
    db: Db,
    settings: Config,
) -> DatasetRead:
    dataset = register_mounted_dataset(
        db,
        settings,
        name=payload.name,
        relative_path=payload.relative_path,
        description=payload.description,
    )
    return DatasetRead.model_validate(dataset)


@router.get("/datasets/{dataset_id}", response_model=DatasetRead, tags=["datasets"])
def dataset_detail(dataset_id: str, db: Db) -> DatasetRead:
    return DatasetRead.model_validate(get_dataset(db, dataset_id))


@router.post(
    "/datasets/{dataset_id}/versions",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    tags=["datasets"],
)
def dataset_version_create(
    dataset_id: str,
    db: Db,
    settings: Config,
    file: Annotated[UploadFile, File(description="Nueva versión CSV o ZIP/BIDS")],
) -> DatasetRead:
    if not file.filename:
        raise ValidationError("El archivo necesita un nombre")
    dataset = create_dataset_version(
        db,
        settings,
        dataset_id,
        filename=file.filename,
        source=file.file,
    )
    return DatasetRead.model_validate(dataset)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["datasets"])
def dataset_delete(dataset_id: str, db: Db, settings: Config) -> Response:
    delete_dataset(db, settings, dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/pipelines", tags=["catalog"])
def pipelines() -> list[dict[str, Any]]:
    return [
        {**plugin.metadata.to_dict(), "source": "built_in"}
        for plugin in pipeline_registry.list()
    ]


@router.get("/pipelines/{pipeline_id}", tags=["catalog"])
def pipeline_detail(pipeline_id: str) -> dict[str, Any]:
    plugin = pipeline_registry.refresh(pipeline_id)
    return {**plugin.metadata.to_dict(), "source": "built_in"}


@router.post("/pipelines/{pipeline_id}/validate-config", tags=["catalog"])
def validate_pipeline_config(pipeline_id: str, config: dict[str, Any]) -> dict[str, Any]:
    plugin = pipeline_registry.get(pipeline_id)
    if not plugin.metadata.available:
        raise ValidationError(plugin.metadata.unavailable_reason or "Pipeline no disponible")
    return {"valid": True, "config": plugin.validate_config(config)}


@router.get("/models/catalog", tags=["catalog"])
def models_catalog() -> list[dict[str, Any]]:
    return [plugin.get_metadata() for plugin in model_registry.list()]


@router.get("/models/catalog/{model_id}", tags=["catalog"])
def model_detail(model_id: str) -> dict[str, Any]:
    return model_registry.get(model_id).get_metadata()


@router.post(
    "/experiments",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["experiments"],
)
async def experiment_create(payload: ExperimentCreate, db: Db) -> JobAccepted:
    experiment = create_experiment(db, payload)
    try:
        await submit_experiment(experiment.id)
    except Exception as exc:
        mark_submission_failed("experiment", experiment.id, exc)
        raise
    return JobAccepted(id=experiment.id, status=experiment.status)


@router.get("/experiments", response_model=list[ExperimentRead], tags=["experiments"])
def experiments(db: Db) -> list[ExperimentRead]:
    return [ExperimentRead.model_validate(item) for item in list_experiments(db)]


@router.get("/experiments/tracking-summary", tags=["experiments"])
def experiments_tracking_summary(db: Db, settings: Config) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for experiment in list_experiments(db):
        result[experiment.id] = {
            "mlflow": mlflow_run_tracking(settings, experiment.mlflow_parent_run_id),
            "prefect": prefect_flow_tracking(experiment.prefect_flow_run_id),
            "runs": {
                run.id: mlflow_run_tracking(settings, run.mlflow_run_id)
                for run in experiment.runs
                if run.mlflow_run_id
            },
        }
    return result


@router.get("/experiments/{experiment_id}", response_model=ExperimentRead, tags=["experiments"])
def experiment_detail(experiment_id: str, db: Db) -> ExperimentRead:
    return ExperimentRead.model_validate(get_experiment(db, experiment_id))


@router.get("/experiments/{experiment_id}/tracking", tags=["experiments"])
def experiment_tracking(experiment_id: str, db: Db, settings: Config) -> dict[str, Any]:
    experiment = get_experiment(db, experiment_id)
    return {
        "mlflow": mlflow_run_tracking(settings, experiment.mlflow_parent_run_id),
        "prefect": prefect_flow_tracking(experiment.prefect_flow_run_id),
        "runs": {
            run.id: mlflow_run_tracking(settings, run.mlflow_run_id)
            for run in experiment.runs
            if run.mlflow_run_id
        },
    }


@router.post(
    "/experiments/{experiment_id}/cancel", response_model=ExperimentRead, tags=["experiments"]
)
async def experiment_cancel(experiment_id: str, db: Db) -> ExperimentRead:
    experiment = cancel_experiment(db, experiment_id)
    if experiment.prefect_flow_run_id:
        await cancel_flow_run(experiment.prefect_flow_run_id)
    return ExperimentRead.model_validate(experiment)


@router.post("/experiments/{experiment_id}/rerun", response_model=JobAccepted, tags=["experiments"])
async def experiment_rerun(experiment_id: str, db: Db) -> JobAccepted:
    experiment = rerun_experiment(db, experiment_id)
    try:
        await submit_experiment(experiment.id)
    except Exception as exc:
        mark_submission_failed("experiment", experiment.id, exc)
        raise
    return JobAccepted(id=experiment.id, status=experiment.status)


@router.post("/experiments/{experiment_id}/winner", tags=["experiments"])
def choose_winner(
    experiment_id: str, payload: WinnerRequest, db: Db, settings: Config
) -> dict[str, Any]:
    experiment = get_experiment(db, experiment_id)
    run = next((item for item in experiment.runs if item.id == payload.training_run_id), None)
    if not run:
        raise ResourceNotFoundError("La ejecución no pertenece al experimento")
    reference = register_candidate(
        db,
        settings,
        experiment,
        run,
        payload.alias,
        payload.model_name,
    )
    return {"name": reference.name, "version": reference.version, "alias": reference.alias}


@router.get("/runs", response_model=list[TrainingRunRead], tags=["runs"])
def runs(db: Db) -> list[TrainingRunRead]:
    items = db.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc())).all()
    return [TrainingRunRead.model_validate(item) for item in items]


@router.get("/runs/{run_id}", response_model=TrainingRunRead, tags=["runs"])
def run_detail(run_id: str, db: Db) -> TrainingRunRead:
    run = db.get(TrainingRun, run_id)
    if not run:
        raise ResourceNotFoundError("Ejecución no encontrada")
    return TrainingRunRead.model_validate(run)


@router.get("/runs/{run_id}/artifacts", tags=["runs"])
def run_artifacts(run_id: str, db: Db) -> list[dict[str, Any]]:
    if not db.get(TrainingRun, run_id):
        raise ResourceNotFoundError("Ejecución no encontrada")
    return [
        {"id": item.id, "name": item.name, "kind": item.kind, "uri": item.uri}
        for item in db.scalars(
            select(ArtifactReference).where(ArtifactReference.training_run_id == run_id)
        ).all()
    ]


def _mlflow_client(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    return MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_registry_uri
    )


@router.get("/registry/models", tags=["registry"])
def registry_models(db: Db, settings: Config) -> list[dict[str, Any]]:
    references = db.scalars(
        select(RegisteredModelReference).order_by(RegisteredModelReference.created_at.desc())
    ).all()
    grouped: dict[str, dict[str, Any]] = {}
    for reference in references:
        entry = grouped.setdefault(reference.name, {"name": reference.name, "versions": []})
        entry["versions"].append(reference_details(db, reference))
    return list(grouped.values())


@router.get("/registry/models/{name}", tags=["registry"])
def registry_model(name: str, db: Db) -> dict[str, Any]:
    references = db.scalars(
        select(RegisteredModelReference)
        .where(RegisteredModelReference.name == name)
        .order_by(RegisteredModelReference.created_at.desc())
    ).all()
    if not references:
        raise ResourceNotFoundError("Modelo registrado no encontrado")
    return {
        "name": name,
        "versions": [reference_details(db, item) for item in references],
    }


@router.get(
    "/registry/models/{name}/versions/{version_or_alias}/input-schema",
    tags=["registry"],
)
def registry_input_schema(name: str, version_or_alias: str, db: Db) -> dict[str, Any]:
    reference = get_model_reference(db, name, version_or_alias)
    return reference_input_schema(db, reference)


@router.post("/registry/models/{name}/aliases", tags=["registry"])
def registry_alias(name: str, payload: AliasRequest, db: Db, settings: Config) -> dict[str, str]:
    reference = db.scalar(
        select(RegisteredModelReference).where(
            RegisteredModelReference.name == name,
            RegisteredModelReference.version == payload.version,
        )
    )
    if not reference:
        raise ResourceNotFoundError("Versión registrada no encontrada")
    _mlflow_client(settings).set_registered_model_alias(name, payload.alias, payload.version)
    for previous in db.scalars(
        select(RegisteredModelReference).where(
            RegisteredModelReference.name == name,
            RegisteredModelReference.alias == payload.alias,
            RegisteredModelReference.id != reference.id,
        )
    ).all():
        previous.alias = None
    reference.alias = payload.alias
    db.commit()
    return {"name": name, "version": payload.version, "alias": payload.alias}


@router.post(
    "/predictions",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["predictions"],
)
async def prediction_create(payload: PredictionCreate, db: Db) -> JobAccepted:
    job = create_prediction_job(
        db,
        model_name=payload.model_name,
        version_or_alias=payload.version_or_alias,
        records=payload.records,
    )
    try:
        await submit_prediction(job.id)
    except Exception as exc:
        mark_submission_failed("prediction", job.id, exc)
        raise
    return JobAccepted(id=job.id, status=job.status)


@router.get("/predictions", response_model=list[PredictionRead], tags=["predictions"])
def predictions(db: Db) -> list[PredictionRead]:
    jobs = db.scalars(select(PredictionJob).order_by(PredictionJob.created_at.desc())).all()
    return [PredictionRead.model_validate(job) for job in jobs]


@router.get("/predictions/tracking-summary", tags=["predictions"])
def predictions_tracking_summary(db: Db, settings: Config) -> dict[str, Any]:
    jobs = db.scalars(
        select(PredictionJob).where(
            (PredictionJob.mlflow_run_id.is_not(None))
            | (PredictionJob.prefect_flow_run_id.is_not(None))
        )
    ).all()
    return {
        job.id: {
            "mlflow": mlflow_run_tracking(settings, job.mlflow_run_id),
            "prefect": prefect_flow_tracking(job.prefect_flow_run_id),
        }
        for job in jobs
    }


@router.get("/predictions/{prediction_id}", response_model=PredictionRead, tags=["predictions"])
def prediction_detail(prediction_id: str, db: Db) -> PredictionRead:
    return PredictionRead.model_validate(get_prediction_job(db, prediction_id))


@router.get("/predictions/{prediction_id}/tracking", tags=["predictions"])
def prediction_tracking(prediction_id: str, db: Db, settings: Config) -> dict[str, Any]:
    job = get_prediction_job(db, prediction_id)
    return {
        "mlflow": mlflow_run_tracking(settings, job.mlflow_run_id),
        "prefect": prefect_flow_tracking(job.prefect_flow_run_id),
    }
