from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api.v1.router import router
from app.core.config import get_settings
from app.core.errors import NeuroOpsError, ResourceNotFoundError
from app.db.models import (
    Base,
    Experiment,
    PipelineDefinition,
    PredictionJob,
)
from app.db.session import get_engine, get_session_factory
from app.ml.pipelines.registry import pipeline_registry
from app.services.jobs import (
    mark_submission_failed,
    submit_experiment,
    submit_prediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    if settings.auto_create_schema:
        Base.metadata.create_all(get_engine())
    session = get_session_factory()()
    try:
        for plugin in pipeline_registry.list():
            current = session.get(PipelineDefinition, plugin.metadata.id)
            if current:
                current.version = plugin.metadata.version
                current.metadata_json = plugin.metadata.to_dict()
            else:
                session.add(
                    PipelineDefinition(
                        id=plugin.metadata.id,
                        version=plugin.metadata.version,
                        metadata_json=plugin.metadata.to_dict(),
                    )
                )
        queued = [
            item.id
            for item in session.scalars(
                select(Experiment).where(
                    Experiment.status == "queued",
                    Experiment.prefect_flow_run_id.is_(None),
                )
            ).all()
        ]
        queued_predictions = [
            item.id
            for item in session.scalars(
                select(PredictionJob).where(
                    PredictionJob.status == "queued",
                    PredictionJob.prefect_flow_run_id.is_(None),
                )
            ).all()
        ]
        session.commit()
    finally:
        session.close()
    if settings.environment.lower() != "test":
        for experiment_id in queued:
            try:
                await submit_experiment(experiment_id)
            except Exception as exc:
                mark_submission_failed("experiment", experiment_id, exc)
        for prediction_id in queued_predictions:
            try:
                await submit_prediction(prediction_id)
            except Exception as exc:
                mark_submission_failed("prediction", prediction_id, exc)
    yield


settings = get_settings()
app = FastAPI(
    title="NeuroOps API",
    version="0.3.0",
    description=(
        "API reproducible para datasets, pipelines, modelos, experimentos, "
        "registro MLflow y predicciones. "
        "Los catálogos son controlados por plugins; nunca se ejecuta código enviado por usuarios."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "system", "description": "Salud y dependencias"},
        {"name": "datasets", "description": "Carga y versionado de datos"},
        {"name": "catalog", "description": "Pipelines y modelos registrados"},
        {"name": "experiments", "description": "Ejecuciones reproducibles"},
        {"name": "registry", "description": "Model Registry y aliases"},
        {"name": "predictions", "description": "Inferencia con modelos registrados"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_prefix)


@app.exception_handler(NeuroOpsError)
async def handle_domain_error(request: Request, exc: NeuroOpsError) -> JSONResponse:
    status_code = 404 if isinstance(exc, ResourceNotFoundError) else 422
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "NeuroOps", "docs": "/docs", "health": f"{settings.api_prefix}/health"}
