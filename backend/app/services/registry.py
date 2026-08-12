from __future__ import annotations

import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ResourceNotFoundError, ValidationError
from app.db.models import (
    Dataset,
    ModelCandidate,
    RegisteredModelReference,
    TrainingRun,
)
from app.services.datasets import get_dataset
from app.services.experiments import get_experiment


def get_model_reference(db: Session, name: str, version_or_alias: str) -> RegisteredModelReference:
    selector = (
        RegisteredModelReference.version == version_or_alias
        if version_or_alias.isdigit()
        else RegisteredModelReference.alias == version_or_alias
    )
    reference = db.scalar(
        select(RegisteredModelReference).where(
            RegisteredModelReference.name == name,
            selector,
        )
    )
    if not reference:
        raise ResourceNotFoundError("Versión o alias registrado no encontrado")
    return reference


def reference_details(db: Session, reference: RegisteredModelReference) -> dict[str, Any]:
    candidate = db.get(ModelCandidate, reference.candidate_id)
    run = db.get(TrainingRun, candidate.training_run_id) if candidate else None
    experiment = get_experiment(db, reference.experiment_id)
    dataset = db.get(Dataset, experiment.dataset_id)
    return {
        "version": reference.version,
        "alias": reference.alias,
        "metrics": reference.metrics,
        "experiment_id": reference.experiment_id,
        "candidate_id": reference.candidate_id,
        "training_run_id": run.id if run else None,
        "mlflow_run_id": run.mlflow_run_id if run else None,
        "model_id": run.model_id if run else None,
        "model_uri": reference.model_uri,
        "dataset_id": experiment.dataset_id,
        "dataset_name": dataset.name if dataset else None,
        "pipeline_id": experiment.pipeline_id,
        "primary_metric": experiment.primary_metric,
        "created_at": reference.created_at,
    }


def _input_type(dtype: str) -> tuple[str, Any]:
    normalized = dtype.lower()
    if normalized.startswith(("int", "uint")):
        return "integer", 0
    if normalized.startswith(("float", "decimal")):
        return "number", 0.0
    if normalized in {"bool", "boolean"}:
        return "boolean", False
    return "string", ""


def reference_input_schema(db: Session, reference: RegisteredModelReference) -> dict[str, Any]:
    experiment = get_experiment(db, reference.experiment_id)
    dataset = get_dataset(db, experiment.dataset_id)
    current = next(
        (item for item in dataset.versions if item.version == experiment.dataset_version),
        None,
    )
    summary = current.schema_summary if current and current.schema_summary else {}
    columns = list(summary.get("columns", []))
    dtypes = dict(summary.get("dtypes", {}))
    config = experiment.pipeline_config or {}
    excluded = {
        config.get("target_column"),
        config.get("group_column"),
        *config.get("drop_columns", []),
    }
    fields = []
    example: dict[str, Any] = {}
    for column in columns:
        if column in excluded:
            continue
        field_type, default = _input_type(str(dtypes.get(column, "string")))
        fields.append(
            {
                "name": column,
                "type": field_type,
                "dtype": dtypes.get(column, "string"),
                "required": True,
                "default": default,
            }
        )
        example[column] = default
    return {
        "model_name": reference.name,
        "version": reference.version,
        "alias": reference.alias,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_version": experiment.dataset_version,
        "data_type": dataset.data_type,
        "target_column": config.get("target_column"),
        "fields": fields,
        "example": example,
    }


def validate_prediction_records(
    db: Session,
    reference: RegisteredModelReference,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = reference_input_schema(db, reference)
    fields = {field["name"]: field for field in schema["fields"]}
    if not fields:
        raise ValidationError("El modelo no expone un esquema de entrada compatible")
    expected = set(fields)
    for index, record in enumerate(records, start=1):
        missing = expected - set(record)
        extra = set(record) - expected
        if missing:
            raise ValidationError(
                f"Registro {index}: faltan campos obligatorios: {', '.join(sorted(missing))}"
            )
        if extra:
            raise ValidationError(
                f"Registro {index}: campos no esperados: {', '.join(sorted(extra))}"
            )
        for name, field in fields.items():
            value = record[name]
            field_type = field["type"]
            valid = (
                isinstance(value, bool)
                if field_type == "boolean"
                else isinstance(value, int) and not isinstance(value, bool)
                if field_type == "integer"
                else isinstance(value, (int, float)) and not isinstance(value, bool)
                if field_type == "number"
                else isinstance(value, str)
            )
            if not valid:
                raise ValidationError(
                    f"Registro {index}: '{name}' debe ser de tipo {field_type}"
                )
            if field_type == "number" and not math.isfinite(float(value)):
                raise ValidationError(f"Registro {index}: '{name}' debe ser un número finito")
    return schema
