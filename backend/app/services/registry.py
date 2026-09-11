from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
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

EEG_PREDICTION_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set"}


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


def _dataset_version(dataset: Dataset, version: int):
    current = next((item for item in dataset.versions if item.version == version), None)
    if not current:
        raise ValidationError(f"La versión v{version} del dataset no está disponible")
    return current


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _tabular_examples(
    storage_path: str,
    columns: list[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(storage_path, usecols=columns)
    except Exception:
        return []
    examples: list[dict[str, Any]] = []
    for _, row in frame.head(200).iterrows():
        record = {
            column: _json_value(row[column])
            for column in columns
        }
        if any(value is None for value in record.values()):
            continue
        examples.append(record)
        if len(examples) >= limit:
            break
    return examples


def _bids_entities(relative_path: Path) -> dict[str, str | None]:
    entities: dict[str, str] = {}
    for part in (*relative_path.parts[:-1], relative_path.stem):
        for token in part.split("_"):
            if "-" not in token:
                continue
            key, value = token.split("-", 1)
            if key in {"sub", "ses", "task", "run"} and value:
                entities.setdefault(key, value)
    return {
        "subject": entities.get("sub"),
        "session": entities.get("ses"),
        "task": entities.get("task"),
        "run": entities.get("run"),
    }


def _list_eeg_recordings(storage_path: str) -> list[dict[str, Any]]:
    root = Path(storage_path).resolve()
    if not root.is_dir():
        return []
    recordings: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*_eeg.*")):
        if not path.is_file() or path.suffix.lower() not in EEG_PREDICTION_EXTENSIONS:
            continue
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        if "derivatives" in relative.parts:
            continue
        entities = _bids_entities(relative)
        label_parts = []
        if entities["subject"]:
            label_parts.append(f"Sujeto {entities['subject']}")
        if entities["session"]:
            label_parts.append(f"Sesión {entities['session']}")
        if entities["task"]:
            label_parts.append(f"Tarea {entities['task']}")
        if entities["run"]:
            label_parts.append(f"Run {entities['run']}")
        recordings.append(
            {
                "value": relative.as_posix(),
                "label": " - ".join(label_parts) or relative.name,
                "subject": entities["subject"],
                "session": entities["session"],
                "task": entities["task"],
                "run": entities["run"],
            }
        )
    return recordings


def reference_input_schema(db: Session, reference: RegisteredModelReference) -> dict[str, Any]:
    experiment = get_experiment(db, reference.experiment_id)
    dataset = get_dataset(db, experiment.dataset_id)
    current = _dataset_version(dataset, experiment.dataset_version)
    config = experiment.pipeline_config or {}

    common = {
        "model_name": reference.name,
        "version": reference.version,
        "alias": reference.alias,
        "experiment_id": experiment.id,
        "experiment_name": experiment.name,
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "dataset_version": experiment.dataset_version,
        "data_type": dataset.data_type,
        "pipeline_id": experiment.pipeline_id,
        "target_column": config.get("target_column"),
    }

    if dataset.data_type == "eeg_bids":
        recordings = _list_eeg_recordings(current.storage_path)
        return {
            **common,
            "input_mode": "eeg_recording",
            "fields": [],
            "example": {},
            "examples": [],
            "recordings": recordings,
            "help": (
                "Selecciona un registro EEG. NeuroOps aplicará automáticamente el mismo "
                "preprocesamiento y extracción de características usados durante el entrenamiento."
            ),
        }

    summary = current.schema_summary if current.schema_summary else {}
    columns = list(summary.get("columns", []))
    dtypes = dict(summary.get("dtypes", {}))
    excluded = {
        config.get("target_column"),
        config.get("group_column"),
        *config.get("drop_columns", []),
    }
    feature_columns = [column for column in columns if column not in excluded]
    fields = []
    example: dict[str, Any] = {}
    for column in feature_columns:
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
    examples = _tabular_examples(current.storage_path, feature_columns)
    if examples:
        example = examples[0]
    return {
        **common,
        "input_mode": "tabular",
        "fields": fields,
        "example": example,
        "examples": examples,
        "recordings": [],
        "help": (
            "NeuroOps cargó ejemplos reales del dataset para que puedas probar el modelo "
            "sin escribir todas las variables manualmente."
        ),
    }


def validate_prediction_records(
    db: Session,
    reference: RegisteredModelReference,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    schema = reference_input_schema(db, reference)
    if schema["input_mode"] == "eeg_recording":
        allowed = {item["value"] for item in schema["recordings"]}
        if not allowed:
            raise ValidationError(
    "El dataset no contiene registros EEG disponibles para predicción"
)
        for index, record in enumerate(records, start=1):
            if set(record) != {"recording"}:
                raise ValidationError(
                    f"Registro {index}: selecciona únicamente un registro EEG válido"
                )
            if record["recording"] not in allowed:
                raise ValidationError(f"Registro {index}: el registro EEG no pertenece al dataset")
        return schema

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
