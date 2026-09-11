from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrmModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DatasetVersionRead(OrmModel):
    id: str
    version: int
    fingerprint: str
    original_filename: str
    size_bytes: int
    row_count: int | None
    column_count: int | None
    schema_summary: dict[str, Any] | None
    created_at: datetime


class DatasetRead(OrmModel):
    id: str
    name: str
    data_type: str
    description: str | None
    status: str
    current_version: int
    created_at: datetime
    versions: list[DatasetVersionRead] = []


class MountedDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    relative_path: str = Field(min_length=1, max_length=500)
    description: str | None = Field(default=None, max_length=2000)


class MountedDatasetCandidate(BaseModel):
    relative_path: str
    name: str
    subject_count: int = 0
    eeg_file_count: int = 0
    size_bytes: int = 0
    registered: bool = False
    valid: bool = True
    validation_error: str | None = None


class ModelSelection(BaseModel):
    model_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: str
    dataset_version: int | None = Field(default=None, ge=1)
    pipeline_id: str
    registered_model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._ -]*$",
    )
    pipeline_config: dict[str, Any]
    models: list[ModelSelection] = Field(min_length=1, max_length=10)
    validation_strategy: Literal[
        "train_test_split", "stratified_kfold", "group_kfold", "stratified_group_kfold"
    ] = "train_test_split"
    validation_config: dict[str, Any] = Field(default_factory=dict)
    primary_metric: Literal[
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
    ] = "f1_macro"
    random_seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("models")
    @classmethod
    def unique_models(cls, models: list[ModelSelection]) -> list[ModelSelection]:
        ids = [model.model_id for model in models]
        if len(ids) != len(set(ids)):
            raise ValueError("No se puede seleccionar el mismo modelo dos veces")
        return models


class TrainingRunRead(OrmModel):
    id: str
    experiment_id: str
    model_id: str
    status: str
    stage: str
    metrics: dict[str, Any]
    parameters: dict[str, Any]
    mlflow_run_id: str | None
    candidate_id: str | None
    error_message: str | None
    created_at: datetime


class ExperimentRead(OrmModel):
    id: str
    name: str
    dataset_id: str
    dataset_version: int
    pipeline_id: str
    registered_model_name: str | None
    pipeline_config: dict[str, Any]
    model_ids: list[str]
    model_parameters: dict[str, Any]
    validation_strategy: str
    validation_config: dict[str, Any]
    primary_metric: str
    random_seed: int
    status: str
    error_message: str | None
    mlflow_parent_run_id: str | None
    prefect_flow_run_id: str | None
    winner_candidate_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    runs: list[TrainingRunRead] = []


class AliasRequest(BaseModel):
    alias: Literal["champion", "challenger"]
    version: str = Field(min_length=1, max_length=50)


class WinnerRequest(BaseModel):
    training_run_id: str
    alias: Literal["champion", "challenger"] = "champion"
    model_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._ -]*$",
    )


class PredictionCreate(BaseModel):
    model_name: str = Field(min_length=1, max_length=200)
    version_or_alias: str = Field(default="champion", min_length=1, max_length=50)
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class PredictionRead(OrmModel):
    id: str
    registered_model_name: str
    version_or_alias: str
    resolved_model_version: str | None
    status: str
    input_payload: list[dict[str, Any]]
    result_payload: dict[str, Any] | None
    error_message: str | None
    mlflow_run_id: str | None
    prefect_flow_run_id: str | None
    created_at: datetime


class JobAccepted(BaseModel):
    id: str
    status: str
