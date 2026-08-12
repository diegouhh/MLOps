from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), index=True)
    data_type: Mapped[str] = mapped_column(String(30), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default="ready", index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=1)
    versions: Mapped[list[DatasetVersion]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="DatasetVersion.version"
    )


class DatasetVersion(Base, TimestampMixin):
    __tablename__ = "dataset_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    row_count: Mapped[int | None] = mapped_column(Integer)
    column_count: Mapped[int | None] = mapped_column(Integer)
    schema_summary: Mapped[dict | None] = mapped_column(JSON)
    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class PipelineDefinition(Base, TimestampMixin):
    __tablename__ = "pipeline_definitions"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    version: Mapped[str] = mapped_column(String(30))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class PipelineConfiguration(Base, TimestampMixin):
    __tablename__ = "pipeline_configurations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    pipeline_id: Mapped[str] = mapped_column(String(100), index=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), index=True)
    dataset_version: Mapped[int] = mapped_column(Integer, default=1)
    pipeline_id: Mapped[str] = mapped_column(String(100), index=True)
    registered_model_name: Mapped[str | None] = mapped_column(String(200), index=True)
    pipeline_config: Mapped[dict] = mapped_column(JSON, default=dict)
    model_ids: Mapped[list] = mapped_column(JSON, default=list)
    model_parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_strategy: Mapped[str] = mapped_column(String(50))
    validation_config: Mapped[dict] = mapped_column(JSON, default=dict)
    primary_metric: Mapped[str] = mapped_column(String(50), default="f1_macro")
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    mlflow_parent_run_id: Mapped[str | None] = mapped_column(String(100), index=True)
    prefect_flow_run_id: Mapped[str | None] = mapped_column(String(100))
    winner_candidate_id: Mapped[str | None] = mapped_column(String(36), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False)
    runs: Mapped[list[TrainingRun]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class TrainingRun(Base, TimestampMixin):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    model_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30), default="queued")
    stage: Mapped[str] = mapped_column(String(80), default="queued")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), index=True)
    error_message: Mapped[str | None] = mapped_column(Text)
    experiment: Mapped[Experiment] = relationship(back_populates="runs")
    candidate: Mapped[ModelCandidate | None] = relationship(
        back_populates="run", cascade="all, delete-orphan", uselist=False
    )

    @property
    def candidate_id(self) -> str | None:
        return self.candidate.id if self.candidate else None


class ModelCandidate(Base, TimestampMixin):
    __tablename__ = "model_candidates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), unique=True
    )
    model_id: Mapped[str] = mapped_column(String(100), index=True)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    is_winner: Mapped[bool] = mapped_column(Boolean, default=False)
    run: Mapped[TrainingRun] = relationship(back_populates="candidate")


class RegisteredModelReference(Base, TimestampMixin):
    __tablename__ = "registered_model_references"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), index=True)
    version: Mapped[str] = mapped_column(String(50))
    alias: Mapped[str | None] = mapped_column(String(50), index=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    candidate_id: Mapped[str] = mapped_column(ForeignKey("model_candidates.id"), index=True)
    model_uri: Mapped[str] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class PredictionJob(Base, TimestampMixin):
    __tablename__ = "prediction_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    registered_model_name: Mapped[str] = mapped_column(String(200), index=True)
    version_or_alias: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    input_payload: Mapped[list] = mapped_column(JSON, default=list)
    result_payload: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(100), index=True)
    prefect_flow_run_id: Mapped[str | None] = mapped_column(String(100), index=True)


class ArtifactReference(Base, TimestampMixin):
    __tablename__ = "artifact_references"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    training_run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(50))
    uri: Mapped[str] = mapped_column(Text)
