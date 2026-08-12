from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="NEUROOPS_", case_sensitive=False, extra="ignore"
    )

    app_name: str = "NeuroOps"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./neuroops.db"
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    mlflow_registry_uri: str = "sqlite:///./mlflow.db"
    mlflow_experiment_name: str = "NeuroOps"
    prefect_api_url: str | None = None
    prefect_worker_health_url: str | None = None
    prefect_experiment_deployment: str = "NeuroOps experiment/neuroops-experiments"
    prefect_prediction_deployment: str = "NeuroOps prediction/neuroops-predictions"
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    max_upload_mb: int = 50
    max_extracted_mb: int = 250
    data_dir: Path = Path("../data")
    mounted_datasets_dir: Path = Path("../datasets")
    artifacts_dir: Path = Path("../artifacts")
    auto_create_schema: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir / "uploads",
            self.data_dir / "processed",
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
