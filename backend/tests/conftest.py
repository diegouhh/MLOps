from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ["MPLCONFIGDIR"] = "/tmp/neuroops-test-matplotlib"
os.environ["PREFECT_HOME"] = "/tmp/neuroops-test-prefect"
os.environ["PREFECT_SERVER_ALLOW_EPHEMERAL_MODE"] = "true"
os.environ["PREFECT_LOGGING_TO_API_ENABLED"] = "false"
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["UV_CACHE_DIR"] = "/tmp/neuroops-test-uv-cache"
os.environ["DO_NOT_TRACK"] = "true"


@pytest.fixture(scope="session", autouse=True)
def test_environment(tmp_path_factory):
    root = tmp_path_factory.mktemp("neuroops")
    os.environ["NEUROOPS_DATABASE_URL"] = f"sqlite:///{root / 'neuroops.db'}"
    os.environ["NEUROOPS_ENVIRONMENT"] = "test"
    os.environ["NEUROOPS_MLFLOW_TRACKING_URI"] = f"sqlite:///{root / 'mlflow.db'}"
    os.environ["NEUROOPS_MLFLOW_REGISTRY_URI"] = f"sqlite:///{root / 'mlflow.db'}"
    os.environ["NEUROOPS_DATA_DIR"] = str(root / "data")
    os.environ["NEUROOPS_MOUNTED_DATASETS_DIR"] = str(root / "datasets")
    os.environ["NEUROOPS_ARTIFACTS_DIR"] = str(root / "artifacts")
    os.environ["NEUROOPS_AUTO_CREATE_SCHEMA"] = "true"
    from app.core.config import clear_settings_cache, get_settings
    from app.db.session import reset_database_caches

    clear_settings_cache()
    reset_database_caches()
    get_settings().ensure_directories()
    (root / "datasets").mkdir()
    yield root


@pytest.fixture()
def db(test_environment):
    from app.db.models import Base
    from app.db.session import get_engine, get_session_factory

    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def settings():
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture()
def demo_path() -> Path:
    return Path(__file__).parents[2] / "data" / "demo" / "iris.csv"


@pytest.fixture()
def demo_dataset(db, settings, demo_path):
    from app.services.datasets import create_dataset

    with demo_path.open("rb") as source:
        return create_dataset(
            db,
            settings,
            name="Iris demo",
            data_type="tabular",
            description="Fixture público",
            filename="iris.csv",
            source=source,
        )
