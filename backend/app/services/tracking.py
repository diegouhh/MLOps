from __future__ import annotations

import threading
from typing import Any

import mlflow
from mlflow import MlflowClient

from app.core.config import Settings

_cache: dict[tuple[str, str], dict[str, Any]] = {}
_lock = threading.Lock()


def _client(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def mlflow_run_tracking(settings: Settings, run_id: str | None) -> dict[str, Any] | None:
    if not run_id:
        return None
    key = (settings.mlflow_tracking_uri, run_id)
    with _lock:
        cached = _cache.get(key)
    if cached:
        return dict(cached)
    try:
        run = _client(settings).get_run(run_id)
        experiment_id = run.info.experiment_id
        result = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "path": f"/#/experiments/{experiment_id}/runs/{run_id}",
        }
        with _lock:
            _cache[key] = result
        return dict(result)
    except Exception as exc:
        return {"run_id": run_id, "experiment_id": None, "path": None, "error": str(exc)}


def prefect_flow_tracking(flow_run_id: str | None) -> dict[str, Any] | None:
    if not flow_run_id:
        return None
    return {
        "flow_run_id": flow_run_id,
        "path": f"/runs/flow-run/{flow_run_id}",
    }
