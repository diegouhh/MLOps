from __future__ import annotations


def _clean_name(value: str, fallback: str) -> str:
    normalized = " ".join(value.split())
    return normalized or fallback


def experiment_run_name(name: str, experiment_id: str) -> str:
    return _clean_name(name, f"Experimento {experiment_id[:8]}")


def prediction_run_name(model_name: str, model_version: str | None, job_id: str) -> str:
    version = model_version or "sin-version"
    model = _clean_name(model_name, "modelo")
    return f"Predicción {model} v{version} [{job_id[:8]}]"


def prefect_trace_tags(kind: str, resource_id: str) -> list[str]:
    return [
        "neuroops",
        kind,
        f"neuroops-{kind}-id:{resource_id}",
    ]
