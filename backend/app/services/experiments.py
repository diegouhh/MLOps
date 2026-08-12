from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.base import clone
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.db.models import (
    ArtifactReference,
    Dataset,
    Experiment,
    ModelCandidate,
    RegisteredModelReference,
    TrainingRun,
    utcnow,
)
from app.ml.evaluators.classification import evaluate_predictions, make_validation_split
from app.ml.models.registry import model_registry
from app.ml.pipelines.registry import pipeline_registry
from app.schemas.api import ExperimentCreate


def _source_commit() -> str:
    value = os.getenv("NEUROOPS_SOURCE_COMMIT")
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
    except Exception:
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    names = ["neuroops", "python", "mlflow", "prefect", "scikit-learn", "pandas", "numpy"]
    versions = {"python": platform.python_version()}
    for name in names:
        if name == "python":
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (datetime, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_confusion(path: Path, matrix: list[list[int]], labels: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicción")
    axis.set_ylabel("Valor real")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _plot_roc(path: Path, y_true: Any, probabilities: Any) -> bool:
    if probabilities is None:
        return False
    probabilities = np.asarray(probabilities)
    labels = np.unique(y_true)
    try:
        figure, axis = plt.subplots(figsize=(5, 4))
        if len(labels) == 2:
            scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
            binary = label_binarize(y_true, classes=labels).ravel()
            false_positive, true_positive, _ = roc_curve(binary, scores)
            area = auc(false_positive, true_positive)
            axis.plot(false_positive, true_positive, label=f"AUC {area:.3f}")
        else:
            binary = label_binarize(y_true, classes=labels)
            for index, label in enumerate(labels):
                false_positive, true_positive, _ = roc_curve(
                    binary[:, index], probabilities[:, index]
                )
                area = auc(false_positive, true_positive)
                axis.plot(false_positive, true_positive, label=f"{label} ({area:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="#72817e")
        axis.set(
            xlabel="Tasa de falsos positivos",
            ylabel="Tasa de verdaderos positivos",
            title="Curva ROC",
        )
        axis.legend(loc="lower right", fontsize=8)
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)
        return True
    except ValueError:
        plt.close("all")
        return False


def _configure_mlflow(settings: Settings) -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_registry_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return MlflowClient(
        tracking_uri=settings.mlflow_tracking_uri, registry_uri=settings.mlflow_registry_uri
    )


def get_experiment(db: Session, experiment_id: str) -> Experiment:
    experiment = db.scalar(
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.runs).selectinload(TrainingRun.candidate))
        .execution_options(populate_existing=True)
    )
    if not experiment:
        raise ResourceNotFoundError("Experimento no encontrado")
    return experiment


def list_experiments(db: Session) -> list[Experiment]:
    return list(
        db.scalars(
            select(Experiment)
            .options(selectinload(Experiment.runs).selectinload(TrainingRun.candidate))
            .order_by(Experiment.created_at.desc())
        ).all()
    )


def create_experiment(db: Session, payload: ExperimentCreate) -> Experiment:
    dataset = db.scalar(
        select(Dataset)
        .where(Dataset.id == payload.dataset_id)
        .options(selectinload(Dataset.versions))
    )
    if not dataset:
        raise ResourceNotFoundError("Dataset no encontrado")
    dataset_version = payload.dataset_version or dataset.current_version
    if not any(item.version == dataset_version for item in dataset.versions):
        raise ValidationError(f"La versión v{dataset_version} del dataset no existe")
    pipeline = pipeline_registry.get(payload.pipeline_id)
    if not pipeline.metadata.available:
        raise ValidationError(pipeline.metadata.unavailable_reason or "Pipeline no disponible")
    if pipeline.metadata.data_type != dataset.data_type:
        raise ValidationError("El tipo del dataset no coincide con el pipeline")
    pipeline_config = pipeline.validate_config(payload.pipeline_config)
    parameters: dict[str, dict[str, Any]] = {}
    for selection in payload.models:
        plugin = model_registry.get(selection.model_id)
        if not plugin.metadata.available:
            raise ValidationError(plugin.metadata.unavailable_reason or "Modelo no disponible")
        if (
            pipeline.metadata.supported_models
            and selection.model_id not in pipeline.metadata.supported_models
        ):
            raise ValidationError(
                f"{selection.model_id} no es compatible con {payload.pipeline_id}"
            )
        parameters[selection.model_id] = plugin.validate_parameters(selection.parameters)
    experiment = Experiment(
        name=payload.name,
        dataset_id=payload.dataset_id,
        dataset_version=dataset_version,
        pipeline_id=payload.pipeline_id,
        registered_model_name=(
            payload.registered_model_name or f"neuroops-{payload.pipeline_id}"
        ).strip(),
        pipeline_config=pipeline_config,
        model_ids=[selection.model_id for selection in payload.models],
        model_parameters=parameters,
        validation_strategy=payload.validation_strategy,
        validation_config=payload.validation_config,
        primary_metric=payload.primary_metric,
        random_seed=payload.random_seed,
        status="queued",
    )
    for model_id in experiment.model_ids:
        experiment.runs.append(
            TrainingRun(
                model_id=model_id, status="queued", stage="queued", parameters=parameters[model_id]
            )
        )
    db.add(experiment)
    db.commit()
    return get_experiment(db, experiment.id)


def _set_all_stages(db: Session, experiment: Experiment, stage: str) -> None:
    for run in experiment.runs:
        if run.status in {"queued", "running"}:
            run.stage = stage
    db.commit()


def _dataset_version(dataset: Dataset, version: int):
    selected = next((item for item in dataset.versions if item.version == version), None)
    if not selected:
        raise ValidationError(f"La versión v{version} del dataset no está disponible")
    return selected


def _predict_for_validation(
    estimator: Pipeline,
    X: Any,
    y: Any,
    groups: Any,
    strategy: str,
    splits: Any,
    supports_probability: bool,
) -> tuple[Any, Any, Pipeline]:
    if strategy == "train_test_split":
        train_index, test_index = splits[0]
        fitted = clone(estimator).fit(X.iloc[train_index], y.iloc[train_index])
        predictions = fitted.predict(X.iloc[test_index])
        probabilities = (
            fitted.predict_proba(X.iloc[test_index])
            if supports_probability and hasattr(fitted, "predict_proba")
            else None
        )
        y_true = y.iloc[test_index]
    else:
        split_args = {"groups": groups} if groups is not None else {}
        predictions = cross_val_predict(estimator, X, y, cv=splits, method="predict", **split_args)
        probabilities = None
        if supports_probability:
            try:
                probabilities = cross_val_predict(
                    estimator, X, y, cv=splits, method="predict_proba", **split_args
                )
            except (AttributeError, ValueError):
                probabilities = None
        y_true = y
    final_estimator = clone(estimator).fit(X, y)
    return (y_true, predictions, probabilities, final_estimator)


def _log_model(
    fitted_model: Pipeline,
    X: Any,
    name: str = "model",
) -> Any:
    requirements = [
        f"cloudpickle=={importlib.metadata.version('cloudpickle')}",
        f"numpy=={importlib.metadata.version('numpy')}",
        f"pandas=={importlib.metadata.version('pandas')}",
        f"scikit-learn=={importlib.metadata.version('scikit-learn')}",
    ]
    kwargs: dict[str, Any] = {
        "sk_model": fitted_model,
        "name": name,
        "serialization_format": mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        "pip_requirements": requirements,
    }
    try:
        kwargs["input_example"] = X.head(min(5, len(X)))
    except Exception:
        pass
    return mlflow.sklearn.log_model(**kwargs)


def register_candidate(
    db: Session,
    settings: Settings,
    experiment: Experiment,
    run: TrainingRun,
    alias: str,
    model_name: str | None = None,
) -> RegisteredModelReference:
    if not run.candidate or not run.candidate.artifact_uri or not run.mlflow_run_id:
        raise ValidationError("La ejecución seleccionada no tiene un modelo serializado")
    client = _configure_mlflow(settings)
    model_name = (
        model_name or experiment.registered_model_name or f"neuroops-{experiment.pipeline_id}"
    ).strip()
    existing = db.scalar(
        select(RegisteredModelReference).where(
            RegisteredModelReference.name == model_name,
            RegisteredModelReference.candidate_id == run.candidate.id,
        )
    )
    if existing:
        client.set_registered_model_alias(model_name, alias, existing.version)
        for previous in db.scalars(
            select(RegisteredModelReference).where(
                RegisteredModelReference.name == model_name,
                RegisteredModelReference.alias == alias,
                RegisteredModelReference.id != existing.id,
            )
        ).all():
            previous.alias = None
        existing.alias = alias
        experiment.registered_model_name = model_name
        experiment.winner_candidate_id = run.candidate.id
        for item in experiment.runs:
            if item.candidate:
                item.candidate.is_winner = item.id == run.id
        db.commit()
        return existing
    try:
        client.create_registered_model(model_name)
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
    logged_model_uri = run.candidate.artifact_uri
    logged_model_id = (
        logged_model_uri.removeprefix("models:/")
        if logged_model_uri.startswith("models:/m-")
        else None
    )
    version = client.create_model_version(
        name=model_name,
        source=logged_model_uri,
        run_id=run.mlflow_run_id,
        model_id=logged_model_id,
        tags={
            "experiment_id": experiment.id,
            "dataset_id": experiment.dataset_id,
            "pipeline_id": experiment.pipeline_id,
            "model_id": run.model_id,
        },
    )
    client.set_registered_model_alias(model_name, alias, version.version)
    for previous in db.scalars(
        select(RegisteredModelReference).where(
            RegisteredModelReference.name == model_name,
            RegisteredModelReference.alias == alias,
        )
    ).all():
        previous.alias = None
    for item in experiment.runs:
        if item.candidate:
            item.candidate.is_winner = item.id == run.id
    experiment.winner_candidate_id = run.candidate.id
    experiment.registered_model_name = model_name
    reference = RegisteredModelReference(
        name=model_name,
        version=str(version.version),
        alias=alias,
        experiment_id=experiment.id,
        candidate_id=run.candidate.id,
        model_uri=f"models:/{model_name}/{version.version}",
        metrics=run.metrics,
    )
    db.add(reference)
    db.commit()
    return reference


def _execution_resources(db: Session, experiment_id: str):
    experiment = get_experiment(db, experiment_id)
    dataset = db.scalar(
        select(Dataset)
        .where(Dataset.id == experiment.dataset_id)
        .options(selectinload(Dataset.versions))
    )
    if not dataset:
        raise ResourceNotFoundError("Dataset no encontrado")
    pipeline_plugin = pipeline_registry.get(experiment.pipeline_id)
    dataset_version = _dataset_version(dataset, experiment.dataset_version)
    return experiment, dataset, dataset_version, pipeline_plugin


def initialize_experiment_execution(
    db: Session,
    settings: Settings,
    experiment_id: str,
    prefect_flow_run_id: str | None = None,
) -> dict[str, Any]:
    experiment, dataset, dataset_version, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    if experiment.status == "running":
        raise ValidationError("El experimento ya está en ejecución")
    if experiment.status in {"completed", "cancelled"}:
        raise ValidationError("El experimento ya terminó")
    experiment.status = "running"
    experiment.started_at = utcnow()
    experiment.completed_at = None
    experiment.error_message = None
    experiment.prefect_flow_run_id = prefect_flow_run_id or experiment.prefect_flow_run_id
    db.commit()
    client = _configure_mlflow(settings)
    parent_config = {
        "dataset_id": dataset.id,
        "dataset_fingerprint": dataset_version.fingerprint,
        "dataset_version": experiment.dataset_version,
        "pipeline_id": experiment.pipeline_id,
        "pipeline_version": pipeline_plugin.metadata.version,
        "pipeline_config": experiment.pipeline_config,
        "model_ids": experiment.model_ids,
        "model_parameters": experiment.model_parameters,
        "validation_strategy": experiment.validation_strategy,
        "validation_config": experiment.validation_config,
        "primary_metric": experiment.primary_metric,
        "random_seed": experiment.random_seed,
        "source_commit": _source_commit(),
        "dependency_versions": _dependency_versions(),
        "start_time": experiment.started_at,
    }
    tracking_experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if not tracking_experiment:
        raise ValidationError("MLflow no pudo crear el experimento de seguimiento")
    parent = client.create_run(
        tracking_experiment.experiment_id,
        tags={
            "mlflow.runName": experiment.name,
            "neuroops.run_type": "experiment",
            "neuroops.experiment_id": experiment.id,
        },
    )
    experiment.mlflow_parent_run_id = parent.info.run_id
    db.commit()
    for key, value in {
        "dataset_id": dataset.id,
        "dataset_fingerprint": parent_config["dataset_fingerprint"],
        "dataset_version": experiment.dataset_version,
        "pipeline_id": experiment.pipeline_id,
        "pipeline_version": pipeline_plugin.metadata.version,
        "validation_strategy": experiment.validation_strategy,
        "primary_metric": experiment.primary_metric,
        "random_seed": experiment.random_seed,
        "source_commit": parent_config["source_commit"],
        "python_version": platform.python_version(),
    }.items():
        client.log_param(parent.info.run_id, key, value)
    parent_dir = settings.artifacts_dir / experiment.id / "experiment"
    parent_dir.mkdir(parents=True, exist_ok=True)
    _write_json(parent_dir / "configuration.json", parent_config)
    _write_json(parent_dir / "environment.json", parent_config["dependency_versions"])
    client.log_artifacts(parent.info.run_id, str(parent_dir), artifact_path="experiment")
    return {
        **parent_config,
        "parent_run_id": parent.info.run_id,
        "dataset_path": dataset_version.storage_path,
        "candidates": [
            {"training_run_id": run.id, "model_id": run.model_id}
            for run in experiment.runs
        ],
    }


def validate_experiment_dataset(db: Session, experiment_id: str) -> dict[str, Any]:
    experiment, _dataset, dataset_version, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    _set_all_stages(db, experiment, "validate_dataset")
    return pipeline_plugin.validate_input(dataset_version.storage_path, experiment.pipeline_config)


def load_experiment_data(db: Session, experiment_id: str) -> Any:
    experiment, _dataset, dataset_version, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    _set_all_stages(db, experiment, "load_data")
    return pipeline_plugin.load_data(dataset_version.storage_path, experiment.pipeline_config)


def preprocess_experiment_data(db: Session, experiment_id: str, data: Any) -> Any:
    experiment, _dataset, _dataset_version_value, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    _set_all_stages(db, experiment, "preprocess")
    return pipeline_plugin.preprocess(data, experiment.pipeline_config)


def extract_experiment_features(db: Session, experiment_id: str, data: Any) -> Any:
    experiment, _dataset, _dataset_version_value, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    _set_all_stages(db, experiment, "extract_features")
    return pipeline_plugin.extract_features(data, experiment.pipeline_config)


def build_experiment_training_data(
    db: Session,
    experiment_id: str,
    data: Any,
) -> dict[str, Any]:
    experiment, _dataset, _dataset_version_value, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    _set_all_stages(db, experiment, "build_training_data")
    prepared = pipeline_plugin.build_training_data(data, experiment.pipeline_config)
    strategy, splits = make_validation_split(
        experiment.validation_strategy,
        prepared["X"],
        prepared["y"],
        prepared.get("groups"),
        experiment.validation_config,
        experiment.random_seed,
    )
    return {**prepared, "validation_strategy_name": strategy, "validation_splits": splits}


def execute_candidate(
    db: Session,
    settings: Settings,
    experiment_id: str,
    training_run_id: str,
    prepared: dict[str, Any],
    dataset_summary: dict[str, Any],
    parent_config: dict[str, Any],
) -> str:
    experiment, dataset, _dataset_version_value, pipeline_plugin = _execution_resources(
        db, experiment_id
    )
    run = next((item for item in experiment.runs if item.id == training_run_id), None)
    if not run:
        raise ResourceNotFoundError("Ejecución candidata no encontrada")
    if run.status == "completed" and run.candidate:
        return run.id
    if experiment.cancelled:
        run.status = "cancelled"
        run.stage = "cancelled"
        db.commit()
        return run.id
    run.status = "running"
    run.stage = "train_model"
    run.error_message = None
    db.commit()
    model_plugin = model_registry.get(run.model_id)
    _configure_mlflow(settings)
    try:
        with mlflow.start_run(
            run_name=run.model_id,
            tags={
                "mlflow.parentRunId": parent_config["parent_run_id"],
                "neuroops.run_type": "model_candidate",
                "neuroops.experiment_id": experiment.id,
                "neuroops.model_id": run.model_id,
            },
        ) as child:
            run.mlflow_run_id = child.info.run_id
            db.commit()
            estimator = Pipeline(
                [
                    (
                        "preprocessor",
                        clone(prepared["preprocessor"])
                        if prepared["preprocessor"] != "passthrough"
                        else "passthrough",
                    ),
                    ("model", model_plugin.build(run.parameters, experiment.random_seed)),
                ]
            )
            y_true, predictions, probabilities, final_estimator = _predict_for_validation(
                estimator,
                prepared["X"],
                prepared["y"],
                prepared.get("groups"),
                prepared["validation_strategy_name"],
                prepared["validation_splits"],
                model_plugin.metadata.supports_probability,
            )
            run.stage = "evaluate"
            result = evaluate_predictions(y_true, predictions, probabilities)
            run.metrics = result["metrics"]
            mlflow.log_params(
                {
                    "dataset_id": dataset.id,
                    "dataset_fingerprint": parent_config["dataset_fingerprint"],
                    "pipeline_id": experiment.pipeline_id,
                    "pipeline_version": pipeline_plugin.metadata.version,
                    "model_id": run.model_id,
                    "model_parameters": json.dumps(run.parameters, sort_keys=True),
                    "validation_strategy": experiment.validation_strategy,
                    "random_seed": experiment.random_seed,
                    "source_commit": parent_config["source_commit"],
                }
            )
            mlflow.log_metrics(run.metrics)
            run.stage = "save_artifacts"
            model_dir = settings.artifacts_dir / experiment.id / run.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            complete_config = {
                **parent_config,
                "model_id": run.model_id,
                "model_parameters": run.parameters,
            }
            _write_json(model_dir / "configuration.json", complete_config)
            _write_json(
                model_dir / "classification_report.json", result["classification_report"]
            )
            _write_json(model_dir / "feature_names.json", prepared["feature_names"])
            _write_json(model_dir / "dataset_summary.json", dataset_summary)
            _write_json(model_dir / "environment.json", parent_config["dependency_versions"])
            _plot_confusion(
                model_dir / "confusion_matrix.png",
                result["confusion_matrix"],
                [str(value) for value in sorted(set(y_true), key=str)],
            )
            has_roc_curve = _plot_roc(model_dir / "roc_curve.png", y_true, probabilities)
            mlflow.log_artifacts(str(model_dir), artifact_path="artifacts")
            logged_model = _log_model(final_estimator, prepared["X"])
            run.candidate = ModelCandidate(
                model_id=run.model_id,
                artifact_uri=logged_model.model_uri,
            )
            artifact_specs = [
                ("configuration.json", "configuration"),
                ("classification_report.json", "report"),
                ("feature_names.json", "features"),
                ("dataset_summary.json", "dataset_summary"),
                ("environment.json", "environment"),
                ("confusion_matrix.png", "plot"),
            ]
            if has_roc_curve:
                artifact_specs.append(("roc_curve.png", "plot"))
            for artifact_name, kind in artifact_specs:
                db.add(
                    ArtifactReference(
                        training_run_id=run.id,
                        name=artifact_name,
                        kind=kind,
                        uri=f"runs:/{run.mlflow_run_id}/artifacts/{artifact_name}",
                    )
                )
            run.status = "completed"
            run.stage = "completed"
            db.commit()
            return run.id
    except Exception as exc:
        run.status = "failed"
        run.stage = "failed"
        run.error_message = str(exc)[:2000]
        db.commit()
        raise


def finalize_experiment_execution(
    db: Session,
    settings: Settings,
    experiment_id: str,
    parent_config: dict[str, Any],
) -> Experiment:
    experiment = get_experiment(db, experiment_id)
    client = _configure_mlflow(settings)
    parent_run_id = parent_config["parent_run_id"]
    if experiment.cancelled:
        experiment.status = "cancelled"
        experiment.completed_at = utcnow()
        client.set_tag(parent_run_id, "neuroops.status", "cancelled")
        client.set_terminated(parent_run_id, status="KILLED")
        db.commit()
        return get_experiment(db, experiment_id)
    successful = [run for run in experiment.runs if run.status == "completed"]
    if not successful:
        raise ValidationError("Ningún modelo terminó correctamente")
    candidates = [run for run in successful if experiment.primary_metric in run.metrics]
    if not candidates:
        raise ValidationError(f"La métrica {experiment.primary_metric} no pudo calcularse")
    winner = max(candidates, key=lambda item: item.metrics[experiment.primary_metric])
    client.log_param(parent_run_id, "winner_model_id", winner.model_id)
    client.log_metric(parent_run_id, "winner_score", winner.metrics[experiment.primary_metric])
    register_candidate(db, settings, experiment, winner, "champion")
    execution_summary = {
        "status": "completed",
        "start_time": experiment.started_at,
        "end_time": utcnow(),
        "winner_model_id": winner.model_id,
        "winner_metric": experiment.primary_metric,
        "winner_score": winner.metrics[experiment.primary_metric],
    }
    summary_path = settings.artifacts_dir / experiment.id / "experiment/execution_summary.json"
    _write_json(summary_path, execution_summary)
    client.log_artifact(parent_run_id, str(summary_path), artifact_path="experiment")
    client.set_tag(parent_run_id, "neuroops.status", "completed")
    client.set_terminated(parent_run_id, status="FINISHED")
    experiment.status = "completed"
    experiment.completed_at = utcnow()
    db.commit()
    return get_experiment(db, experiment_id)


def fail_experiment_execution(
    db: Session,
    settings: Settings,
    experiment_id: str,
    error: Exception,
) -> None:
    experiment = get_experiment(db, experiment_id)
    if experiment.cancelled:
        experiment.status = "cancelled"
    else:
        experiment.status = "failed"
        experiment.error_message = f"{type(error).__name__}: {error}"[:4000]
    experiment.completed_at = utcnow()
    for run in experiment.runs:
        if run.status in {"queued", "running"}:
            run.status = "cancelled" if experiment.cancelled else "failed"
            run.stage = run.status
            run.error_message = None if experiment.cancelled else str(error)[:2000]
    db.commit()
    if experiment.mlflow_parent_run_id:
        try:
            client = _configure_mlflow(settings)
            client.set_tag(
                experiment.mlflow_parent_run_id,
                "neuroops.status",
                "cancelled" if experiment.cancelled else "failed",
            )
            client.set_terminated(
                experiment.mlflow_parent_run_id,
                status="KILLED" if experiment.cancelled else "FAILED",
            )
        except Exception:
            pass


def execute_experiment(db: Session, settings: Settings, experiment_id: str) -> Experiment:
    try:
        context = initialize_experiment_execution(db, settings, experiment_id)
        dataset_summary = validate_experiment_dataset(db, experiment_id)
        data = load_experiment_data(db, experiment_id)
        data = preprocess_experiment_data(db, experiment_id, data)
        data = extract_experiment_features(db, experiment_id, data)
        prepared = build_experiment_training_data(db, experiment_id, data)
        for candidate in context["candidates"]:
            try:
                execute_candidate(
                    db,
                    settings,
                    experiment_id,
                    candidate["training_run_id"],
                    prepared,
                    dataset_summary,
                    context,
                )
            except Exception:
                continue
        return finalize_experiment_execution(db, settings, experiment_id, context)
    except Exception as exc:
        fail_experiment_execution(db, settings, experiment_id, exc)
        raise


def cancel_experiment(db: Session, experiment_id: str) -> Experiment:
    experiment = get_experiment(db, experiment_id)
    if experiment.status in {"completed", "failed", "cancelled"}:
        raise ValidationError("El experimento ya terminó")
    experiment.cancelled = True
    experiment.status = "cancelled"
    experiment.completed_at = utcnow()
    for run in experiment.runs:
        if run.status in {"queued", "running"}:
            run.status = "cancelled"
            run.stage = "cancelled"
    db.commit()
    return get_experiment(db, experiment_id)


def rerun_experiment(db: Session, experiment_id: str) -> Experiment:
    source = get_experiment(db, experiment_id)
    payload = ExperimentCreate(
        name=f"{source.name} (reejecución)",
        dataset_id=source.dataset_id,
        dataset_version=source.dataset_version,
        pipeline_id=source.pipeline_id,
        registered_model_name=source.registered_model_name,
        pipeline_config=source.pipeline_config,
        models=[
            {"model_id": model_id, "parameters": source.model_parameters.get(model_id, {})}
            for model_id in source.model_ids
        ],
        validation_strategy=source.validation_strategy,
        validation_config=source.validation_config,
        primary_metric=source.primary_metric,
        random_seed=source.random_seed,
    )
    return create_experiment(db, payload)
