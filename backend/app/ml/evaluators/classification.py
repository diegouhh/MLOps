from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from app.core.errors import ValidationError

SUPPORTED_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
}


def make_validation_split(
    strategy: str,
    X: Any,
    y: Any,
    groups: Any,
    config: dict[str, Any],
    random_seed: int,
) -> tuple[str, Any]:
    folds = int(config.get("n_splits", 5))
    group_strategies = {"group_kfold", "stratified_group_kfold"}
    if groups is not None and strategy not in group_strategies:
        raise ValidationError(
            "Se detectaron grupos en los datos. Usa group_kfold o stratified_group_kfold "
            "para evitar mezclar el mismo sujeto o grupo entre entrenamiento y validación"
        )
    if strategy == "train_test_split":
        test_size = float(config.get("test_size", 0.2))
        if not 0.05 <= test_size <= 0.5:
            raise ValidationError("test_size debe estar entre 0.05 y 0.5")
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_seed, stratify=y
        )
        return strategy, [(train_idx, test_idx)]
    if folds < 2 or folds > 20:
        raise ValidationError("n_splits debe estar entre 2 y 20")
    if strategy == "stratified_kfold":
        return strategy, StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)
    if strategy in {"group_kfold", "stratified_group_kfold"}:
        if groups is None:
            raise ValidationError("La estrategia por grupos requiere group_column o sujetos EEG")
        if len(set(groups)) < folds:
            raise ValidationError("No hay suficientes grupos únicos para n_splits")
        splitter = (
            GroupKFold(n_splits=folds)
            if strategy == "group_kfold"
            else StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_seed)
        )
        return strategy, splitter
    raise ValidationError(f"Estrategia de validación no soportada: {strategy}")


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    probabilities: Any = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probabilities is not None:
        try:
            probability_matrix = np.asarray(probabilities, dtype=float)
            labels = np.unique(y_true)
            if len(labels) == 2:
                score = (
                    probability_matrix[:, 1]
                    if probability_matrix.ndim == 2
                    else probability_matrix
                )
                metrics["roc_auc"] = float(roc_auc_score(y_true, score))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        probability_matrix,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except (IndexError, ValueError):
            pass
    return metrics


def evaluate_predictions(y_true: Any, y_pred: Any, probabilities: Any = None) -> dict[str, Any]:
    return {
        "metrics": classification_metrics(y_true, y_pred, probabilities),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def aggregate_group_predictions(
    y_true: Any,
    y_pred: Any,
    groups: Any,
    probabilities: Any = None,
    probability_labels: Any = None,
) -> dict[str, Any]:
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    group_values = np.asarray(groups)
    if not (len(true_values) == len(predicted_values) == len(group_values)):
        raise ValidationError("Las predicciones agrupadas no tienen longitudes compatibles")

    probability_matrix = None
    labels = None
    if probabilities is not None and probability_labels is not None:
        probability_matrix = np.asarray(probabilities, dtype=float)
        labels = np.asarray(probability_labels)
        if probability_matrix.ndim != 2 or probability_matrix.shape[0] != len(true_values):
            probability_matrix = None
            labels = None

    ordered_groups = list(dict.fromkeys(group_values.tolist()))
    aggregated_true: list[Any] = []
    aggregated_predicted: list[Any] = []
    aggregated_probabilities: list[np.ndarray] = []

    for group in ordered_groups:
        positions = np.flatnonzero(group_values == group)
        group_true = np.unique(true_values[positions])
        if len(group_true) != 1:
            raise ValidationError(
                f"El grupo {group} contiene más de una etiqueta objetivo y no puede agregarse"
            )
        aggregated_true.append(group_true[0])

        if probability_matrix is not None and labels is not None:
            mean_probabilities = probability_matrix[positions].mean(axis=0)
            aggregated_probabilities.append(mean_probabilities)
            aggregated_predicted.append(labels[int(np.argmax(mean_probabilities))])
        else:
            values, counts = np.unique(predicted_values[positions], return_counts=True)
            aggregated_predicted.append(values[int(np.argmax(counts))])

    return {
        "groups": np.asarray(ordered_groups),
        "y_true": np.asarray(aggregated_true),
        "y_pred": np.asarray(aggregated_predicted),
        "probabilities": (
            np.vstack(aggregated_probabilities) if aggregated_probabilities else None
        ),
    }


def summarize_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    metric_names = sorted(
        {
            name
            for fold in folds
            for name, value in fold.get("metrics", {}).items()
            if isinstance(value, (int, float, np.number))
        }
    )
    summary: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [
            float(fold["metrics"][name])
            for fold in folds
            if name in fold.get("metrics", {})
        ]
        if not values:
            continue
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
        }
    return summary
