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


def evaluate_predictions(y_true: Any, y_pred: Any, probabilities: Any = None) -> dict[str, Any]:
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
            probabilities = np.asarray(probabilities)
            labels = np.unique(y_true)
            if len(labels) == 2:
                score = probabilities[:, 1] if np.ndim(probabilities) == 2 else probabilities
                metrics["roc_auc"] = float(roc_auc_score(y_true, score))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }
