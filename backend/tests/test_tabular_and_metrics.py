from __future__ import annotations

import pandas as pd
import pytest

from app.core.errors import ValidationError
from app.ml.evaluators.classification import evaluate_predictions, make_validation_split
from app.ml.pipelines.registry import pipeline_registry


def test_tabular_pipeline_builds_leakage_safe_transformer(demo_path):
    pipeline = pipeline_registry.get("tabular_basic")
    config = {"target_column": "species"}
    summary = pipeline.validate_input(str(demo_path), config)
    data = pipeline.load_data(str(demo_path), config)
    prepared = pipeline.build_training_data(data, config)
    assert summary["rows"] == 30
    assert prepared["preprocessor"].transformers
    assert "species" not in prepared["feature_names"]
    assert prepared["X"].shape == (30, 4)


def test_tabular_pipeline_rejects_missing_target(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="no existe"):
        pipeline_registry.get("tabular_basic").validate_input(str(path), {"target_column": "label"})


def test_group_validation_never_mixes_subjects():
    X = pd.DataFrame({"x": range(12)})
    y = pd.Series([0, 1] * 6)
    groups = pd.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3)
    _, splitter = make_validation_split("group_kfold", X, y, groups, {"n_splits": 4}, 42)
    for train, test in splitter.split(X, y, groups):
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))


def test_grouped_data_rejects_non_group_validation():
    X = pd.DataFrame({"x": range(12)})
    y = pd.Series([0, 1] * 6)
    groups = pd.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3)
    with pytest.raises(ValidationError, match="Se detectaron grupos"):
        make_validation_split(
            "train_test_split", X, y, groups, {"test_size": 0.25}, 42
        )


def test_seeded_split_is_reproducible():
    X = pd.DataFrame({"x": range(20)})
    y = pd.Series([0, 1] * 10)
    _, first = make_validation_split("train_test_split", X, y, None, {"test_size": 0.25}, 42)
    _, second = make_validation_split("train_test_split", X, y, None, {"test_size": 0.25}, 42)
    assert first[0][0].tolist() == second[0][0].tolist()
    assert first[0][1].tolist() == second[0][1].tolist()


def test_classification_metrics_are_complete():
    result = evaluate_predictions(
        [0, 0, 1, 1], [0, 1, 1, 1], [[0.8, 0.2], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]
    )
    assert {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc",
    } <= set(result["metrics"])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
