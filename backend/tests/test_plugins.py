from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.ml.models.registry import model_registry
from app.ml.pipelines.registry import pipeline_registry


def test_pipeline_registry_discovers_core_and_optional_plugins():
    plugins = {plugin.metadata.id: plugin for plugin in pipeline_registry.list()}
    assert set(plugins) == {"tabular_basic", "eeg_mne_basic", "sovaharmony_legacy"}
    assert plugins["tabular_basic"].metadata.available is True
    assert plugins["eeg_mne_basic"].metadata.available is True
    assert plugins["eeg_mne_basic"].metadata.required_packages == ["mne", "mne-bids"]
    assert plugins["sovaharmony_legacy"].metadata.available is False
    assert plugins["sovaharmony_legacy"].metadata.unavailable_reason


def test_model_registry_contains_five_safe_models():
    assert {plugin.metadata.id for plugin in model_registry.list()} == {
        "logistic_regression",
        "random_forest",
        "svm",
        "knn",
        "gradient_boosting",
    }


def test_model_parameter_allowlist_and_ranges():
    plugin = model_registry.get("random_forest")
    with pytest.raises(ValidationError, match="no permitidos"):
        plugin.validate_parameters({"python_code": "print('unsafe')"})
    with pytest.raises(ValidationError, match=">="):
        plugin.validate_parameters({"n_estimators": 1})
    assert plugin.validate_parameters({"n_estimators": 20})["n_estimators"] == 20


def test_pipeline_configuration_validation():
    pipeline = pipeline_registry.get("tabular_basic")
    with pytest.raises(ValidationError, match="obligatorio"):
        pipeline.validate_config({})
    with pytest.raises(ValidationError, match="no permitidos"):
        pipeline.validate_config({"target_column": "species", "callable": "os.system"})
    assert pipeline.validate_config({"target_column": "species"})["scale_numeric"] is True

# NeuroOps catalogs v2


def test_catalog_metadata_is_descriptive_and_filterable():
    tabular = pipeline_registry.get("tabular_basic").metadata
    eeg = pipeline_registry.get("eeg_mne_basic").metadata
    logistic = model_registry.get("logistic_regression").metadata
    random_forest = model_registry.get("random_forest").metadata

    assert tabular.task_types == ["classification"]
    assert tabular.input_label == "CSV tabular"
    assert tabular.category == "Preparación tabular"
    assert eeg.input_label == "BIDS EEG"

    assert logistic.family == "linear"
    assert logistic.interpretability == "high"
    assert logistic.compute_cost == "low"
    assert logistic.strengths
    assert logistic.limitations

    assert random_forest.family == "ensemble"
    assert random_forest.interpretability == "medium"
    assert random_forest.compute_cost == "medium"
