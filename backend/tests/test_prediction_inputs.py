from __future__ import annotations

from pathlib import Path

from app.db.models import ModelCandidate, RegisteredModelReference
from app.schemas.api import ExperimentCreate
from app.services.experiments import create_experiment
from app.services.predictions import _aggregate_eeg_result
from app.services.registry import _list_eeg_recordings, reference_input_schema


def _registered_reference(db, dataset_id: str):
    experiment = create_experiment(
        db,
        ExperimentCreate(
            name="Predicción sencilla",
            dataset_id=dataset_id,
            pipeline_id="tabular_basic",
            registered_model_name="easy-prediction-model",
            pipeline_config={"target_column": "species"},
            models=[{"model_id": "logistic_regression", "parameters": {}}],
        ),
    )
    run = experiment.runs[0]
    run.status = "completed"
    run.mlflow_run_id = "run-id"
    run.metrics = {"f1_macro": 0.9}
    run.candidate = ModelCandidate(model_id=run.model_id, artifact_uri="models:/m-test")
    db.commit()
    reference = RegisteredModelReference(
        name="easy-prediction-model",
        version="1",
        alias="champion",
        experiment_id=experiment.id,
        candidate_id=run.candidate.id,
        model_uri="models:/easy-prediction-model/1",
        metrics=run.metrics,
    )
    db.add(reference)
    db.commit()
    return reference


def test_tabular_schema_preloads_real_examples(db, demo_dataset):
    reference = _registered_reference(db, demo_dataset.id)
    schema = reference_input_schema(db, reference)

    assert schema["input_mode"] == "tabular"
    assert schema["examples"]
    assert schema["example"] == schema["examples"][0]
    assert "species" not in schema["example"]
    assert set(schema["example"]) == {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    }


def test_eeg_recording_discovery_returns_friendly_metadata(tmp_path: Path):
    root = tmp_path / "bids"
    eeg = root / "sub-01" / "ses-02" / "eeg"
    eeg.mkdir(parents=True)
    recording = eeg / "sub-01_ses-02_task-rest_run-1_eeg.edf"
    recording.write_bytes(b"test")
    derivative = root / "derivatives" / "sub-01" / "eeg"
    derivative.mkdir(parents=True)
    (derivative / "sub-01_task-rest_eeg.edf").write_bytes(b"ignored")

    recordings = _list_eeg_recordings(str(root))

    assert len(recordings) == 1
    assert recordings[0]["value"] == "sub-01/ses-02/eeg/sub-01_ses-02_task-rest_run-1_eeg.edf"
    assert recordings[0]["subject"] == "01"
    assert recordings[0]["session"] == "02"
    assert recordings[0]["task"] == "rest"
    assert recordings[0]["run"] == "1"
    assert recordings[0]["label"] == "Sujeto 01 - Sesión 02 - Tarea rest - Run 1"


def test_eeg_prediction_result_is_aggregated_for_non_expert_users():
    result = _aggregate_eeg_result(
        "sub-01/eeg/sub-01_task-rest_eeg.edf",
        ["control", "control", "case"],
        [[0.8, 0.2], [0.7, 0.3], [0.4, 0.6]],
        ["control", "case"],
    )

    assert result["prediction"] == "control"
    assert result["epochs_analyzed"] == 3
    assert result["class_distribution"] == {"control": 2, "case": 1}
    assert result["aggregation"] == "majority_vote"
    assert result["mean_probabilities"]["control"] > result["mean_probabilities"]["case"]
