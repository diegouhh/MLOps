from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.jobs import _deployment_for
from app.services.predictions import _direct_eeg_result
from app.services.registry import _list_eeg_recordings


def test_applee_jobs_use_dedicated_deployments():
    settings = Settings(_env_file=None)
    assert _deployment_for(
        settings,
        kind="experiment",
        execution_profile="applee",
    ) == "NeuroOps experiment/neuroops-applee-experiments"
    assert _deployment_for(
        settings,
        kind="prediction",
        execution_profile="applee",
    ) == "NeuroOps prediction/neuroops-applee-predictions"
    assert _deployment_for(
        settings,
        kind="experiment",
        execution_profile="default",
    ) == settings.prefect_experiment_deployment


def test_direct_eeg_result_does_not_present_feature_row_as_epoch():
    result = _direct_eeg_result(
        "sub-01/eeg/sub-01_task-rest_eeg.edf",
        np.array(["A"]),
        np.array([[0.8, 0.2]]),
        np.array(["A", "B"]),
    )
    assert result["prediction"] == "A"
    assert result["aggregation"] == "direct"
    assert result["analysis_unit"] == "feature_row"
    assert result["units_analyzed"] == 1
    assert "epochs_analyzed" not in result
    assert result["probabilities"] == {"A": 0.8, "B": 0.2}


def test_eeg_recording_list_respects_pipeline_task_and_session(tmp_path: Path):
    for relative in (
        "sub-01/ses-V0/eeg/sub-01_ses-V0_task-rest_eeg.edf",
        "sub-01/ses-V1/eeg/sub-01_ses-V1_task-rest_eeg.edf",
        "sub-01/ses-V0/eeg/sub-01_ses-V0_task-motor_eeg.edf",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    recordings = _list_eeg_recordings(
        str(tmp_path),
        {"task": "rest", "session": "V0"},
    )
    assert len(recordings) == 1
    assert recordings[0]["task"] == "rest"
    assert recordings[0]["session"] == "V0"
