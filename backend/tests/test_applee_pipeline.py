from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.core.errors import ValidationError
from app.ml.pipelines.applee import AppleEPipeline, DEFAULT_CHANNELS


def _bids_fixture(root: Path) -> Path:
    (root / "dataset_description.json").write_text(
        '{"Name":"NeuroOps APPLEE test","BIDSVersion":"1.9.0"}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"participant_id": "sub-01", "group": "A"},
            {"participant_id": "sub-02", "group": "B"},
        ]
    ).to_csv(root / "participants.tsv", sep="\t", index=False)
    for subject in ("01", "02"):
        eeg_dir = root / f"sub-{subject}" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        (eeg_dir / f"sub-{subject}_task-rest_eeg.edf").write_bytes(b"not-a-real-edf")
    return root


def test_applee_metadata_is_isolated_and_removable():
    pipeline = AppleEPipeline(enabled=True)
    assert pipeline.metadata.id == "applee"
    assert pipeline.metadata.execution_profile == "applee"
    assert pipeline.metadata.prediction_mode == "direct_record"
    assert pipeline.metadata.source_revision
    assert pipeline.metadata.category == "Procesamiento EEG"
    assert "cient" not in pipeline.metadata.description.lower()


def test_applee_configuration_defaults_and_allowlist():
    pipeline = AppleEPipeline(enabled=True)
    config = pipeline.validate_config({"target_column": "group"})
    assert config["channels"] == DEFAULT_CHANNELS
    assert config["features"] == ["cohfreq", "sl", "power", "crossfreq", "entropy"]
    assert config["l_freq"] == 4.0
    assert config["h_freq"] == 50.0
    assert config["epoch"] == 2.0

    with pytest.raises(ValidationError, match="no soportadas"):
        pipeline.validate_config({"target_column": "group", "features": ["made_up"]})
    with pytest.raises(ValidationError, match="l_freq"):
        pipeline.validate_config({"target_column": "group", "l_freq": 60, "h_freq": 50})
    with pytest.raises(ValidationError, match="no permitidos"):
        pipeline.validate_config({"target_column": "group", "python_code": "print(1)"})


def test_applee_validates_bids_target_and_task(tmp_path: Path):
    root = _bids_fixture(tmp_path)
    pipeline = AppleEPipeline(enabled=True)
    summary = pipeline.validate_input(str(root), {"target_column": "group"})
    assert summary["subjects"] == 2
    assert summary["eeg_files"] == 2
    assert summary["task"] == "rest"

    with pytest.raises(ValidationError, match="no contiene missing"):
        pipeline.validate_input(str(root), {"target_column": "missing"})
