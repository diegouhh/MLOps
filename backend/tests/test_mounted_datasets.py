from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from app.core.errors import ValidationError
from app.services.datasets import (
    create_dataset_version,
    delete_dataset,
    discover_mounted_datasets,
    register_mounted_dataset,
)


def _bids_dataset(root: Path, name: str = "Demo EEG") -> Path:
    dataset = root / "ds-demo"
    eeg_dir = dataset / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (dataset / "dataset_description.json").write_text(
        json.dumps({"Name": name, "BIDSVersion": "1.10.0"}),
        encoding="utf-8",
    )
    (dataset / "participants.tsv").write_text(
        "participant_id\tgroup\nsub-01\tcontrol\nsub-02\tpatient\n",
        encoding="utf-8",
    )
    (eeg_dir / "sub-01_task-rest_eeg.set").write_bytes(b"fake-eeg")
    return dataset


def test_mounted_bids_is_discovered_registered_and_never_deleted(
    db, settings, test_environment
):
    mounted = _bids_dataset(test_environment / "datasets")
    candidates = discover_mounted_datasets(db, settings)
    assert candidates == [
        {
            "relative_path": "ds-demo",
            "name": "Demo EEG",
            "subject_count": 2,
            "eeg_file_count": 1,
            "size_bytes": candidates[0]["size_bytes"],
            "registered": False,
            "valid": True,
            "validation_error": None,
        }
    ]

    dataset = register_mounted_dataset(
        db,
        settings,
        name="EEG local",
        relative_path="ds-demo",
        description="Solo lectura",
    )
    version = dataset.versions[0]
    assert version.storage_path == str(mounted.resolve())
    assert version.schema_summary["source"] == "mounted"
    assert version.schema_summary["subject_count"] == 2
    assert discover_mounted_datasets(db, settings)[0]["registered"] is True

    with pytest.raises(ValidationError, match="solo lectura"):
        create_dataset_version(
            db,
            settings,
            dataset.id,
            filename="other.zip",
            source=io.BytesIO(b"unused"),
        )

    delete_dataset(db, settings, dataset.id)
    assert mounted.is_dir()
    assert (mounted / "participants.tsv").is_file()


@pytest.mark.parametrize("relative_path", ["../outside", "/tmp/outside", "."])
def test_mounted_dataset_rejects_unsafe_paths(db, settings, relative_path):
    with pytest.raises(ValidationError, match="ruta|carpeta"):
        register_mounted_dataset(
            db,
            settings,
            name="Unsafe",
            relative_path=relative_path,
            description=None,
        )


def test_mounted_dataset_requires_participants_and_eeg(db, settings, test_environment):
    invalid = test_environment / "datasets" / "invalid"
    invalid.mkdir()
    (invalid / "dataset_description.json").write_text(
        '{"Name":"Invalid"}',
        encoding="utf-8",
    )
    candidates = discover_mounted_datasets(db, settings)
    candidate = next(item for item in candidates if item["relative_path"] == "invalid")
    assert candidate["valid"] is False
    assert "participants.tsv" in candidate["validation_error"]
