from __future__ import annotations

import io
import zipfile

import pytest

from app.core.errors import ValidationError
from app.services.file_security import safe_extract_zip, safe_filename, validate_extension


def test_safe_filename_and_extension():
    assert safe_filename("../../mi dataset.csv") == "mi_dataset.csv"
    assert validate_extension("data.CSV", "tabular") == ".csv"
    with pytest.raises(ValidationError):
        validate_extension("payload.py", "tabular")


def test_zip_slip_is_blocked(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../../outside.txt", "unsafe")
    with pytest.raises(ValidationError, match="ruta insegura"):
        safe_extract_zip(archive, tmp_path / "out", 1024)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_extracted_size_limit(tmp_path):
    archive = tmp_path / "large.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("dataset_description.json", io.BytesIO(b"x" * 100).getvalue())
    with pytest.raises(ValidationError, match="límite"):
        safe_extract_zip(archive, tmp_path / "out", 50)
