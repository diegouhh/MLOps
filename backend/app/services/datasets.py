from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import Settings
from app.core.errors import ResourceNotFoundError, ValidationError
from app.db.models import Dataset, DatasetVersion, Experiment
from app.services.file_security import (
    assert_within,
    safe_extract_zip,
    safe_filename,
    validate_extension,
)

EEG_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set", ".fif"}


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_mounted_path(settings: Settings, relative_path: str) -> tuple[Path, Path]:
    raw = Path(relative_path.strip())
    if not relative_path.strip() or raw.is_absolute() or raw in {Path("."), Path("..")}:
        raise ValidationError("La ruta debe ser relativa a la carpeta datasets")
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ValidationError("La ruta contiene segmentos no permitidos")
    root = settings.mounted_datasets_dir.resolve()
    target = (root / raw).resolve()
    if target == root or not _is_within(target, root):
        raise ValidationError("La ruta sale de la carpeta datasets")
    if not target.is_dir():
        raise ValidationError("La carpeta BIDS no existe dentro de datasets")
    return root, target


def _mounted_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not _is_within(resolved, root):
            raise ValidationError("El dataset contiene un enlace fuera de su carpeta")
        files.append(candidate)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _inspect_mounted_bids(settings: Settings, relative_path: str) -> dict:
    _mount_root, root = _resolve_mounted_path(settings, relative_path)
    description_path = root / "dataset_description.json"
    participants_path = root / "participants.tsv"
    if not description_path.is_file():
        raise ValidationError("La carpeta no contiene dataset_description.json")
    if not _is_within(description_path.resolve(), root):
        raise ValidationError("dataset_description.json apunta fuera del dataset")
    try:
        description = json.loads(description_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("dataset_description.json no es válido") from exc
    if not isinstance(description, dict):
        raise ValidationError("dataset_description.json debe contener un objeto JSON")
    if not participants_path.is_file():
        raise ValidationError("La carpeta BIDS no contiene participants.tsv")
    if not _is_within(participants_path.resolve(), root):
        raise ValidationError("participants.tsv apunta fuera del dataset")
    try:
        participants = pd.read_csv(participants_path, sep="\t")
    except Exception as exc:
        raise ValidationError("participants.tsv no se pudo leer") from exc
    if participants.empty:
        raise ValidationError("participants.tsv está vacío")

    files = _mounted_files(root)
    eeg_files = [
        path
        for path in files
        if "_eeg" in path.stem.lower() and path.suffix.lower() in EEG_EXTENSIONS
    ]
    if not eeg_files:
        raise ValidationError("No se encontraron archivos EEG compatibles con MNE-BIDS")

    digest = hashlib.sha256()
    digest.update(description_path.read_bytes())
    digest.update(participants_path.read_bytes())
    total_size = 0
    for path in files:
        size = path.stat().st_size
        total_size += size
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(size).encode("ascii"))

    relative = root.relative_to(settings.mounted_datasets_dir.resolve()).as_posix()
    return {
        "root": root,
        "relative_path": relative,
        "dataset_name": str(description.get("Name") or root.name),
        "subject_count": len(participants),
        "eeg_file_count": len(eeg_files),
        "columns": participants.columns.tolist(),
        "total_size_bytes": total_size,
        "fingerprint": digest.hexdigest(),
    }


def discover_mounted_datasets(db: Session, settings: Settings) -> list[dict]:
    mount_root = settings.mounted_datasets_dir
    if not mount_root.is_dir():
        return []
    registered_paths = {
        Path(path).resolve()
        for path in db.scalars(select(DatasetVersion.storage_path)).all()
    }
    candidates: list[dict] = []
    for child in sorted(mount_root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or not (child / "dataset_description.json").is_file():
            continue
        relative_path = child.name
        try:
            inspected = _inspect_mounted_bids(settings, relative_path)
            candidates.append(
                {
                    "relative_path": inspected["relative_path"],
                    "name": inspected["dataset_name"],
                    "subject_count": inspected["subject_count"],
                    "eeg_file_count": inspected["eeg_file_count"],
                    "size_bytes": inspected["total_size_bytes"],
                    "registered": inspected["root"] in registered_paths,
                    "valid": True,
                    "validation_error": None,
                }
            )
        except ValidationError as exc:
            candidates.append(
                {
                    "relative_path": relative_path,
                    "name": child.name,
                    "registered": child.resolve() in registered_paths,
                    "valid": False,
                    "validation_error": str(exc),
                }
            )
    return candidates


def register_mounted_dataset(
    db: Session,
    settings: Settings,
    *,
    name: str,
    relative_path: str,
    description: str | None,
) -> Dataset:
    inspected = _inspect_mounted_bids(settings, relative_path)
    storage_path = str(inspected["root"])
    if db.scalar(select(DatasetVersion.id).where(DatasetVersion.storage_path == storage_path)):
        raise ValidationError("Esta carpeta BIDS ya está registrada")
    dataset = Dataset(
        name=name.strip(),
        data_type="eeg_bids",
        description=description,
        status="ready",
        current_version=1,
    )
    summary = {
        "source": "mounted",
        "relative_path": inspected["relative_path"],
        "dataset_name": inspected["dataset_name"],
        "subject_count": inspected["subject_count"],
        "eeg_file_count": inspected["eeg_file_count"],
        "columns": inspected["columns"],
        "total_size_bytes": inspected["total_size_bytes"],
    }
    dataset.versions.append(
        DatasetVersion(
            version=1,
            fingerprint=inspected["fingerprint"],
            original_filename=f"mounted:{inspected['relative_path']}",
            storage_path=storage_path,
            size_bytes=inspected["total_size_bytes"],
            row_count=inspected["subject_count"],
            column_count=len(inspected["columns"]),
            schema_summary=json.loads(json.dumps(summary)),
        )
    )
    db.add(dataset)
    db.commit()
    return get_dataset(db, dataset.id)


def _copy_limited(source: BinaryIO, target: Path, max_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with target.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                output.close()
                target.unlink(missing_ok=True)
                raise ValidationError("El archivo supera el tamaño máximo permitido")
            digest.update(chunk)
            output.write(chunk)
    return size, digest.hexdigest()


def create_dataset(
    db: Session,
    settings: Settings,
    *,
    name: str,
    data_type: str,
    description: str | None,
    filename: str,
    source: BinaryIO,
) -> Dataset:
    suffix = validate_extension(filename, data_type)
    original_name = safe_filename(filename)
    dataset_id = str(uuid.uuid4())
    upload_root = settings.data_dir / "uploads" / dataset_id
    upload_root.mkdir(parents=True, exist_ok=False)
    stored_file = upload_root / f"source{suffix}"
    try:
        size, fingerprint = _copy_limited(source, stored_file, settings.max_upload_mb * 1024 * 1024)
        storage_path: Path = stored_file
        row_count = None
        column_count = None
        summary: dict = {}
        if data_type == "tabular":
            try:
                frame = pd.read_csv(stored_file)
            except Exception as exc:
                raise ValidationError("No fue posible analizar el CSV") from exc
            if frame.empty:
                raise ValidationError("El CSV está vacío")
            row_count, column_count = frame.shape
            summary = {
                "columns": frame.columns.tolist(),
                "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                "missing_values": int(frame.isna().sum().sum()),
            }
        else:
            extracted = upload_root / "bids"
            extracted.mkdir()
            storage_path = safe_extract_zip(
                stored_file, extracted, settings.max_extracted_mb * 1024 * 1024
            )
            summary = {"root": storage_path.name}
        dataset = Dataset(
            id=dataset_id,
            name=name.strip(),
            data_type=data_type,
            description=description,
            status="ready",
            current_version=1,
        )
        dataset.versions.append(
            DatasetVersion(
                version=1,
                fingerprint=fingerprint,
                original_filename=original_name,
                storage_path=str(storage_path.resolve()),
                size_bytes=size,
                row_count=row_count,
                column_count=column_count,
                schema_summary=json.loads(json.dumps(summary)),
            )
        )
        db.add(dataset)
        db.commit()
        return get_dataset(db, dataset_id)
    except Exception:
        shutil.rmtree(upload_root, ignore_errors=True)
        db.rollback()
        raise


def create_dataset_version(
    db: Session,
    settings: Settings,
    dataset_id: str,
    *,
    filename: str,
    source: BinaryIO,
) -> Dataset:
    dataset = get_dataset(db, dataset_id)
    if any(
        (version.schema_summary or {}).get("source") == "mounted"
        for version in dataset.versions
    ):
        raise ValidationError(
            "Los datasets montados son de solo lectura; registra otra carpeta "
            "para una nueva versión"
        )
    suffix = validate_extension(filename, dataset.data_type)
    original_name = safe_filename(filename)
    next_version = max((item.version for item in dataset.versions), default=0) + 1
    version_root = settings.data_dir / "uploads" / dataset.id / f"v{next_version}"
    version_root.mkdir(parents=True, exist_ok=False)
    stored_file = version_root / f"source{suffix}"
    try:
        size, fingerprint = _copy_limited(
            source, stored_file, settings.max_upload_mb * 1024 * 1024
        )
        if any(item.fingerprint == fingerprint for item in dataset.versions):
            raise ValidationError("Este archivo ya está registrado en el dataset")
        storage_path: Path = stored_file
        row_count = None
        column_count = None
        summary: dict = {}
        if dataset.data_type == "tabular":
            try:
                frame = pd.read_csv(stored_file)
            except Exception as exc:
                raise ValidationError("No fue posible analizar el CSV") from exc
            if frame.empty:
                raise ValidationError("El CSV está vacío")
            row_count, column_count = frame.shape
            summary = {
                "columns": frame.columns.tolist(),
                "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
                "missing_values": int(frame.isna().sum().sum()),
            }
        else:
            extracted = version_root / "bids"
            extracted.mkdir()
            storage_path = safe_extract_zip(
                stored_file, extracted, settings.max_extracted_mb * 1024 * 1024
            )
            summary = {"root": storage_path.name}
        dataset.versions.append(
            DatasetVersion(
                version=next_version,
                fingerprint=fingerprint,
                original_filename=original_name,
                storage_path=str(storage_path.resolve()),
                size_bytes=size,
                row_count=row_count,
                column_count=column_count,
                schema_summary=json.loads(json.dumps(summary)),
            )
        )
        dataset.current_version = next_version
        db.commit()
        return get_dataset(db, dataset.id)
    except Exception:
        shutil.rmtree(version_root, ignore_errors=True)
        db.rollback()
        raise


def get_dataset(db: Session, dataset_id: str) -> Dataset:
    dataset = db.scalar(
        select(Dataset).where(Dataset.id == dataset_id).options(selectinload(Dataset.versions))
    )
    if not dataset:
        raise ResourceNotFoundError("Dataset no encontrado")
    return dataset


def list_datasets(db: Session) -> list[Dataset]:
    return list(
        db.scalars(
            select(Dataset)
            .options(selectinload(Dataset.versions))
            .order_by(Dataset.created_at.desc())
        ).all()
    )


def delete_dataset(db: Session, settings: Settings, dataset_id: str) -> None:
    dataset = get_dataset(db, dataset_id)
    if db.scalar(select(Experiment.id).where(Experiment.dataset_id == dataset_id).limit(1)):
        raise ValidationError("No se puede eliminar un dataset utilizado por experimentos")
    upload_root = assert_within(
        settings.data_dir / "uploads" / dataset_id, settings.data_dir / "uploads"
    )
    db.delete(dataset)
    db.commit()
    shutil.rmtree(upload_root, ignore_errors=True)
