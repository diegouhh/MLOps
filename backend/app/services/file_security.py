from __future__ import annotations

import stat
import zipfile
from pathlib import Path, PurePosixPath

from app.core.errors import ValidationError

ALLOWED_EXTENSIONS = {"tabular": {".csv"}, "eeg_bids": {".zip"}}


def validate_extension(filename: str, data_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if data_type not in ALLOWED_EXTENSIONS:
        raise ValidationError("Tipo de dataset no soportado")
    if suffix not in ALLOWED_EXTENSIONS[data_type]:
        expected = ", ".join(sorted(ALLOWED_EXTENSIONS[data_type]))
        raise ValidationError(f"Extensión inválida para {data_type}; se permite {expected}")
    return suffix


def safe_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise ValidationError("Nombre de archivo inválido")
    cleaned = "".join(
        character if character.isalnum() or character in ".-_" else "_" for character in name
    )
    return cleaned[:255]


def _safe_member(member: zipfile.ZipInfo) -> PurePosixPath:
    normalized = PurePosixPath(member.filename.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValidationError("El ZIP contiene una ruta insegura")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValidationError("El ZIP no puede contener enlaces simbólicos")
    return normalized


def safe_extract_zip(zip_path: Path, destination: Path, max_extracted_bytes: int) -> Path:
    destination = destination.resolve()
    total_size = 0
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValidationError("El archivo no es un ZIP válido") from exc
    with archive:
        members = archive.infolist()
        if not members:
            raise ValidationError("El ZIP está vacío")
        for member in members:
            path = _safe_member(member)
            total_size += member.file_size
            if total_size > max_extracted_bytes:
                raise ValidationError("El contenido descomprimido supera el límite permitido")
            output = (destination / Path(*path.parts)).resolve()
            if destination not in output.parents and output != destination:
                raise ValidationError("El ZIP intenta escribir fuera del directorio asignado")
        archive.extractall(destination)
    entries = [entry for entry in destination.iterdir() if entry.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


def assert_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValidationError("Ruta de almacenamiento fuera del área permitida")
    return resolved
