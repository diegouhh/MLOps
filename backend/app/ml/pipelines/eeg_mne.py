from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import PluginUnavailableError, ValidationError
from app.domain.plugins import PipelineMetadata, PipelinePlugin

SUPPORTED_EEG_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set"}


def _is_raw_eeg_file(root: Path, path: Path) -> bool:
    """Return whether path is an existing raw EEG file inside the BIDS root."""
    if not path.is_file() or path.suffix.lower() not in SUPPORTED_EEG_EXTENSIONS:
        return False
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return "derivatives" not in relative.parts


class BasicEegMnePipeline(PipelinePlugin):
    def __init__(self) -> None:
        available = bool(importlib.util.find_spec("mne") and importlib.util.find_spec("mne_bids"))
        self.metadata = PipelineMetadata(
            id="eeg_mne_basic",
            name="eeg_mne_basic",
            display_name="EEG básico con MNE-BIDS",
            description="Validación BIDS, filtrado, épocas y potencia espectral por bandas.",
            version="1.0.0",
            data_type="eeg_bids",
            available=available,
            unavailable_reason=None if available else "Faltan MNE y MNE-BIDS.",
            required_packages=["mne", "mne-bids"],
            supported_models=[
                "logistic_regression",
                "random_forest",
                "svm",
                "knn",
                "gradient_boosting",
            ],
            config_schema={
                "type": "object",
                "required": ["target_column"],
                "properties": {
                    "target_column": {"type": "string"},
                    "subjects": {"type": "array", "items": {"type": "string"}},
                    "sessions": {"type": "array", "items": {"type": "string"}},
                    "tasks": {"type": "array", "items": {"type": "string"}},
                    "channels": {"type": "array", "items": {"type": "string"}},
                    "l_freq": {"type": "number", "default": 1.0},
                    "h_freq": {"type": "number", "default": 45.0},
                    "notch_freq": {"type": ["number", "null"], "default": 50.0},
                    "reference": {"type": ["string", "null"], "default": "average"},
                    "epoch_duration": {"type": "number", "default": 2.0},
                },
            },
            steps=[
                "validar BIDS",
                "cargar EEG",
                "filtrar",
                "referenciar",
                "segmentar",
                "extraer PSD",
            ],
        )

    def _require(self) -> None:
        if not self.metadata.available:
            raise PluginUnavailableError(
                self.metadata.unavailable_reason or "Plugin EEG no disponible"
            )

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "target_column",
            "subjects",
            "sessions",
            "tasks",
            "channels",
            "l_freq",
            "h_freq",
            "notch_freq",
            "reference",
            "epoch_duration",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValidationError(f"Parámetros EEG no permitidos: {sorted(unknown)}")
        target = str(config.get("target_column", "")).strip()
        if not target:
            raise ValidationError("target_column es obligatorio y debe existir en participants.tsv")
        normalized = {
            "target_column": target,
            "subjects": list(config.get("subjects", [])),
            "sessions": list(config.get("sessions", [])),
            "tasks": list(config.get("tasks", [])),
            "channels": list(config.get("channels", [])),
            "l_freq": float(config.get("l_freq", 1.0)),
            "h_freq": float(config.get("h_freq", 45.0)),
            "notch_freq": config.get("notch_freq", 50.0),
            "reference": config.get("reference", "average"),
            "epoch_duration": float(config.get("epoch_duration", 2.0)),
        }
        if not 0 <= normalized["l_freq"] < normalized["h_freq"]:
            raise ValidationError("Se requiere 0 <= l_freq < h_freq")
        if normalized["epoch_duration"] <= 0:
            raise ValidationError("epoch_duration debe ser positivo")
        return normalized

    def validate_input(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        self._require()
        config = self.validate_config(config)
        root = Path(dataset_path)
        if not root.is_dir() or not (root / "dataset_description.json").is_file():
            raise ValidationError("La entrada no es una raíz BIDS válida")
        participants = root / "participants.tsv"
        if not participants.is_file():
            raise ValidationError("BIDS debe contener participants.tsv")
        table = pd.read_csv(participants, sep="\t")
        if config["target_column"] not in table.columns:
            raise ValidationError(f"participants.tsv no contiene {config['target_column']}")
        eeg_files = [path for path in root.rglob("*_eeg.*") if _is_raw_eeg_file(root, path)]
        if not eeg_files:
            raise ValidationError("No se encontraron archivos EEG compatibles en BIDS")
        return {"subjects": int(table.shape[0]), "eeg_files": len(eeg_files)}

    def load_data(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        self.validate_input(dataset_path, config)
        from mne_bids import find_matching_paths, read_raw_bids

        root = Path(dataset_path)
        discovered_paths = find_matching_paths(
            root,
            subjects=config.get("subjects") or None,
            sessions=config.get("sessions") or None,
            tasks=config.get("tasks") or None,
            datatypes="eeg",
            suffixes="eeg",
            extensions=sorted(SUPPORTED_EEG_EXTENSIONS),
            ignore_json=True,
        )
        paths = [
            bids_path
            for bids_path in discovered_paths
            if _is_raw_eeg_file(root, Path(bids_path.fpath))
        ]
        if not paths:
            raise ValidationError("La selección no coincide con registros EEG")
        participants = pd.read_csv(root / "participants.tsv", sep="\t")
        return {
            "root": root,
            "paths": paths,
            "participants": participants,
            "reader": read_raw_bids,
        }

    def preprocess(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        processed = []
        for bids_path in data["paths"]:
            raw = data["reader"](
                bids_path=bids_path, verbose="ERROR", extra_params={"preload": True}
            )
            if config.get("channels"):
                raw.pick(config["channels"])
            raw.filter(config["l_freq"], config["h_freq"], verbose="ERROR")
            if config.get("notch_freq"):
                raw.notch_filter(float(config["notch_freq"]), verbose="ERROR")
            if config.get("reference"):
                raw.set_eeg_reference(config["reference"], verbose="ERROR")
            processed.append((bids_path, raw))
        return {**data, "processed": processed}

    def extract_features(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        import mne

        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
        rows: list[dict[str, Any]] = []
        for bids_path, raw in data["processed"]:
            epochs = mne.make_fixed_length_epochs(
                raw, duration=config["epoch_duration"], preload=True, verbose="ERROR"
            )
            spectrum = epochs.compute_psd(method="welch", fmin=1, fmax=45, verbose="ERROR")
            psds, freqs = spectrum.get_data(return_freqs=True)
            subject = str(bids_path.subject)
            for epoch_index, epoch_psd in enumerate(psds):
                row: dict[str, Any] = {"subject": subject, "epoch": epoch_index}
                for band, (low, high) in bands.items():
                    mask = (freqs >= low) & (freqs < high)
                    for channel_index, channel in enumerate(raw.ch_names):
                        row[f"{channel}_{band}"] = float(np.mean(epoch_psd[channel_index, mask]))
                rows.append(row)
        if not rows:
            raise ValidationError("No se pudieron producir épocas EEG")
        return {**data, "features": pd.DataFrame(rows)}

    def build_training_data(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        participants = data["participants"].copy()
        participants["subject"] = participants["participant_id"].str.removeprefix("sub-")
        frame = data["features"].merge(
            participants[["subject", config["target_column"]]], on="subject", how="inner"
        )
        if frame.empty:
            raise ValidationError("No fue posible asociar épocas con participants.tsv")
        y = frame[config["target_column"]]
        groups = frame["subject"]
        X = frame.drop(columns=[config["target_column"], "subject", "epoch"])
        return {
            "X": X,
            "y": y,
            "groups": groups,
            "preprocessor": "passthrough",
            "feature_names": X.columns.tolist(),
            "numeric_features": X.columns.tolist(),
            "categorical_features": [],
        }

    def build_prediction_features(
        self,
        dataset_path: str,
        config: dict[str, Any],
        recording: str,
    ) -> pd.DataFrame:
        """Apply the training EEG preprocessing to one BIDS recording for inference."""
        self._require()
        normalized = self.validate_config(config)
        root = Path(dataset_path).resolve()
        if not root.is_dir():
            raise ValidationError("El dataset EEG ya no está disponible")
        raw_relative = Path(recording)
        if raw_relative.is_absolute() or ".." in raw_relative.parts:
            raise ValidationError("El registro EEG seleccionado no es válido")
        selected_file = (root / raw_relative).resolve()
        try:
            selected_file.relative_to(root)
        except ValueError as exc:
            raise ValidationError("El registro EEG sale del dataset") from exc
        if not _is_raw_eeg_file(root, selected_file):
            raise ValidationError("El archivo seleccionado no es un registro EEG compatible")

        from mne_bids import find_matching_paths, read_raw_bids

        discovered = find_matching_paths(
            root,
            datatypes="eeg",
            suffixes="eeg",
            extensions=sorted(SUPPORTED_EEG_EXTENSIONS),
            ignore_json=True,
        )
        bids_path = next(
            (
                candidate
                for candidate in discovered
                if Path(candidate.fpath).resolve() == selected_file
            ),
            None,
        )
        if bids_path is None:
            raise ValidationError("El registro seleccionado no pudo resolverse como BIDS EEG")

        data = {"root": root, "paths": [bids_path], "reader": read_raw_bids}
        processed = self.preprocess(data, normalized)
        featured = self.extract_features(processed, normalized)
        frame = featured["features"].drop(columns=["subject", "epoch"], errors="ignore")
        if frame.empty:
            raise ValidationError("El registro EEG no produjo características para inferencia")
        return frame
