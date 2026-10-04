from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.errors import PluginUnavailableError, ValidationError
from app.domain.plugins import PipelineMetadata, PipelinePlugin

APPLEE_REPOSITORY = "https://github.com/GNACo/APPLEE"
APPLEE_REVISION = "5b554417070aed47fcd240b6923ada68f62db029"
SUPPORTED_EEG_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set"}
DEFAULT_CHANNELS = ["FP1", "FP2", "C3", "C4", "O1", "O2", "P7", "P8"]
SUPPORTED_FEATURES = {"cohfreq", "sl", "power", "crossfreq", "entropy"}
DEFAULT_FEATURES = ["cohfreq", "sl", "power", "crossfreq", "entropy"]


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _raw_eeg_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*_eeg.*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EEG_EXTENSIONS:
            continue
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if "derivatives" not in relative.parts:
            files.append(path)
    return sorted(files)


def _entity_from_name(path: Path, key: str) -> str | None:
    prefix = f"{key}-"
    for token in path.stem.split("_"):
        if token.startswith(prefix) and len(token) > len(prefix):
            return token[len(prefix) :]
    for part in path.parts:
        if part.startswith(prefix) and len(part) > len(prefix):
            return part[len(prefix) :]
    return None


class AppleEPipeline(PipelinePlugin):
    """NeuroOps adapter for the APPLEE low-electrode EEG workflow.

    APPLEE itself runs in an isolated Python environment through a small subprocess
    adapter. The NeuroOps process never imports APPLEE or its historical dependency
    chain directly, which keeps this plugin removable and prevents dependency clashes
    with the core MLOps runtime.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        enabled = _env_enabled("NEUROOPS_APPLEE_ENABLED") if enabled is None else enabled
        self.metadata = PipelineMetadata(
            id="applee",
            name="applee",
            display_name="APPLEE",
            description=(
                "Preprocesamiento y extracción de características para EEG de baja densidad "
                "en formato BIDS."
            ),
            version="1.2-neuroops.1",
            data_type="eeg_bids",
            task_types=["classification"],
            input_label="BIDS EEG",
            category="Procesamiento EEG",
            available=enabled,
            unavailable_reason=(
                None
                if enabled
                else "Activa el perfil APPLEE para habilitar su worker y entorno aislado."
            ),
            required_packages=[
                "APPLEE",
                "sovaflow",
                "sovaharmony",
                "sovareject",
                "sovawica",
                "sovachronux",
            ],
            supported_models=[
                "logistic_regression",
                "random_forest",
                "svm",
                "knn",
                "gradient_boosting",
            ],
            execution_profile="applee",
            prediction_mode="direct_record",
            source_repository=APPLEE_REPOSITORY,
            source_revision=APPLEE_REVISION,
            config_schema={
                "type": "object",
                "required": ["target_column"],
                "properties": {
                    "target_column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Columna objetivo en participants.tsv.",
                    },
                    "task": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": (
                            "Tarea BIDS. Déjala vacía si el dataset contiene una sola tarea."
                        ),
                    },
                    "session": {
                        "type": ["string", "null"],
                        "default": None,
                        "description": "Sesión BIDS opcional.",
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": DEFAULT_CHANNELS,
                        "description": "Canales que APPLEE conservará para el análisis.",
                    },
                    "l_freq": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 100,
                        "default": 4.0,
                        "description": "Frecuencia inferior del filtrado en Hz.",
                    },
                    "h_freq": {
                        "type": "number",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 50.0,
                        "description": "Frecuencia superior del filtrado en Hz.",
                    },
                    "line_freq": {
                        "type": "number",
                        "minimum": 40,
                        "maximum": 70,
                        "default": 60.0,
                        "description": "Frecuencia de red eléctrica usada por APPLEE.",
                    },
                    "epoch": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 30,
                        "default": 2.0,
                        "description": "Duración de época para extracción de características.",
                    },
                    "features": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["cohfreq", "sl", "power", "crossfreq", "entropy"],
                        },
                        "default": DEFAULT_FEATURES,
                        "description": (
                            "Características APPLEE: cohfreq, sl, power, crossfreq y entropy."
                        ),
                    },
                    "prep": {
                        "type": "boolean",
                        "default": False,
                        "description": "Activa la etapa PREP incluida por APPLEE.",
                    },
                    "normalization": {
                        "enum": ["none", "zscore", "huber"],
                        "default": "none",
                        "description": "Normalización previa a la extracción de características.",
                    },
                    "resample": {
                        "type": ["number", "null"],
                        "minimum": 50,
                        "maximum": 2000,
                        "default": None,
                        "description": "Frecuencia de remuestreo opcional en Hz.",
                    },
                },
            },
            steps=[
                "validar BIDS",
                "ejecutar APPLEE",
                "control de artefactos",
                "extraer características",
                "preparar matriz por sujeto",
            ],
        )

    def _require_enabled(self) -> None:
        if not self.metadata.available:
            raise PluginUnavailableError(
                self.metadata.unavailable_reason or "APPLEE no está habilitado"
            )

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "target_column",
            "task",
            "session",
            "channels",
            "l_freq",
            "h_freq",
            "line_freq",
            "epoch",
            "features",
            "prep",
            "normalization",
            "resample",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValidationError(f"Parámetros APPLEE no permitidos: {sorted(unknown)}")

        target = str(config.get("target_column", "")).strip()
        if not target:
            raise ValidationError("target_column es obligatorio")

        channels = [
            str(value).strip().upper()
            for value in config.get("channels", DEFAULT_CHANNELS)
        ]
        channels = [value for value in channels if value]
        if not channels:
            raise ValidationError("APPLEE necesita al menos un canal")
        if len(channels) != len(set(channels)):
            raise ValidationError("La lista de canales APPLEE contiene duplicados")

        features = [
            str(value).strip().lower()
            for value in config.get("features", DEFAULT_FEATURES)
        ]
        if not features:
            raise ValidationError("Selecciona al menos una característica APPLEE")
        invalid_features = sorted(set(features) - SUPPORTED_FEATURES)
        if invalid_features:
            raise ValidationError(
                "Características APPLEE no soportadas: " + ", ".join(invalid_features)
            )
        features = list(dict.fromkeys(features))

        l_freq = float(config.get("l_freq", 4.0))
        h_freq = float(config.get("h_freq", 50.0))
        if not 0 < l_freq < h_freq:
            raise ValidationError("APPLEE requiere 0 < l_freq < h_freq")
        line_freq = float(config.get("line_freq", 60.0))
        epoch = float(config.get("epoch", 2.0))
        if epoch <= 0:
            raise ValidationError("epoch debe ser positivo")

        normalization = str(config.get("normalization", "none")).lower()
        if normalization not in {"none", "zscore", "huber"}:
            raise ValidationError("normalization debe ser none, zscore o huber")

        resample = config.get("resample")
        if resample in {"", None}:
            normalized_resample = None
        else:
            normalized_resample = float(resample)
            if normalized_resample <= 0:
                raise ValidationError("resample debe ser positivo")

        return {
            "target_column": target,
            "task": str(config.get("task") or "").strip() or None,
            "session": str(config.get("session") or "").strip() or None,
            "channels": channels,
            "l_freq": l_freq,
            "h_freq": h_freq,
            "line_freq": line_freq,
            "epoch": epoch,
            "features": features,
            "prep": bool(config.get("prep", False)),
            "normalization": normalization,
            "resample": normalized_resample,
        }

    def validate_input(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        self._require_enabled()
        normalized = self.validate_config(config)
        root = Path(dataset_path).resolve()
        if not root.is_dir() or not (root / "dataset_description.json").is_file():
            raise ValidationError("APPLEE requiere una raíz BIDS válida")
        participants_path = root / "participants.tsv"
        if not participants_path.is_file():
            raise ValidationError("APPLEE requiere participants.tsv")
        participants = pd.read_csv(participants_path, sep="\t")
        if "participant_id" not in participants.columns:
            raise ValidationError("participants.tsv debe contener participant_id")
        if normalized["target_column"] not in participants.columns:
            raise ValidationError(
                f"participants.tsv no contiene {normalized['target_column']}"
            )

        recordings = _raw_eeg_files(root)
        if normalized["task"]:
            recordings = [
                path for path in recordings if _entity_from_name(path, "task") == normalized["task"]
            ]
        if normalized["session"]:
            recordings = [
                path
                for path in recordings
                if _entity_from_name(path, "ses") == normalized["session"]
            ]
        if not recordings:
            raise ValidationError("No se encontraron registros EEG compatibles para APPLEE")

        tasks = sorted(
            {
                task
                for task in (_entity_from_name(path, "task") for path in recordings)
                if task
            }
        )
        if not normalized["task"] and len(tasks) > 1:
            raise ValidationError(
                "El dataset contiene varias tareas BIDS. Selecciona task en la configuración APPLEE"
            )
        sessions = sorted(
            {
                session
                for session in (_entity_from_name(path, "ses") for path in recordings)
                if session
            }
        )
        if not normalized["session"] and len(sessions) > 1:
            raise ValidationError(
                "El dataset contiene varias sesiones BIDS. Selecciona session en la configuración "
                "APPLEE"
            )
        return {
            "subjects": int(participants.shape[0]),
            "eeg_files": len(recordings),
            "task": normalized["task"] or (tasks[0] if tasks else None),
            "session": normalized["session"] or (sessions[0] if sessions else None),
            "applee_revision": APPLEE_REVISION,
        }

    def load_data(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        summary = self.validate_input(dataset_path, config)
        root = Path(dataset_path).resolve()
        return {
            "dataset_path": str(root),
            "participants": pd.read_csv(root / "participants.tsv", sep="\t"),
            "input_summary": summary,
        }

    def _runtime_paths(self) -> tuple[Path, Path]:
        python_path = Path(
            os.getenv("NEUROOPS_APPLEE_RUNTIME_PYTHON", "/opt/applee-venv/bin/python")
        )
        runner_path = Path(
            os.getenv("NEUROOPS_APPLEE_RUNNER", "/app/applee_runtime/runner.py")
        )
        if not python_path.is_file():
            raise PluginUnavailableError(
                f"No se encontró el intérprete aislado de APPLEE en {python_path}"
            )
        if not runner_path.is_file():
            raise PluginUnavailableError(f"No se encontró el adaptador APPLEE en {runner_path}")
        return python_path, runner_path

    def _run_runtime(
        self,
        dataset_path: str,
        config: dict[str, Any],
        recording: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        self._require_enabled()
        normalized = self.validate_config(config)
        python_path, runner_path = self._runtime_paths()
        work_root_value = os.getenv("NEUROOPS_APPLEE_WORK_DIR")
        temporary_kwargs: dict[str, Any] = {"prefix": "neuroops-applee-"}
        if work_root_value:
            work_root = Path(work_root_value)
            work_root.mkdir(parents=True, exist_ok=True)
            temporary_kwargs["dir"] = str(work_root)
        with tempfile.TemporaryDirectory(**temporary_kwargs) as temporary:
            temporary_path = Path(temporary)
            request_path = temporary_path / "request.json"
            output_path = temporary_path / "features.csv"
            summary_path = temporary_path / "summary.json"
            request_path.write_text(
                json.dumps(
                    {
                        "dataset_path": str(Path(dataset_path).resolve()),
                        "config": normalized,
                        "recording": recording,
                        "workspace": str(temporary_path / "workspace"),
                        "output_csv": str(output_path),
                        "summary_json": str(summary_path),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [str(python_path), str(runner_path), str(request_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                details = (completed.stderr or completed.stdout or "Error desconocido").strip()
                details = details[-4000:]
                raise ValidationError(f"APPLEE no pudo procesar el registro: {details}")
            if not output_path.is_file():
                raise ValidationError("APPLEE terminó sin producir la matriz de características")
            frame = pd.read_csv(output_path)
            if frame.empty:
                raise ValidationError("APPLEE produjo una matriz de características vacía")
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.is_file()
                else {}
            )
            return frame, summary

    def preprocess(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        frame, summary = self._run_runtime(data["dataset_path"], config)
        return {**data, "features": frame, "applee_summary": summary}

    def extract_features(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        frame = data.get("features")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValidationError("APPLEE no produjo características utilizables")
        return data

    def build_training_data(self, data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        normalized = self.validate_config(config)
        features = data["features"].copy()
        participants = data["participants"].copy()
        if "subject" not in features.columns:
            raise ValidationError("La salida de APPLEE no contiene la columna subject")

        features["subject_key"] = features["subject"].astype(str).str.removeprefix("sub-")
        participants["subject_key"] = (
            participants["participant_id"].astype(str).str.removeprefix("sub-")
        )
        frame = features.merge(
            participants[["subject_key", normalized["target_column"]]],
            on="subject_key",
            how="inner",
        )
        if frame.empty:
            raise ValidationError(
                "No fue posible asociar las características APPLEE con participants.tsv"
            )
        frame = frame.dropna(subset=[normalized["target_column"]]).reset_index(drop=True)
        if frame.empty:
            raise ValidationError("No hay sujetos con una etiqueta objetivo válida para APPLEE")
        y = frame[normalized["target_column"]]
        if y.nunique(dropna=True) < 2:
            raise ValidationError("APPLEE necesita al menos dos clases objetivo para clasificación")
        groups = frame["subject_key"]
        X = frame.drop(
            columns=[normalized["target_column"], "subject", "subject_key"],
            errors="ignore",
        )
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.shape[1] == 0:
            raise ValidationError("APPLEE no produjo variables numéricas para entrenar")
        preprocessor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        return {
            "X": X,
            "y": y,
            "groups": groups,
            "evaluation_groups": groups,
            "evaluation_unit": "subject",
            "sample_unit": "subject",
            "preprocessor": preprocessor,
            "feature_names": X.columns.tolist(),
            "numeric_features": X.columns.tolist(),
            "categorical_features": [],
            "pipeline_runtime_summary": data.get("applee_summary", {}),
        }

    def build_prediction_features(
        self,
        dataset_path: str,
        config: dict[str, Any],
        recording: str,
    ) -> pd.DataFrame:
        root = Path(dataset_path).resolve()
        raw_relative = Path(recording)
        if raw_relative.is_absolute() or ".." in raw_relative.parts:
            raise ValidationError("El registro EEG seleccionado no es válido")
        selected = (root / raw_relative).resolve()
        try:
            selected.relative_to(root)
        except ValueError as exc:
            raise ValidationError("El registro EEG sale del dataset") from exc
        if not selected.is_file() or selected.suffix.lower() not in SUPPORTED_EEG_EXTENSIONS:
            raise ValidationError("El registro EEG no es compatible con APPLEE")

        frame, _summary = self._run_runtime(dataset_path, config, recording=recording)
        frame = frame.drop(columns=["subject"], errors="ignore")
        frame = frame.apply(pd.to_numeric, errors="coerce")
        if frame.empty:
            raise ValidationError("APPLEE no produjo características para la predicción")
        return frame
