from __future__ import annotations

import importlib.util
from typing import Any

from app.core.errors import PluginUnavailableError, ValidationError
from app.domain.plugins import PipelineMetadata, PipelinePlugin


class SovaharmonyPipeline(PipelinePlugin):
    def __init__(self) -> None:
        packages = ["sovaharmony", "sovaflow", "sovachronux", "sovareject", "sovawica"]
        missing = [package for package in packages if importlib.util.find_spec(package) is None]
        available = not missing
        self.metadata = PipelineMetadata(
            id="sovaharmony_legacy",
            name="sovaharmony_legacy",
            display_name="Sovaharmony (opcional)",
            description=(
                "Adaptador diferido para compatibilidad con el pipeline científico heredado."
            ),
            version="0.1.0",
            data_type="eeg_bids",
            task_types=["preprocessing"],
            input_label="BIDS EEG",
            category="Preprocesamiento EEG",
            available=available,
            unavailable_reason=None
            if available
            else f"Dependencias opcionales ausentes: {', '.join(missing)}.",
            required_packages=packages,
            supported_models=[],
            config_schema={"type": "object", "additionalProperties": False},
            steps=["adaptador legado"],
        )

    def _unavailable(self) -> None:
        if not self.metadata.available:
            raise PluginUnavailableError(
                self.metadata.unavailable_reason or "Sovaharmony no disponible"
            )

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        if config:
            raise ValidationError("El adaptador aún no expone parámetros seguros")
        return {}

    def validate_input(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        self._unavailable()
        return {"path": dataset_path}

    def load_data(self, dataset_path: str, config: dict[str, Any]) -> Any:
        self._unavailable()
        from sovaharmony.pipeline import pipeline

        return {"pipeline": pipeline, "dataset_path": dataset_path}

    def preprocess(self, data: Any, config: dict[str, Any]) -> Any:
        raise PluginUnavailableError(
            "El adaptador requiere una distribución reproducible de Sovaharmony antes de ejecutar."
        )

    def extract_features(self, data: Any, config: dict[str, Any]) -> Any:
        return data

    def build_training_data(self, data: Any, config: dict[str, Any]) -> Any:
        return data
