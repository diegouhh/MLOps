from __future__ import annotations

import importlib

from app.core.errors import ResourceNotFoundError
from app.domain.plugins import PipelinePlugin


class PipelineRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PipelinePlugin] = {}

    def register(self, plugin: PipelinePlugin) -> None:
        plugin_id = plugin.metadata.id
        if plugin_id in self._plugins:
            raise ValueError(f"Pipeline duplicado: {plugin_id}")
        self._plugins[plugin_id] = plugin

    def get(self, plugin_id: str) -> PipelinePlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise ResourceNotFoundError(f"Pipeline no encontrado: {plugin_id}") from exc

    def list(self) -> list[PipelinePlugin]:
        return list(self._plugins.values())

    def refresh(self, plugin_id: str) -> PipelinePlugin:
        factories = {
            "tabular_basic": lambda: _tabular_pipeline(),
            "eeg_mne_basic": lambda: _eeg_pipeline(),
            "sovaharmony_legacy": lambda: _sovaharmony_pipeline(),
        }
        if plugin_id not in factories:
            raise ResourceNotFoundError(f"Pipeline no encontrado: {plugin_id}")
        importlib.invalidate_caches()
        plugin = factories[plugin_id]()
        self._plugins[plugin_id] = plugin
        return plugin


pipeline_registry = PipelineRegistry()


def _tabular_pipeline() -> PipelinePlugin:
    from app.ml.pipelines.tabular import BasicTabularPipeline

    return BasicTabularPipeline()


def _eeg_pipeline() -> PipelinePlugin:
    from app.ml.pipelines.eeg_mne import BasicEegMnePipeline

    return BasicEegMnePipeline()


def _sovaharmony_pipeline() -> PipelinePlugin:
    from app.ml.pipelines.sovaharmony import SovaharmonyPipeline

    return SovaharmonyPipeline()


def register_builtin_pipelines() -> None:
    if pipeline_registry.list():
        return
    pipeline_registry.register(_tabular_pipeline())
    pipeline_registry.register(_eeg_pipeline())
    pipeline_registry.register(_sovaharmony_pipeline())


register_builtin_pipelines()
