from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PipelineMetadata:
    id: str
    name: str
    display_name: str
    description: str
    version: str
    data_type: Literal["tabular", "eeg_bids"]
    available: bool = True
    unavailable_reason: str | None = None
    required_packages: list[str] = field(default_factory=list)
    supported_models: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    task_types: list[str] = field(default_factory=list)
    input_label: str | None = None
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelMetadata:
    id: str
    name: str
    display_name: str
    description: str
    task_types: list[str]
    default_parameters: dict[str, Any]
    editable_parameters: list[str]
    parameter_schema: dict[str, Any]
    supports_probability: bool
    available: bool = True
    unavailable_reason: str | None = None
    required_packages: list[str] = field(default_factory=list)
    family: str = "other"
    interpretability: str = "medium"
    training_cost: str = "medium"
    inference_cost: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelinePlugin(ABC):
    metadata: PipelineMetadata

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def validate_input(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def load_data(self, dataset_path: str, config: dict[str, Any]) -> Any: ...

    @abstractmethod
    def preprocess(self, data: Any, config: dict[str, Any]) -> Any: ...

    @abstractmethod
    def extract_features(self, data: Any, config: dict[str, Any]) -> Any: ...

    @abstractmethod
    def build_training_data(self, data: Any, config: dict[str, Any]) -> Any: ...

    def produce_artifacts(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}


class ModelPlugin(ABC):
    metadata: ModelMetadata

    @abstractmethod
    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def build(self, parameters: dict[str, Any], random_state: int) -> Any: ...

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata.to_dict()
