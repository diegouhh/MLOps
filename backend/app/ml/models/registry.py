from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from app.core.errors import ResourceNotFoundError, ValidationError
from app.domain.plugins import ModelMetadata, ModelPlugin


def _coerce(value: Any, spec: dict[str, Any], name: str) -> Any:
    expected = spec.get("type")
    if value is None and isinstance(expected, list) and "null" in expected:
        return None
    if isinstance(expected, list):
        expected = next((item for item in expected if item != "null"), None)
    if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValidationError(f"{name} debe ser entero")
    if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ValidationError(f"{name} debe ser numérico")
    if expected == "boolean" and not isinstance(value, bool):
        raise ValidationError(f"{name} debe ser booleano")
    if "enum" in spec and value not in spec["enum"]:
        raise ValidationError(f"{name} debe ser uno de {spec['enum']}")
    if "minimum" in spec and value < spec["minimum"]:
        raise ValidationError(f"{name} debe ser >= {spec['minimum']}")
    if "maximum" in spec and value > spec["maximum"]:
        raise ValidationError(f"{name} debe ser <= {spec['maximum']}")
    return value


class SklearnModelPlugin(ModelPlugin):
    estimator_class: type

    def validate_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        unknown = set(parameters) - set(self.metadata.editable_parameters)
        if unknown:
            raise ValidationError(
                f"Parámetros no permitidos para {self.metadata.id}: {sorted(unknown)}"
            )
        merged = {**self.metadata.default_parameters, **parameters}
        properties = self.metadata.parameter_schema.get("properties", {})
        return {name: _coerce(value, properties[name], name) for name, value in merged.items()}

    def build(self, parameters: dict[str, Any], random_state: int) -> Any:
        validated = self.validate_parameters(parameters)
        if "random_state" in self.estimator_class().get_params():
            validated["random_state"] = random_state
        return self.estimator_class(**validated)


class LogisticRegressionPlugin(SklearnModelPlugin):
    estimator_class = LogisticRegression
    metadata = ModelMetadata(
        id="logistic_regression",
        name="logistic_regression",
        display_name="Regresión logística",
        description="Clasificador lineal interpretable y reproducible.",
        task_types=["classification"],
        family="linear",
        interpretability="high",
        compute_cost="low",
        strengths=[
            "Baseline rápido y reproducible",
            "Coeficientes fáciles de inspeccionar",
            "Probabilidades disponibles",
        ],
        limitations=[
            "La frontera de decisión es lineal",
            "Puede requerir escalado de variables",
            "Puede ser sensible a variables muy correlacionadas",
        ],
        default_parameters={"C": 1.0, "max_iter": 1000, "class_weight": None},
        editable_parameters=["C", "max_iter", "class_weight"],
        parameter_schema={
            "type": "object",
            "properties": {
                "C": {"type": "number", "minimum": 0.0001, "maximum": 10000},
                "max_iter": {"type": "integer", "minimum": 100, "maximum": 10000},
                "class_weight": {"enum": [None, "balanced"]},
            },
        },
        supports_probability=True,
    )


class RandomForestPlugin(SklearnModelPlugin):
    estimator_class = RandomForestClassifier
    metadata = ModelMetadata(
        id="random_forest",
        name="random_forest",
        display_name="Random Forest",
        description="Ensamble robusto de árboles de decisión.",
        task_types=["classification"],
        family="ensemble",
        interpretability="medium",
        compute_cost="medium",
        strengths=[
            "Captura relaciones no lineales",
            "No depende fuertemente del escalado",
            "Permite inspeccionar importancia de variables",
        ],
        limitations=[
            "Es menos interpretable que un modelo lineal",
            "Muchos árboles aumentan memoria y costo",
            "Las probabilidades pueden requerir calibración",
        ],
        default_parameters={
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "class_weight": None,
        },
        editable_parameters=["n_estimators", "max_depth", "min_samples_split", "class_weight"],
        parameter_schema={
            "type": "object",
            "properties": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 2000},
                "max_depth": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
                "min_samples_split": {"type": "integer", "minimum": 2, "maximum": 100},
                "class_weight": {"enum": [None, "balanced", "balanced_subsample"]},
            },
        },
        supports_probability=True,
    )


class SvmPlugin(SklearnModelPlugin):
    estimator_class = SVC
    metadata = ModelMetadata(
        id="svm",
        name="svm",
        display_name="Support Vector Machine",
        description="Clasificador de margen máximo con kernels controlados.",
        task_types=["classification"],
        family="kernel",
        interpretability="medium",
        compute_cost="medium",
        strengths=[
            "Funciona bien en espacios de muchas variables",
            "Los kernels permiten fronteras no lineales",
            "Ofrece control explícito de regularización",
        ],
        limitations=[
            "Es sensible al escalado de variables",
            "Puede ser costoso con datasets grandes",
            "Calcular probabilidades añade trabajo adicional",
        ],
        default_parameters={"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True},
        editable_parameters=["C", "kernel", "gamma", "probability"],
        parameter_schema={
            "type": "object",
            "properties": {
                "C": {"type": "number", "minimum": 0.0001, "maximum": 10000},
                "kernel": {"enum": ["linear", "rbf", "poly", "sigmoid"]},
                "gamma": {"enum": ["scale", "auto"]},
                "probability": {"type": "boolean"},
            },
        },
        supports_probability=True,
    )


class KnnPlugin(SklearnModelPlugin):
    estimator_class = KNeighborsClassifier
    metadata = ModelMetadata(
        id="knn",
        name="knn",
        display_name="K-Nearest Neighbors",
        description="Clasificación por vecinos cercanos.",
        task_types=["classification"],
        family="neighbors",
        interpretability="medium",
        compute_cost="medium",
        strengths=[
            "Concepto simple de entender",
            "No impone una frontera paramétrica",
            "Es útil como baseline basado en cercanía",
        ],
        limitations=[
            "La inferencia crece con el tamaño del dataset",
            "Es sensible al escalado y a la métrica de distancia",
            "Puede degradarse con alta dimensionalidad",
        ],
        default_parameters={"n_neighbors": 5, "weights": "uniform", "p": 2},
        editable_parameters=["n_neighbors", "weights", "p"],
        parameter_schema={
            "type": "object",
            "properties": {
                "n_neighbors": {"type": "integer", "minimum": 1, "maximum": 100},
                "weights": {"enum": ["uniform", "distance"]},
                "p": {"type": "integer", "minimum": 1, "maximum": 2},
            },
        },
        supports_probability=True,
    )


class GradientBoostingPlugin(SklearnModelPlugin):
    estimator_class = GradientBoostingClassifier
    metadata = ModelMetadata(
        id="gradient_boosting",
        name="gradient_boosting",
        display_name="Gradient Boosting",
        description="Ensamble secuencial de árboles ligeros.",
        task_types=["classification"],
        family="ensemble",
        interpretability="medium",
        compute_cost="medium",
        strengths=[
            "Captura no linealidades e interacciones",
            "Suele ser competitivo en datos tabulares",
            "Ofrece probabilidades de clasificación",
        ],
        limitations=[
            "Es sensible a varios hiperparámetros",
            "El entrenamiento es secuencial",
            "Es menos interpretable que un modelo lineal",
        ],
        default_parameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        editable_parameters=["n_estimators", "learning_rate", "max_depth"],
        parameter_schema={
            "type": "object",
            "properties": {
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 1000},
                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        supports_probability=True,
    )


class ModelRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ModelPlugin] = {}

    def register(self, plugin: ModelPlugin) -> None:
        if plugin.metadata.id in self._plugins:
            raise ValueError(f"Modelo duplicado: {plugin.metadata.id}")
        self._plugins[plugin.metadata.id] = plugin

    def get(self, model_id: str) -> ModelPlugin:
        try:
            return self._plugins[model_id]
        except KeyError as exc:
            raise ResourceNotFoundError(f"Modelo no encontrado: {model_id}") from exc

    def list(self) -> list[ModelPlugin]:
        return list(self._plugins.values())


model_registry = ModelRegistry()
for builtin in (
    LogisticRegressionPlugin(),
    RandomForestPlugin(),
    SvmPlugin(),
    KnnPlugin(),
    GradientBoostingPlugin(),
):
    model_registry.register(builtin)
