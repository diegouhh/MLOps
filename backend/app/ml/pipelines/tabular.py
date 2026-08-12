from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.errors import ValidationError
from app.domain.plugins import PipelineMetadata, PipelinePlugin


class BasicTabularPipeline(PipelinePlugin):
    metadata = PipelineMetadata(
        id="tabular_basic",
        name="tabular_basic",
        display_name="Tabular básico",
        description="CSV con imputación, escalado y codificación sin fuga de información.",
        version="1.0.0",
        data_type="tabular",
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
                "target_column": {"type": "string", "minLength": 1},
                "group_column": {"type": ["string", "null"]},
                "drop_columns": {"type": "array", "items": {"type": "string"}, "default": []},
                "numeric_imputation": {"enum": ["mean", "median"], "default": "median"},
                "categorical_imputation": {"enum": ["most_frequent"], "default": "most_frequent"},
                "scale_numeric": {"type": "boolean", "default": True},
            },
        },
        steps=["validar", "cargar", "separar objetivo", "imputar", "codificar", "escalar"],
    )

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "target_column",
            "group_column",
            "drop_columns",
            "numeric_imputation",
            "categorical_imputation",
            "scale_numeric",
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValidationError(f"Parámetros de pipeline no permitidos: {sorted(unknown)}")
        target = str(config.get("target_column", "")).strip()
        if not target:
            raise ValidationError("target_column es obligatorio")
        normalized = {
            "target_column": target,
            "group_column": config.get("group_column") or None,
            "drop_columns": list(config.get("drop_columns", [])),
            "numeric_imputation": config.get("numeric_imputation", "median"),
            "categorical_imputation": config.get("categorical_imputation", "most_frequent"),
            "scale_numeric": bool(config.get("scale_numeric", True)),
        }
        if normalized["numeric_imputation"] not in {"mean", "median"}:
            raise ValidationError("numeric_imputation debe ser mean o median")
        return normalized

    def validate_input(self, dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
        config = self.validate_config(config)
        path = Path(dataset_path)
        if path.suffix.lower() != ".csv":
            raise ValidationError("El pipeline tabular requiere un archivo CSV")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValidationError("No fue posible leer el CSV") from exc
        if frame.empty:
            raise ValidationError("El CSV está vacío")
        target = config["target_column"]
        if target not in frame.columns:
            raise ValidationError(f"La columna objetivo '{target}' no existe")
        if frame[target].isna().any():
            raise ValidationError("La variable objetivo contiene valores faltantes")
        if frame[target].nunique() < 2:
            raise ValidationError("La variable objetivo necesita al menos dos clases")
        group = config.get("group_column")
        if group and group not in frame.columns:
            raise ValidationError(f"La columna de grupos '{group}' no existe")
        missing_drop = set(config["drop_columns"]) - set(frame.columns)
        if missing_drop:
            raise ValidationError(f"Columnas a eliminar inexistentes: {sorted(missing_drop)}")
        return {
            "rows": len(frame),
            "columns": len(frame.columns),
            "target": target,
            "classes": [str(value) for value in sorted(frame[target].unique(), key=str)],
            "missing_values": int(frame.isna().sum().sum()),
        }

    def load_data(self, dataset_path: str, config: dict[str, Any]) -> pd.DataFrame:
        self.validate_input(dataset_path, config)
        return pd.read_csv(dataset_path)

    def preprocess(self, data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        return data.copy()

    def extract_features(self, data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        return data

    def build_training_data(self, data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        config = self.validate_config(config)
        target_column = config["target_column"]
        group_column = config.get("group_column")
        excluded = set(config["drop_columns"]) | {target_column}
        if group_column:
            excluded.add(group_column)
        feature_columns = [column for column in data.columns if column not in excluded]
        if not feature_columns:
            raise ValidationError(
                "No quedan variables predictoras después de aplicar la configuración"
            )
        features = data[feature_columns]
        target = data[target_column]
        groups = data[group_column] if group_column else None
        numeric = features.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in feature_columns if column not in numeric]
        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric:
            numeric_steps: list[tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy=config["numeric_imputation"]))
            ]
            if config["scale_numeric"]:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(("numeric", Pipeline(numeric_steps), numeric))
        if categorical:
            categorical_steps = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("categorical", categorical_steps, categorical))
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        return {
            "X": features,
            "y": target,
            "groups": groups,
            "preprocessor": preprocessor,
            "feature_names": feature_columns,
            "numeric_features": numeric,
            "categorical_features": categorical,
        }

    def produce_artifacts(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "feature_names": context.get("feature_names", []),
            "numeric_features": context.get("numeric_features", []),
            "categorical_features": context.get("categorical_features", []),
        }
