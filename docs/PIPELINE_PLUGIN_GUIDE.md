# Guía de plugins de pipeline

## Contrato

Todo pipeline implementa `PipelinePlugin` y declara `PipelineMetadata`.

```python
class ExamplePipeline(PipelinePlugin):
    metadata = PipelineMetadata(
        id="example_v1",
        name="example_v1",
        display_name="Ejemplo",
        description="Descripción para investigadores.",
        version="1.0.0",
        data_type="tabular",
        available=True,
        required_packages=[],
        supported_models=["logistic_regression"],
        config_schema={"type": "object", "properties": {}},
        steps=["validar", "cargar", "preparar"],
    )
```

Métodos requeridos:

```python
def validate_config(self, config): ...
def validate_input(self, dataset_path, config): ...
def load_data(self, dataset_path, config): ...
def preprocess(self, data, config): ...
def extract_features(self, data, config): ...
def build_training_data(self, data, config): ...
def produce_artifacts(self, context): ...
```

## Salida de entrenamiento

`build_training_data` debe retornar:

```python
{
    "X": features,
    "y": target,
    "groups": groups_or_none,
    "preprocessor": sklearn_transformer_or_passthrough,
    "feature_names": ["feature_a", "feature_b"],
    "numeric_features": ["feature_a"],
    "categorical_features": ["feature_b"],
}
```

No ajuste el preprocesador sobre todo el dataset. NeuroOps lo inserta en un `sklearn.Pipeline` y lo ajusta dentro de cada partición. Esta regla es esencial para evitar fuga de información.

## Registro

Añada el plugin a `register_builtin_pipelines` en `backend/app/ml/pipelines/registry.py`:

```python
pipeline_registry.register(ExamplePipeline())
```

El ID debe ser estable y único. Un cambio incompatible requiere nueva versión y, si cambia el contrato de entrada, normalmente un ID nuevo.

## Dependencias reproducibles

La detección debe ser ligera y no importar el paquete pesado:

```python
available = importlib.util.find_spec("optional_package") is not None
```

Declare `available=False` y `unavailable_reason` para una integración que todavía no pueda incluirse. Importe la dependencia solo dentro del método que la usa para que el catálogo siga disponible.

No instale paquetes desde la interfaz ni dentro de un flow. Para habilitar un pipeline:

1. agregue versiones fijas a un extra de `backend/pyproject.toml`;
2. ejecute `uv lock` y confirme `backend/uv.lock`;
3. incluya el extra en el target de `backend/Dockerfile` usado por API y worker;
4. reconstruya ambos servicios;
5. verifique el paquete en `/api/v1/system/dependencies`.

El pipeline debe vivir en el repositorio y registrarse explícitamente. Así la API y el worker ejecutan exactamente el mismo código y las mismas dependencias.

## Configuración segura

- Mantenga una lista explícita de claves.
- Valide tipos, rangos y combinaciones.
- No acepte callables, módulos, rutas ejecutables ni fragmentos Python.
- Normalice valores predeterminados en `validate_config`.
- Devuelva mensajes comprensibles para la UI.

## Pipeline EEG

Para datos EEG:

- valide BIDS antes de cargar;
- preserve el sujeto en `groups`;
- recomiende Group K-Fold o Stratified Group K-Fold;
- registre filtros, canales, épocas y bandas;
- use fixtures sintéticos o públicos en pruebas;
- no incluya datos clínicos sensibles en el repositorio.

## Pruebas mínimas

1. El registro descubre el plugin.
2. Configuración válida se normaliza.
3. Claves desconocidas se rechazan.
4. Entrada inválida produce `ValidationError`.
5. La salida contiene todas las claves esperadas.
6. El preprocesamiento funciona dentro de una partición.
7. API y worker descubren la misma versión del pipeline en la imagen construida.
