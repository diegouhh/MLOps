# Guía de plugins de modelo

## Contrato

Un modelo implementa `ModelPlugin`:

```python
class ExampleModel(ModelPlugin):
    metadata = ModelMetadata(
        id="example_classifier",
        name="example_classifier",
        display_name="Clasificador de ejemplo",
        description="Explicación breve.",
        task_types=["classification"],
        default_parameters={"alpha": 1.0},
        editable_parameters=["alpha"],
        parameter_schema={
            "type": "object",
            "properties": {
                "alpha": {"type": "number", "minimum": 0.001, "maximum": 100}
            },
        },
        supports_probability=True,
    )

    def validate_parameters(self, parameters): ...
    def build(self, parameters, random_state): ...
```

## Reglas

- El ID es la única selección aceptada desde la UI.
- `editable_parameters` es una lista blanca.
- La validación ocurre antes de almacenar el experimento.
- `random_state` se inyecta desde la semilla del experimento cuando el estimador lo admite.
- La factory debe retornar un estimador compatible con scikit-learn.
- No se aceptan import paths, clases ni código enviados por el usuario.

## Registro

Registre una instancia en `backend/app/ml/models/registry.py`:

```python
model_registry.register(ExampleModel())
```

Añada el ID a `supported_models` de los pipelines compatibles.

## Dependencias opcionales

XGBoost, PyTorch u otros backends deben vivir en extras separados. El plugin debe comunicar disponibilidad y razón, igual que los pipelines opcionales. No añada Torch al grupo core.

## Modelos serializados

MLflow 3 crea un logged model por candidato. NeuroOps usa Cloudpickle solo para estimadores construidos por el registro interno y guarda requisitos exactos del entorno. Nunca cargue bytes de modelo subidos por usuarios sin una política de confianza separada.

## Pruebas mínimas

1. El modelo aparece en el catálogo.
2. Los parámetros predeterminados son válidos.
3. Claves desconocidas se rechazan.
4. Tipos y rangos se validan.
5. La semilla produce resultados repetibles.
6. `fit` y `predict` funcionan dentro del pipeline tabular.
7. `predict_proba` coincide con `supports_probability`.

