# Demostración reproducible

## Objetivo

Ejecutar el flujo vertical completo con `data/demo/iris.csv`, un dataset público, pequeño y no sensible.

## Con la interfaz

Arranque:

```bash
cp .env.example .env
docker compose up --build
```

### 1. Dataset

En `http://localhost:3000` abra **Datasets**:

- Nombre: `Iris demo`.
- Tipo: `Datos tabulares`.
- Archivo: `data/demo/iris.csv`.

La respuesta debe mostrar 30 filas, 5 columnas, versión 1 y estado listo.

### 2. Experimento

Abra **Nuevo experimento** y seleccione:

| Campo | Valor |
| --- | --- |
| Dataset | Iris demo |
| Pipeline | Tabular básico |
| Variable objetivo | `species` |
| Modelos | Regresión logística, Random Forest, SVM |
| Validación | Train/test estratificado |
| Test size | `0.30` |
| Métrica principal | F1 macro |
| Semilla | `42` |

Ejecute. La API responde con un ID y la vista de detalle empieza a consultar el estado.

### 3. Resultado

Verifique:

- un run padre en MLflow;
- tres runs hijos;
- métricas independientes;
- configuración y entorno como JSON;
- matriz de confusión y curva ROC;
- un candidato ganador;
- un modelo registrado `neuroops-<experiment-id>`;
- alias `champion`.

Los botones **View in MLflow** y **View in Prefect** deben abrir directamente el run y el flow-run de este experimento, no las páginas iniciales de los servicios.

### 4. Predicción

En **Predicciones** seleccione el modelo registrado y alias `champion`. NeuroOps genera cuatro campos numéricos. Cree dos registros y use estos valores:

| Registro | sepal_length | sepal_width | petal_length | petal_width |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5.1 | 3.5 | 1.4 | 0.2 |
| 2 | 6.7 | 3.0 | 5.2 | 2.3 |

El mismo lote también puede pegarse en la pestaña opcional **JSON avanzado**:

```json
[
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3
  }
]
```

La respuesta contiene dos predicciones y, cuando el modelo lo permite, probabilidades y clases.

## Prueba automatizada

```bash
cd backend
uv sync --extra dev --extra eeg
uv run pytest -q tests/test_e2e.py
```

Esta prueba no simula el resultado: entrena tres modelos, escribe en MLflow, crea la versión registrada, asigna el alias, carga el modelo desde el Registry y predice.
