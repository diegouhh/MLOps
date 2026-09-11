# Trazabilidad de NeuroOps

NeuroOps usa los identificadores de su base de datos como identidad canónica. Prefect y MLflow conservan sus propios identificadores internos, pero reciben metadatos que permiten volver al recurso de NeuroOps.

## Experimentos

El nombre visible del experimento es el nombre definido por el usuario en NeuroOps.

- NeuroOps: `Experiment.id` es el identificador canónico.
- Prefect: el flow run recibe el mismo nombre visible del experimento y una etiqueta `neuroops-experiment-id:<uuid>`.
- MLflow: el parent run usa el nombre del experimento y el tag `neuroops.experiment_id`.
- Training runs: conservan su UUID de NeuroOps y su `mlflow_run_id`.

Esto permite localizar el mismo experimento por nombre para lectura humana y por UUID para trazabilidad inequívoca.

## Predicciones

Una predicción conserva dos conceptos distintos:

- `version_or_alias`: lo que pidió el usuario, por ejemplo `champion`.
- `resolved_model_version`: la versión numérica concreta que se resolvió al crear el trabajo.

La ejecución siempre carga `resolved_model_version`. Un cambio posterior del alias `champion` no puede cambiar el modelo usado por una predicción ya creada.

El nombre visible de la ejecución de predicción es determinista:

`Predicción <modelo> v<versión> [<id-corto>]`

Prefect y MLflow usan ese mismo nombre.

## Validación con grupos

Si un pipeline produce `groups`, NeuroOps rechaza `train_test_split` y `stratified_kfold`. Debe utilizarse `group_kfold` o `stratified_group_kfold` para impedir que el mismo sujeto o grupo aparezca en entrenamiento y validación.

En EEG, la interfaz selecciona por defecto `stratified_group_kfold` y deshabilita las estrategias que ignoran el sujeto.

## Copias Docker independientes

El Compose conserva `neuroops` como nombre por defecto para no desconectar los volúmenes existentes. Para ejecutar otra copia de forma independiente, usa otro valor:

```bash
COMPOSE_PROJECT_NAME=neuroops-dev2 docker compose up -d
```

Cada copia tendrá nombres de contenedores y volúmenes separados por el proyecto de Compose.
