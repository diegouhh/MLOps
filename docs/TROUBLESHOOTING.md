# Solución de problemas

## Docker no existe

Compruebe:

```bash
docker version
docker compose version
```

En Windows o macOS, inicie Docker Desktop. En Linux, instale Docker Engine y el plugin Compose.

## Un puerto está ocupado

Cambie `.env`:

```env
FRONTEND_PORT=3001
API_PORT=8001
MLFLOW_PORT=5001
PREFECT_PORT=4201
NEUROOPS_CORS_ORIGINS=http://localhost:3001
VITE_API_URL=http://localhost:8001/api/v1
VITE_MLFLOW_URL=http://localhost:5001
VITE_PREFECT_URL=http://localhost:4201
PREFECT_UI_API_URL=http://localhost:4201/api
```

Reconstruya el frontend porque las URLs públicas se insertan durante el build:

```bash
docker compose up --build
```

## La API espera a MLflow o Prefect

Revise health checks:

```bash
docker compose ps
docker compose logs mlflow
docker compose logs prefect
docker compose logs prefect-bootstrap
docker compose logs prefect-worker
docker compose logs api
```

Los contenedores se comunican mediante nombres de servicio (`mlflow`, `prefect`), no `localhost`.

En Prefect deben aparecer el work pool `neuroops-process`, el worker `neuroops-worker` y los deployments `neuroops-experiments` y `neuroops-predictions`. Si cambió el código de los flows, reconstruya para volver a registrar los deployments:

```bash
docker compose up -d --build prefect-bootstrap prefect-worker api frontend
```

## El pipeline EEG aparece no disponible

MNE y MNE-BIDS están incluidos en la imagen predeterminada. Reconstruya API y worker para aplicar el lock actual:

```bash
docker compose build --no-cache api prefect-worker
docker compose up -d
```

Local:

```bash
cd backend
uv sync --extra eeg
```

## Sovaharmony aparece no disponible

Es el comportamiento esperado. NeuroOps comprueba toda la cadena opcional y explica qué paquetes faltan. No copie el wheel histórico `dirty` al core. El adaptador se habilitará cuando la cadena pueda instalarse de fuentes públicas y fijadas.

## El CSV se rechaza

Verifique:

- extensión `.csv`;
- encabezados en la primera fila;
- al menos una fila;
- variable objetivo existente, sin nulos y con dos o más clases;
- codificación de texto legible por pandas;
- tamaño menor que `NEUROOPS_MAX_UPLOAD_MB`.

## Group K-Fold falla

Configure `group_column` en el pipeline y asegúrese de tener al menos tantos grupos únicos como `n_splits`. En EEG, los grupos se derivan del sujeto.

## El ZIP/BIDS se rechaza

El ZIP debe contener `dataset_description.json`, `participants.tsv` y archivos EEG compatibles. Se rechazan rutas `../`, rutas absolutas, enlaces simbólicos y archivos cuya expansión exceda el límite.

## La carpeta BIDS local no aparece

La raíz debe estar directamente bajo `datasets/`, por ejemplo `datasets/ds004504/dataset_description.json`. Después pulse **Datasets → Carpeta local BIDS → Actualizar carpetas**. Compruebe que `participants.tsv` no esté vacío y que exista al menos un archivo `*_eeg.edf`, `.bdf`, `.vhdr`, `.set` o `.fif`.

Si añadió la carpeta fuera del directorio del proyecto, muévala o cambie de forma explícita el bind mount `./datasets:/datasets:ro` en API y worker. No use una ruta de Windows dentro del formulario; la interfaz solo acepta carpetas detectadas bajo el montaje.

## Un experimento queda fallido

Abra el detalle y revise el error del run. Luego consulte MLflow y Prefect. El fallo queda persistido y no bloquea nuevos experimentos. Corrija configuración o datos y use **Reejecutar**.

## MLflow no puede cargar un modelo antiguo

Esta versión usa MLflow 3 y logged models (`models:/m-...`). Los modelos creados por la versión nueva se registran automáticamente. No mezcle la base MLflow heredada con los volúmenes nuevos sin una migración específica.

## Restablecer el entorno local

`docker compose down` conserva datos. Eliminar volúmenes borra de forma irreversible datasets, bases y modelos:

```bash
docker compose down --volumes
```

Haga una copia de seguridad antes de usar ese comando.

## Ejecutar diagnósticos

```bash
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/system/dependencies
```

El segundo endpoint muestra paquetes opcionales y estado de servicios sin exponer secretos.
