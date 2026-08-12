# NeuroOps

NeuroOps es una plataforma MLOps web, modular y reproducible para investigadores que necesitan registrar datasets, seleccionar pipelines, comparar varios modelos de clasificación, seguir ejecuciones, registrar el ganador y reutilizarlo para predicciones.

Esta versión es una evolución del proyecto NeuroOps/MLOps original de Emmanuel Arizabaleta y del trabajo del Grupo de Neurociencias de Antioquia. Conserva la licencia MIT, la atribución y el objetivo científico de procesar y clasificar señales EEG, pero desacopla el núcleo de dependencias históricas no reproducibles.

## Qué funciona

- Carga segura de CSV y ZIP/BIDS, y registro de carpetas BIDS locales grandes sin copiarlas.
- Catálogo real de pipelines y modelos servido por la API.
- Pipeline tabular sin fuga de información: imputación, escalado y one-hot encoding dentro de `scikit-learn.Pipeline`.
- Pipeline EEG público con MNE/MNE-BIDS incluido en la imagen predeterminada.
- Adaptador Sovaharmony aislado y visible como no disponible cuando faltan dependencias.
- Selección de varios modelos en una misma ejecución.
- Regresión logística, Random Forest, SVM, KNN y Gradient Boosting.
- Train/test estratificado, Stratified K-Fold, Group K-Fold y Stratified Group K-Fold.
- Accuracy, balanced accuracy, precision macro, recall macro, F1 macro, F1 weighted y ROC AUC cuando aplica.
- Matriz de confusión, reporte de clasificación, curva ROC, configuración, resumen del dataset, variables y entorno como artefactos.
- MLflow 3 con run padre, runs hijos, logged models, Model Registry y aliases `champion`/`challenger`.
- Prefect 3 como cola y orquestador: deployments para entrenamiento y predicción.
- Tareas visibles para validar, cargar, preprocesar, extraer características, preparar validación, entrenar cada candidato y registrar el ganador.
- Worker Prefect independiente; reiniciar FastAPI no interrumpe los trabajos activos.
- API FastAPI versionada que responde `202` después de enviar cada trabajo a Prefect.
- Interfaz React/Ant Design basada en la experiencia visual original, conectada a la API real.
- Model Management con champion, versiones, aliases, métricas e historial de entrenamientos.
- Enlaces profundos **View in MLflow** y **View in Prefect** para cada ejecución.
- Formularios de pipeline, parámetros y predicción generados desde los esquemas de la API; JSON queda como modo avanzado.
- SQLite local y PostgreSQL en el perfil de producción.
- Docker Compose con imágenes propias y health checks.
- Pruebas unitarias, de integración y E2E con un dataset público pequeño.

## Inicio rápido con Docker

Requisitos:

- Docker Desktop 4.x o Docker Engine 25+.
- Docker Compose v2.
- Al menos 6 GB de RAM libres para construir y ejecutar los servicios.

Desde la raíz del proyecto:

```bash
cp .env.example .env
docker compose up --build
```

En PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Servicios:

| Servicio | URL | Uso |
| --- | --- | --- |
| NeuroOps | http://localhost:3000 | Interfaz web |
| FastAPI | http://localhost:8000/docs | API y OpenAPI |
| MLflow | http://localhost:5000 | Runs, artefactos y registro |
| Prefect | http://localhost:4200 | Flujos, tareas y logs |

`prefect-bootstrap` registra los deployments `neuroops-experiments` y `neuroops-predictions`; `prefect-worker` consume la cola `neuroops-process`. El contenedor de bootstrap termina después de configurar Prefect; que no permanezca en `docker compose ps` es normal.

Detener:

```bash
docker compose down
```

Eliminar también los volúmenes locales requiere una decisión explícita, porque borra datasets, experimentos y modelos:

```bash
docker compose down --volumes
```

## Primer experimento

1. Abre `http://localhost:3000`.
2. En **Datasets**, registra `data/demo/iris.csv` como `Datos tabulares`.
3. Abre **Nuevo experimento**.
4. Selecciona el dataset y el pipeline **Tabular básico**.
5. Usa `species` como variable objetivo.
6. Selecciona, por ejemplo, Regresión logística, Random Forest y SVM.
7. Elige `Train/test estratificado`, `F1 macro` y semilla `42`.
8. Ejecuta y abre el detalle para seguir etapas y métricas; usa **View in MLflow** y **View in Prefect** para abrir la ejecución exacta.
9. El ganador se registra automáticamente con alias `champion`.
10. En **Model management**, cambia el nombre registrado o promueve otro candidato si lo necesitas.
11. En **Predicciones**, selecciona `champion` y completa las cuatro variables en el formulario generado.

Consulta [docs/DEMO.md](docs/DEMO.md) para el ejemplo completo.

## Dataset EEG grande en una carpeta local

No comprimas ni subas una colección BIDS grande desde el navegador:

1. Descarga o extrae la raíz BIDS dentro de `datasets/`, por ejemplo `datasets/ds004504/`.
2. Comprueba que esa carpeta contenga `dataset_description.json`, `participants.tsv` y archivos `*_eeg.*`.
3. Inicia o reconstruye NeuroOps con `docker compose up -d --build`.
4. Abre **Datasets → Carpeta local BIDS**, pulsa **Actualizar carpetas** y registra la carpeta detectada.

La API y el worker ven `datasets/` en `/datasets` con montaje de solo lectura. NeuroOps guarda únicamente metadatos, huella y ruta; eliminar el registro no elimina los archivos locales.

## Perfiles de Docker

El comando predeterminado incluye el núcleo tabular, MNE y MNE-BIDS. No instala Torch ni la cadena histórica de Sovaharmony. Las dependencias se fijan en `pyproject.toml` y `uv.lock`; la aplicación no instala paquetes durante la ejecución.

### Core

```bash
docker compose up --build
```

Incluye FastAPI, React, MLflow, Prefect, dependencias tabulares, MNE y MNE-BIDS.

### Deep learning preparado

```bash
docker compose -f docker-compose.yml -f docker-compose.deep-learning.yml up --build
```

Instala PyTorch de forma aislada. La versión actual no registra todavía un modelo deep learning en el catálogo; el perfil prepara esa extensión sin aumentar la imagen core.

### Producción con PostgreSQL

Define una contraseña fuerte en `.env`:

```env
POSTGRES_PASSWORD=una-clave-larga-y-segura
NEUROOPS_CORS_ORIGINS=https://neuroops.example.org
VITE_API_URL=https://api.neuroops.example.org/api/v1
VITE_MLFLOW_URL=https://mlflow.neuroops.example.org
VITE_PREFECT_URL=https://prefect.neuroops.example.org
PREFECT_UI_API_URL=https://prefect.neuroops.example.org/api
MLFLOW_SERVER_ALLOWED_HOSTS=mlflow,mlflow:5000,mlflow.neuroops.example.org
```

Después:

```bash
docker compose -f docker-compose.yml -f docker-compose.production.yml up --build
```

El perfil crea bases separadas para la aplicación, MLflow y Prefect, evitando colisiones entre migraciones.

## Ejecución local sin Docker

Requisitos: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/) y Node.js 24.

Backend:

```bash
cd backend
cp .env.example .env
uv sync --extra dev --extra eeg
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 8000
```

La configuración local usa SQLite y almacenamiento de MLflow integrado. Para ejecutar la misma arquitectura sin Docker, inicia MLflow y Prefect en procesos separados:

```bash
cd backend
uv run mlflow server --backend-store-uri sqlite:///./mlflow.db --host 0.0.0.0 --port 5000
```

```bash
cd backend
uv run prefect server start --host 0.0.0.0 --port 4200
```

En una tercera terminal registra los deployments y arranca el worker:

```bash
cd backend
uv run prefect work-pool create neuroops-process --type process --overwrite
uv run prefect deploy --all
uv run prefect worker start --pool neuroops-process --type process --limit 1 --with-healthcheck
```

Configura en `backend/.env`:

```env
NEUROOPS_MLFLOW_TRACKING_URI=http://localhost:5000
NEUROOPS_MLFLOW_REGISTRY_URI=http://localhost:5000
NEUROOPS_PREFECT_API_URL=http://localhost:4200/api
```

Frontend:

```bash
cd frontend
npm ci
npm run dev
```

## Pruebas y calidad

```bash
cd backend
uv sync --extra dev --extra eeg
uv run ruff check app tests alembic
uv run pytest
```

```bash
cd frontend
npm ci
npm run lint
npm run build
```

La prueba E2E ejecuta:

```text
Iris CSV → pipeline tabular → 3 modelos → métricas → ganador
→ MLflow Registry → alias champion → predicción → respuesta verificada
```

## Catálogo inicial

### Pipelines

| ID | Tipo | Disponibilidad core |
| --- | --- | --- |
| `tabular_basic` | CSV | Sí |
| `eeg_mne_basic` | EEG BIDS | Sí; incluido en Docker |
| `sovaharmony_legacy` | EEG BIDS | Solo si toda la cadena histórica está instalada |

### Modelos

| ID | Modelo | Probabilidades |
| --- | --- | --- |
| `logistic_regression` | Regresión logística | Sí |
| `random_forest` | Random Forest | Sí |
| `svm` | Support Vector Machine | Sí por configuración controlada |
| `knn` | K-Nearest Neighbors | Sí |
| `gradient_boosting` | Gradient Boosting | Sí |

La API solo acepta IDs y parámetros incluidos en los registros. Nunca evalúa código Python enviado por usuarios.

## Estructura

```text
backend/
  app/api/v1/          API FastAPI
  app/core/            configuración y errores
  app/db/              entidades SQLAlchemy
  app/domain/          contratos de plugins
  app/ml/              pipelines, modelos y evaluación
  app/orchestration/   flujos Prefect
  prefect.yaml         deployments y work pool
  app/services/        casos de uso y MLflow
  alembic/             migraciones
  tests/               unitarias, integración y E2E
frontend/              aplicación React/Vite
data/demo/             dataset no sensible
datasets/              raíces BIDS locales, excluidas de Git
docs/                  arquitectura y guías
docker/                soporte de perfiles
```

## API principal

La especificación exacta está disponible en `/docs`. Los grupos principales son:

- `/api/v1/datasets`
- `/api/v1/pipelines`
- `/api/v1/models/catalog`
- `/api/v1/experiments`
- `/api/v1/runs`
- `/api/v1/registry/models`
- `/api/v1/predictions`
- `/api/v1/system/dependencies`

Los endpoints de tracking devuelven rutas profundas de MLflow/Prefect y el endpoint de esquema de entrada del Registry permite construir formularios de inferencia sin codificar JSON manualmente.

Las operaciones largas de entrenamiento y predicción responden `202 Accepted` con un ID. Prefect conserva la cola y la interfaz consulta el estado sin mantener bloqueada la petición inicial.

## Seguridad del MVP

- Lista blanca de extensiones por tipo de dataset.
- Límite de tamaño de carga y tamaño total extraído.
- Normalización de nombres.
- Bloqueo de Zip Slip y enlaces simbólicos.
- Directorio aislado por dataset y experimento.
- Parámetros permitidos por esquema.
- Sin ejecución de código subido.
- CORS configurable.
- Secretos fuera del repositorio.
- Telemetría externa de MLflow y Prefect desactivada por defecto.

La autenticación y los roles están previstos como fase posterior. Antes de exponer el servicio a Internet se debe añadir identidad, autorización, TLS, límites por usuario y almacenamiento de objetos externo.

## Extensión

- [Guía de pipelines](docs/PIPELINE_PLUGIN_GUIDE.md)
- [Guía de modelos](docs/MODEL_PLUGIN_GUIDE.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Migración desde el legado](docs/MIGRATION_FROM_LEGACY.md)
- [Solución de problemas](docs/TROUBLESHOOTING.md)

## Limitaciones actuales

- El perfil local usa un Process Worker con límite de un flow simultáneo para proteger SQLite; los candidatos de un experimento sí se ejecutan como tareas Prefect concurrentes.
- La cancelación se solicita al flow-run desplegado y se refleja también en la base de NeuroOps.
- El pipeline EEG implementa un flujo base de potencia espectral; no reemplaza una validación clínica ni todos los métodos avanzados del proyecto original.
- Sovaharmony se muestra como adaptador opcional hasta contar con distribuciones públicas, completas y reproducibles de su cadena.
- No hay autenticación en el MVP.

## Atribución y licencia

NeuroOps parte del repositorio MLOps original de Emmanuel Arizabaleta y del trabajo científico asociado al Grupo de Neurociencias de Antioquia. Este repositorio fue reconstruido como una nueva línea de desarrollo y no comparte el historial Git original; la atribución, la licencia y la documentación de migración se conservan explícitamente.

Distribuido bajo la licencia MIT incluida en [LICENSE](LICENSE).
