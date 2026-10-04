# Integración APPLEE

NeuroOps integra APPLEE como un pipeline EEG opcional y modular. La integración no copia el código de APPLEE dentro de NeuroOps: usa el repositorio externo `GNACo/APPLEE` fijado a una revisión concreta y lo ejecuta en un entorno Python aislado.

## Versión integrada

- Repositorio: `https://github.com/GNACo/APPLEE`
- Revisión: `5b554417070aed47fcd240b6923ada68f62db029`
- Identificador en NeuroOps: `applee`
- Nombre visible: `APPLEE`
- Tipo de entrada: `BIDS EEG`

La revisión se registra en los metadatos del pipeline y en la configuración de los experimentos para mantener trazabilidad.

## Aislamiento

La integración está separada en tres piezas:

- `backend/app/ml/pipelines/applee.py`: plugin de NeuroOps y contrato con el resto de la plataforma.
- `backend/applee_runtime/`: adaptador de ejecución y dependencias de APPLEE.
- `docker-compose.applee.yml`: worker Prefect dedicado a APPLEE.

El proceso principal de NeuroOps no importa APPLEE ni sus dependencias. El worker APPLEE dispone de un segundo entorno virtual en `/opt/applee-venv` y el plugin se comunica con él mediante un subprocess controlado. Esto evita que la cadena de dependencias de APPLEE altere el entorno de FastAPI, MLflow, Prefect o los demás pipelines.

Si se inicia NeuroOps solamente con `docker-compose.yml`, APPLEE aparece como no disponible y el resto de la plataforma sigue funcionando. Para habilitarlo:

```powershell
docker compose -f docker-compose.yml -f docker-compose.applee.yml up -d --build
```

Para desactivarlo después de haber usado el overlay, baja primero ese stack y vuelve a iniciar el compose base:

```powershell
docker compose -f docker-compose.yml -f docker-compose.applee.yml down
docker compose up -d
```

## Datos fuente

Los datasets montados continúan en solo lectura. APPLEE necesita escribir derivados BIDS, por lo que el adaptador crea un workspace temporal y replica allí la estructura BIDS mediante enlaces simbólicos o copia de respaldo. Los derivados de APPLEE se escriben exclusivamente en ese workspace.

El dataset original no se hace escribible y no se modifica.

## Configuración por defecto

Los valores iniciales reproducen la configuración de `run_config_params/config_params_plataforma.py` del repositorio APPLEE revisado:

- Canales: `FP1`, `FP2`, `C3`, `C4`, `O1`, `O2`, `P7`, `P8`.
- Filtro: 4–50 Hz.
- Frecuencia de línea: 60 Hz.
- Duración de época para características: 2 s.
- Montaje: `standard_1005`.
- ICA: `infomax`.
- PREP: desactivado por defecto.
- Scorepochs: desactivado en esta integración.
- Normalización: desactivada por defecto.
- Características: coherencia (`cohfreq`), Synchronization Likelihood (`sl`), potencia (`power`), cross-frequency (`crossfreq`) y entropía (`entropy`).

Las bandas usadas por el adaptador son las del preset revisado: Delta 4–6, Theta 6–8.5, Alpha-1 8.5–10.5, Alpha-2 10.5–12.5, Beta1 12.5–18.5, Beta2 18.5–21, Beta3 21–30 y Gamma 30–45 Hz.

## Entrenamiento

APPLEE procesa el BIDS y genera características por sujeto. NeuroOps asocia esas características con `participants.tsv` mediante `participant_id`, toma como objetivo la columna elegida por el usuario y entrega la matriz resultante al mismo motor de modelos utilizado por los demás pipelines.

La validación es por sujeto. Se pueden usar los modelos habilitados en el catálogo de NeuroOps y el seguimiento continúa en MLflow y Prefect.

## Predicciones

Las predicciones APPLEE no solicitan al usuario que escriba manualmente las características. El usuario selecciona un registro EEG del mismo dataset utilizado para entrenar el modelo. El worker APPLEE vuelve a ejecutar el mismo procesamiento y extracción de características, NeuroOps comprueba que las columnas coincidan con las usadas en entrenamiento y después ejecuta la versión concreta del modelo registrado.

A diferencia de `eeg_mne_basic`, APPLEE produce una fila de características agregada por sujeto/registro. Por eso la respuesta de predicción no se etiqueta como una agregación de épocas.

## Dependencias externas

El worker APPLEE instala las revisiones externas referenciadas por el repositorio APPLEE para `sovaflow`, `sovaharmony`, `sovareject`, `sovawica` y `sovachronux`, además de las librerías necesarias para ejecutar el adaptador.

Estas dependencias no forman parte del entorno Python principal de NeuroOps.

## Licencia del proyecto externo

En la revisión consultada, el repositorio público `GNACo/APPLEE` no expone un archivo de licencia. NeuroOps no redistribuye su código: el worker instala el repositorio externo durante la construcción de su imagen. Antes de redistribuir, modificar o incorporar código de APPLEE directamente en NeuroOps, se debe confirmar con el grupo la licencia y los permisos aplicables.
