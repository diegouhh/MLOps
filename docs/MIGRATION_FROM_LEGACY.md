# Migración desde NeuroOps heredado

## Propósito

El repositorio original demostró un flujo MLOps valioso alrededor de EEG, FastAPI, MLflow, Prefect y una interfaz web. Esta migración conserva esa visión científica y su atribución, y reorganiza la ejecución para que un clon limpio pueda arrancar sin archivos locales del desarrollador original.

## Diagnóstico técnico

### Reproducibilidad

La ejecución histórica dependía de un estado local de Sovaharmony asociado aproximadamente con el commit `92ce965` y marcado `dirty`. El wheel disponible se llamaba:

```text
sovaharmony-0+untagged.460.g92ce965.dirty-py3-none-any.whl
```

Ese nombre indica que el artefacto fue construido con cambios sin registrar. El wheel tampoco incluía de forma verificable todos los módulos que el código esperaba.

### Dependencias privadas o frágiles

La cadena incluía Sovaflow, Sovachronux, Sovareject y Sovawica. Algunos orígenes históricos estaban en GitFront o podían no estar disponibles públicamente. Sovaflow se usaba sin quedar declarado correctamente en los metadatos del wheel.

Estas dependencias no son adecuadas como requisito de arranque del perfil principal. Su valor científico no se discute; el problema es de distribución y disponibilidad reproducible.

### Acoplamiento

El backend importaba Sovaharmony durante el arranque y varios flujos estaban diseñados alrededor de un clasificador y scripts concretos. Un fallo en esa cadena podía impedir usar incluso las capacidades tabulares o administrativas.

### Entorno y Docker

Se encontraron, entre otros puntos:

- Prefect 2 en el entorno principal;
- Torch y herramientas de AutoML instaladas aunque no fueran necesarias para arrancar;
- imágenes históricas `imeag/mlops-*` como referencia;
- comunicación mediante `localhost` desde contenedores;
- montaje del directorio completo sobre código instalado;
- `service_started` donde se necesitaba disponibilidad real;
- instalación editable dirigida a un wheel;
- extracción ZIP y borrado de archivos con validación insuficiente.

## Clasificación de componentes

| Componente | Decisión | Resultado |
| --- | --- | --- |
| Licencia MIT | Reutilizar | Conservada sin cambios |
| Visión EEG + MLOps | Reutilizar | Se mantiene como objetivo principal |
| FastAPI | Reutilizar con refactor | API versionada y Pydantic |
| React/Vite + Ant Design | Reutilizar y ampliar | Se conserva el lenguaje visual original; navegación y pantallas se conectan a la API nueva |
| MLflow | Reutilizar con actualización | MLflow 3, logged models y aliases |
| Prefect | Reutilizar con actualización | Prefect 3 y ejecución observable |
| PostgreSQL | Reutilizar como perfil | SQLite local, PostgreSQL producción |
| Scripts de modelo fijo | Reemplazar | Registro de cinco factories |
| Pipeline EEG acoplado | Reemplazar | MNE-BIDS público incluido en la imagen soportada |
| Wheel `dirty` | Eliminar del core | No se distribuye ni se instala |
| Sovaharmony | Módulo opcional | Importación diferida y razón visible |
| Capturas y manuales obsoletos | Eliminar | Documentación corresponde al código actual |

## Compatibilidad conservada

- Objetivo científico de procesamiento y clasificación EEG.
- FastAPI, React, MLflow, Prefect y SQL como tecnologías principales.
- Cabecera, navegación lateral y patrones de Model Management del frontend original.
- Historial Git del fork.
- Licencia y atribución al trabajo original.
- Posibilidad de volver a integrar Sovaharmony mediante un adaptador cuando exista una distribución completa.

No se conserva compatibilidad binaria con el wheel local ni con endpoints heredados, porque hacerlo habría mantenido el acoplamiento que impedía la instalación limpia.

## Nueva ruta de migración

1. Ejecutar el perfil core y validar el demo tabular.
2. Incorporar datos propios mediante el contrato de Dataset.
3. Registrar una raíz BIDS montada y ejecutar el pipeline EEG público.
4. Validar separación por sujeto y artefactos científicos.
5. Solo después, empaquetar de forma pública y fijada toda la cadena Sovaharmony.
6. Implementar el adaptador usando la guía de pipelines, sin imports globales.

## Reconocimiento

El sistema nuevo existe gracias a la base conceptual y práctica del repositorio original. Esta documentación describe límites técnicos de reproducibilidad, no una valoración negativa de ese trabajo ni de sus autores.
