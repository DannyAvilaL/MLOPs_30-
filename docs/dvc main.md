# DVC: Versionado y Automatización de Datos en MLOps

## 1. Introducción: ¿Qué es DVC?

**DVC (Data Version Control)** es una herramienta diseñada para resolver una limitación fundamental de Git:  
Git es excelente versionando **código**, pero no está pensado para manejar **datasets, modelos o archivos grandes**.  
DVC complementa a Git permitiéndote versionar datos, automatizar pipelines de Machine Learning y garantizar **reproducibilidad y trazabilidad**.

En el contexto de la **Fase 1** del proyecto (*limpieza, exploración y versionado de datos*), DVC es una herramienta clave porque asegura que:

- Cada versión del dataset quede registrada.  
- Los cambios en los datos o scripts se detecten automáticamente.  
- El pipeline completo de limpieza y preprocesamiento se pueda reproducir en cualquier momento.  

---

## 2. Git vs DVC: Diferencias Fundamentales

| Herramienta | Qué versiona bien | Qué no maneja bien |
|------------|--------------------|--------------------|
| **Git**    | Código, notebooks, configuración | Archivos grandes (datasets `.csv`, modelos `.pkl`) |
| **DVC**    | Datos, modelos, resultados, pipelines | Código (eso sigue siendo tarea de Git) |

En resumen:

- Git controla **el cómo**: qué código se ejecuta.  
- DVC controla **el qué**: qué datos se usaron, qué resultados se generaron y cómo se transformaron.

---

## 3. Flujo Básico de Trabajo con DVC

### Paso 1: Inicializar DVC en tu proyecto

Se ejecuta una sola vez:

```bash
dvc init
```

Esto crea la carpeta `.dvc/`, que guarda la configuración interna.

---

### Paso 2: Rastrear tu dataset

Supongamos que tienes un archivo `data/raw/dataset.csv`. Puedes versionarlo así:

```bash
dvc add data/raw/dataset.csv
```

Esto crea un archivo `dataset.csv.dvc`, que sí se guarda en Git, mientras que el archivo original se ignora (`.gitignore`).

Cuando ejecutes:

```bash
dvc push
```

…el dataset se subirá a un **almacenamiento remoto** (por ejemplo, Google Drive o S3).

---

### Paso 3: Crear un pipeline reproducible (`dvc.yaml`)

Puedes definir todas las etapas de tu flujo de datos (limpieza, transformación, entrenamiento, etc.) en un archivo `dvc.yaml`.

Ejemplo:

```yaml
stages:
  cleaning:
    cmd: python scripts/cleaning.py data/raw/dataset.csv data/processed/clean.csv
    deps:
      - scripts/cleaning.py
      - data/raw/dataset.csv
    outs:
      - data/processed/clean.csv

  preprocessing:
    cmd: python scripts/preprocessing.py data/processed/clean.csv data/processed/preprocessed.csv
    deps:
      - scripts/preprocessing.py
      - data/processed/clean.csv
    outs:
      - data/processed/preprocessed.csv
```

Este archivo le dice a DVC:
- `cmd`: qué comando ejecutar.  
- `deps`: de qué archivos depende la etapa.  
- `outs`: qué archivo produce.

---

### Paso 4: Ejecutar el pipeline

Cuando quieras correr todo el flujo de transformación:

```bash
dvc repro
```

DVC revisará qué cambió (datasets, scripts, parámetros) y ejecutará **solo las etapas necesarias**.  
Si nada cambió, no hará nada.

---

### Paso 5: Versionar con Git y DVC

Cada vez que hagas cambios en el código o datos:

```bash
git add dvc.yaml data/raw/dataset.csv.dvc .gitignore
git commit -m "feat: add cleaning and preprocessing pipeline"
git push
dvc push
```

Luego, cualquier persona (o tú mismo en otro momento) puede recuperar la versión exacta con:

```bash
git pull
dvc pull
```

---

## 4. Sí: DVC Detecta Cambios en el CSV

Cuando haces `dvc add`, DVC guarda un **hash** del contenido del archivo.  
Si cambias el CSV (por ejemplo, agregas filas o lo limpias), ese hash cambia. Entonces, la próxima vez que ejecutes:

```bash
dvc repro
```

DVC sabe que el archivo cambió y vuelve a ejecutar el pipeline desde ese punto.

✅ Si el CSV no cambió → no hace nada.  
✅ Si cambió el CSV → vuelve a ejecutar desde esa etapa.  
✅ Si cambió un script → también se vuelve a ejecutar.

---

## 5. Ejemplo Real: Pipeline Simple

Supón que tienes este flujo:

```
data/raw/dataset.csv  --->  cleaning.py  --->  data/processed/clean.csv
```

Puedes configurar tu `dvc.yaml` así:

```yaml
stages:
  cleaning:
    cmd: python scripts/cleaning.py data/raw/dataset.csv data/processed/clean.csv
    deps:
      - scripts/cleaning.py
      - data/raw/dataset.csv
    outs:
      - data/processed/clean.csv
```

Ahora:

- Si cambias `cleaning.py` → se vuelve a ejecutar.  
- Si cambias `dataset.csv` → se vuelve a ejecutar.  
- Si no cambias nada → no hace nada.

---

## 6. Versiones de Dataset Controladas

Cada vez que cambies los datos y ejecutes `dvc add` o `dvc repro`, DVC puede subir esa nueva versión al almacenamiento remoto:

```bash
dvc push
```

Esto te permite mantener un historial como:

- `v1` - dataset original  
- `v2` - dataset con valores nulos eliminados  
- `v3` - dataset con nuevas columnas procesadas

Y puedes volver a cualquiera en cualquier momento con:

```bash
dvc pull
```

---

## 7. Beneficios para el Proyecto (y la Materia)

| Beneficio | Cómo te ayuda |
|----------|----------------|
| **Reproducibilidad** | Puedes regenerar resultados exactamente iguales. |
| **Versionado de datos** | Puedes regresar a cualquier versión del dataset. |
| **Historial de experimentos** | Puedes comparar transformaciones entre versiones. |
| **Colaboración segura** | Tus compañeros pueden trabajar con el mismo dataset sin enviarlo por email. |

---

## 8. Conectar un Almacenamiento Remoto (Opcional)

Puedes almacenar los datos en Google Drive:

```bash
dvc remote add -d myremote gdrive://<folder-id>
dvc push
```

Cualquiera que clone el repo puede luego obtener los datos con:

```bash
git pull
dvc pull
```

---

## 9. Resumen del Comportamiento de DVC

| Situación | Comportamiento |
|----------|------------------|
| Cambia el `.csv` | DVC detecta el cambio y ejecuta de nuevo. |
| Cambia el script | DVC detecta el cambio y ejecuta las etapas necesarias. |
| Nada cambió | DVC no hace nada (pipeline válido). |
| Quieres volver a una versión anterior | Puedes hacerlo fácilmente con `dvc pull`. |
| Quieres compartir datasets | DVC los gestiona fuera del repo con `dvc push`. |

---

## 10. Conclusión

Sí: **DVC es útil "por si cambia el archivo CSV"**, pero su poder va mucho más allá.  
Permite rastrear **qué cambió**, **cuándo cambió**, **cómo se procesó** y **qué resultados produjo**.  
Gracias a ello, tu proyecto de Machine Learning se vuelve **reproducible, trazable y colaborativo**, lo cual es un criterio clave en la rúbrica de la Fase 1.