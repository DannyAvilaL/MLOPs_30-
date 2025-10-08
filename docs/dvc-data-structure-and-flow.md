# Estructura de Datos y Flujo con DVC

## 1. Flujo de datos en un proyecto de Machine Learning

En un proyecto de Machine Learning, los datos pasan por distintas etapas desde su forma cruda hasta versiones procesadas y listas para modelar. Para organizar estas etapas y facilitar el versionado con **DVC (Data Version Control)**, se suele usar la siguiente estructura de carpetas:

```
data/
├── raw/         ← datos originales sin modificar
├── processed/   ← datos limpios y transformados
└── versions/    ← versiones archivadas o "congeladas" de datasets importantes
```

Cada carpeta cumple un rol distinto en el ciclo de vida del proyecto:

---

## 2. Significado y uso de cada carpeta

### `data/raw/` – Datos originales de entrada

- Contiene los archivos **sin modificar** tal como fueron recibidos o descargados.  
- Es el **punto de partida** de todo el pipeline.  
- Siempre deben ser versionados con DVC (`dvc add`).  
- Nunca se deben editar manualmente.

Ejemplo de lectura en notebooks de EDA o scripts iniciales:

```python
df = pd.read_csv("data/raw/work_absenteeism_modified.csv")
```

---

### `data/processed/` – Datos derivados y transformados

- Contiene los resultados de limpieza, transformación y preprocesamiento.  
- Estos archivos se generan automáticamente con scripts o pipelines, no se modifican manualmente.  
- Son los que se utilizan normalmente en las etapas posteriores de modelado o entrenamiento.

Ejemplo de lectura en scripts o notebooks de modelado:

```python
df = pd.read_csv("data/processed/preprocessed.csv")
```

---

### `data/versions/` – Versiones archivadas de datasets importantes

- Carpeta opcional pero recomendada para guardar versiones “congeladas” de datasets.  
- Se utiliza cuando se quiere conservar un snapshot exacto antes de cambiar el pipeline, o al finalizar un experimento.  
- Muy útil para reproducir resultados pasados o documentar el estado final de un proyecto.

Ejemplos de archivos:

```
data/versions/
├── v1_clean_2025-10-01.csv
├── v2_preprocessed_no_outliers.csv
```

---

## 3. Ejemplo de flujo de datos

```
data/raw/work_absenteeism_modified.csv   ← leído por 01_eda.ipynb y cleaning.py
        │
        ▼
data/processed/clean.csv                 ← salida del script de limpieza
        │
        ▼
data/processed/preprocessed.csv          ← salida del preprocesamiento
        │
        ▼
data/versions/v1_final_preprocessed.csv  ← versión congelada antes del modelado
```

---

## 4. Integración con DVC

La integración con DVC permite que este flujo sea completamente reproducible.  
Un ejemplo de pipeline en `dvc.yaml` sería:

```yaml
stages:
  cleaning:
    cmd: python scripts/cleaning.py data/raw/work_absenteeism_modified.csv data/processed/clean.csv
    deps:
      - scripts/cleaning.py
      - data/raw/work_absenteeism_modified.csv
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

- `deps`: archivos de entrada que DVC vigila.  
- `outs`: archivos de salida que DVC genera y versiona.

---

## 5. Buenas prácticas

| Carpeta | Se lee desde | Cuándo se usa | Quién la genera |
|--------|---------------|----------------|------------------|
| `raw/` | En notebooks de EDA o scripts de limpieza | Al inicio del pipeline | Manualmente o por descarga externa |
| `processed/` | En notebooks posteriores o scripts de modelado | Después del procesamiento | Scripts de limpieza/preprocesamiento |
| `versions/` | En análisis posteriores o entregas finales | Para guardar snapshots estables | Manual o script de archivado |

---

## 6. Regla de oro

- **Nunca sobrescribas `raw/`** – debe mantenerse siempre igual.  
- **Siempre genera `processed/` automáticamente** – nunca lo edites a mano.  
- **Guarda en `versions/` solo los datasets importantes** – sirven como puntos de control históricos.

---

## 7. Ejemplo del ciclo completo

1. `01_eda.ipynb` lee datos desde `data/raw/`.  
2. `cleaning.py` procesa esos datos y escribe en `data/processed/clean.csv`.  
3. `preprocessing.py` transforma los datos limpios y genera `data/processed/preprocessed.csv`.  
4. Se guarda una versión congelada en `data/versions/` para reproducir experimentos o entregar resultados.

---

## 8. Conclusión

- **`raw/`** es el origen de la verdad: los datos tal como llegaron.  
- **`processed/`** es el estado intermedio: datos preparados para modelar.  
- **`versions/`** son snapshots históricos: estados importantes guardados permanentemente.  

Seguir esta estructura garantiza que tu proyecto cumpla con las buenas prácticas de MLOps, sea **reproducible, trazable y fácil de mantener** con DVC.