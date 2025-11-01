# MNA - MLOps Data Cleaning Phase 1

**Proyecto académico - Maestría en Analítica (MNA) | Operaciones de Aprendizaje Automático (Gpo 10)**  
Tecnológico de Monterrey

## Descripción General

Este repositorio corresponde a la Fase 1 del proyecto de la materia Operaciones de Aprendizaje Automático (MLOps).  
El objetivo principal es realizar el análisis exploratorio (EDA), la limpieza, el preprocesamiento y el versionado del dataset asignado, preparando así la base de datos para etapas posteriores de modelado y despliegue en un flujo MLOps.

## Objetivos de la Fase 1

- Analizar y comprender el problema a resolver.
- Manipular y preparar los datos para su uso en modelos de Machine Learning.
- Realizar tareas de EDA (Exploratory Data Analysis).
- Limpiar datos eliminando valores nulos, inconsistentes, duplicados o inválidos.
- Aplicar técnicas de preprocesamiento: normalización, codificación, transformación, etc.
- Implementar herramientas de versionado de datos (DVC) para asegurar reproducibilidad y trazabilidad.
- Documentar las actividades por rol y los resultados obtenidos.

## Estructura del Repositorio

```
mna-mlops-data-cleaning-phase1/
│
├── data/
│   ├── raw/                # Dataset original sin modificar
│   ├── processed/          # Dataset limpio y transformado
│   └── versions/           # Versiones rastreadas con DVC
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_cleaning.ipynb   # Limpieza de datos
│   └── 03_preprocessing.ipynb
│
├── scripts/
│   ├── cleaning.py         # Código de limpieza automatizado
│   └── utils.py            # Funciones auxiliares
│
├── reports/               # Resultados, visualizaciones, informes
├── docs/                  # Documentación adicional del proyecto
├── dvc.yaml              # Pipeline de versionado de datos
├── requirements.txt      # Dependencias
└── README.md             # Documentación principal
```

## Tecnologías Utilizadas

- Python 3.x
- Pandas / NumPy - Manipulación de datos
- Matplotlib / Seaborn - Visualización exploratoria
- Scikit-learn - Preprocesamiento
- DVC - Versionado de datos
- Git / GitHub - Control de versiones de código

## Ejecución del Proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/<tu-usuario>/mna-mlops-data-cleaning-phase1.git
   cd mna-mlops-data-cleaning-phase1
   ```

2. Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   pip install -r requirements.txt
   ```

3. Ejecutar los notebooks de análisis y limpieza:
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

4. Sincronizar versiones de datos con DVC:
   ```bash
   dvc pull
   dvc repro
   ```

## Roles del Equipo

| Rol | Responsabilidad |
|-----|------------------|
| Data Engineer | Preparación, limpieza y versionado de datos. |
| ML Engineer | Diseño de pipeline de preprocesamiento y documentación técnica. |
| Project Lead | Coordinación, integración de entregables y documentación final. |

## Resultados Esperados

- Dataset limpio, transformado y listo para modelado.
- Documentación detallada del proceso de EDA y preprocesamiento.
- Versiones de dataset controladas y reproducibles con DVC.
- Visualizaciones descriptivas que respalden decisiones de ingeniería de datos.



# MNA – MLOps Proyecto Predictivo de Ausentismo | Fase 2

**Proyecto académico – Maestría en Analítica (MNA) | Operaciones de Aprendizaje Automático (Gpo 10)**  
Tecnológico de Monterrey

---

## Descripción General

Este repositorio corresponde a la **Fase 2** del proyecto de la materia Operaciones de Aprendizaje Automático (MLOps).  

El objetivo principal es **estructurar profesionalmente el proyecto**, implementar un **pipeline reproducible de modelado**, refactorizar y modularizar el código, y registrar **experimentos y modelos** utilizando MLflow y DVC, garantizando trazabilidad, reproducibilidad y buenas prácticas de ingeniería de Machine Learning.  

El sistema predice **horas de ausentismo laboral**, apoyando la planificación de personal en departamentos de Recursos Humanos (RRHH).

---

## Objetivos de la Fase 2

- Aplicar el **template Cookiecutter** para estructuración profesional del proyecto.  
- Refactorizar scripts en **módulos y clases**, aplicando principios de POO.  
- Construir un **pipeline reproducible de Scikit-Learn**, integrando preprocesamiento, entrenamiento y evaluación.  
- Registrar **experimentos y modelos** en MLflow, incluyendo métricas, hiperparámetros y artefactos.  
- Comparar resultados de modelos mediante **visualización de métricas** y reportes ejecutivos.  
- Documentar la participación de cada rol y las interacciones del equipo durante la fase.

---

## Resultados Esperados

- Pipeline reproducible para preprocesamiento, entrenamiento y evaluación.

- Modelos registrados y versionados con MLflow, incluyendo métricas y artefactos.

- Comparativas de resultados mediante gráficos y reportes ejecutivos.

- Código refactorizado y modularizado listo para mantenimiento y escalabilidad.

- Repositorio estructurado profesionalmente siguiendo Cookiecutter.

## Roles del Equipo

| Rol          | Responsabilidad                                              |
| :----------- | :----------------------------------------------------------- |
| Data Engineer | Preparación, limpieza y versionado de datos.                |
| ML Engineer   | Diseño de pipeline de preprocesamiento y documentación técnica. |
| Project Lead  | Coordinación, integración de entregables y documentación final. |


## Licencia

Este proyecto es parte del curso Operaciones de Aprendizaje Automático (Gpo 10) de la Maestría en Analítica (MNA) en el Tecnológico de Monterrey.  
Uso académico y educativo únicamente.
