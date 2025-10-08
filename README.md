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

## Licencia

Este proyecto es parte del curso Operaciones de Aprendizaje Automático (Gpo 10) de la Maestría en Analítica (MNA) en el Tecnológico de Monterrey.  
Uso académico y educativo únicamente.