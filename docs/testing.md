# Pruebas unitarias con Pytest

Este documento explicará la implementación y uso de `pytest` y de `pytest-mocker` para realizar pruebas unitarias en las códigos de prepare y train. 

El objetivo de las pruebas unitarias es garantizar que la lógica dentro de cada clase del pipeline funcione como se espera.

- **Aislamiento:** Las pruebas **no** leen los datos reales de la carpeta `/data`. Crean datos falsos en memoria (`fixtures`) para ejecutarse de forma rápida y predecible.
- **Mocking (simulación):** Las pruebas **no** se conectan a servicios externos. Para eso se utilizó `pytest-mock`, con el fin de simular las llamadas a MLflow, asegurandeo que el código intenta conectarse y registrar métricas sin necesidad de un servidor real.

## Organización de scripts para pruebas unitarias y de integración

```
scripts/
  ├── featurization.py
  ├── prepare.py
  └── train.py
tests/
  ├── __init__.py
  ├── test_integracion.py
  ├── test_prepare.py
  └── test_train.py
```
Para instalarlo se tiene que relizar de la siguente forma:
```
(env)$ pip install pytest
(env)$ pip install pytest-mocker
```

## Modo de ejecución
Desde el directorio raíz del proyecto, sólo se ejecuta el siguiente comando
```
$ pytest -v
```
`pytest` descubrirá y ejecutará automáticamente todos los archivos `test_*.py` dentro de la carpeta `tests/`.

## Pruebas unitarias por clase
- `test/test_prepare.py`: Prueba la clase `DataPreparer`. Crea un CSV falso con datos sucios como NaNs y outliers y verifica que el script de limpieza produce los archivos `train.tsv` y `test.tsv` limpios y divididos correctamente.
- `tests/test_train.py`: Prueba la clase `ModelTrainer`. Simula las llamadas a `mflow` y verifica que el script llama a `mlflow.log_params`, `mlflow.log_metrics` y que guarda un archivo de modelo `.plk` al final.
- `tests/test_integracion.py`: Prueba todas las clases `DataPreparer`, `FeatureEngineer` y `ModelTrainer` utilizando un sistema de archivos temporal. Cada clase usa los artefactos de la etapa anterior como la entrada la la siguiente.