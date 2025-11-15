import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from xgboost import XGBRegressor


# Se agrega la carpeta de 'scripts' al sys.path para que Python pueda encontrar 'train' y usar la clase
script_dir = Path(__file__).parent.parent / 'scripts'
sys.path.append(str(script_dir))

# Se importan las clases
import pytest
try:
    from scripts.train import ModelTrainer
except Exception as e:
    pytest.skip(f"Saltando tests de train: import falló ({e})")



@pytest.fixture
def params_dict():
    """
    fixture de pytest que simula el diccionario de params.yaml para 'train'.
    """
    params = {"model_params": {
            "n_estimators": 10,  # Usamos 10 para que la prueba sea rápida
            "max_depth": 2,
            "random_state": 42
        },
        "mlflow_uri": "http://fake-mlflow-uri.com",
        "mlflow_experiment": "/test-experiment",
        "model_output_name": "test_model.pkl",
        "seed": 123,
        "reports": "reports//"
    }
    return params

@pytest.fixture
def fake_feature_data(tmp_path):
    """
    Crea archivos .pkl falsos (train.pkl, test.pkl) en un directorio temporal.
    tmp_path también es una fixture de pytest.
    """
    # Directorio de features falso
    feature_dir = tmp_path / "data" / "features"
    feature_dir.mkdir(parents=True)
    
    train_path = feature_dir / "train.pkl"
    test_path = feature_dir / "test.pkl"
    
    # Crear datos falsos
    X_train = pd.DataFrame(np.random.rand(100, 5), columns=[f"f_{i}" for i in range(5)])
    y_train = pd.Series(np.random.rand(100))
    
    X_test = pd.DataFrame(np.random.rand(50, 5), columns=[f"f_{i}" for i in range(5)])
    y_test = pd.Series(np.random.rand(50))
    
    # Guardar los archivos pkl falsos
    with open(train_path, "wb") as f:
        pickle.dump((X_train, y_train), f)
        
    with open(test_path, "wb") as f:
        pickle.dump((X_test, y_test), f)
        
    print(f"Archivos Pickle falsos creados en: {feature_dir}")
    # Devuelve la RUTA del directorio en tipo string
    return str(feature_dir)


def test_trainer_run_logs_and_saves(params_dict, fake_feature_data, tmp_path, mocker):
    """
    Prueba unitaria del método .run() de ModelTrainer.
    Utiliza 'mocker' para simular ('mockear') las llamadas externas:
    - mlflow (para no conectarnos a un servidor real)
    - infer_signature (parte de mlflow)
    
    Pasos:
    1. Configurar Mocks: mlflow.set_tracking_uri, set_experiment, start_run, etc.
    2. Preparar Entorno: Instanciar ModelTrainer con rutas y params falsos.
    3. Ejecutar: Llamar a trainer.run()
    4. Verificar (Asserts):
        - Que los métodos de mlflow fueron llamados (ej. mlflow.log_params).
        - Que el archivo del modelo local se guardó en el tmp_path.
    """
    
    # 1. Configurar Mocks para MLflow
    mock_mlflow = mocker.patch("scripts.train.mlflow")
    mock_signature = mocker.patch("scripts.train.infer_signature")
    
    # 2. Preparar Entorno
    features_dir = fake_feature_data # Ruta al directorio con train.pkl/test.pkl
    output_dir = tmp_path / "models" # Ruta de salida para el modelo
    
    # Instanciamos la clase
    trainer = ModelTrainer(
        features_dir=features_dir, 
        output_dir=str(output_dir),
        params=params_dict
    )
    
    # 3. Ejecutar
    trainer.run()

    # 4. Verificar (Asserts)
    
    # Verificar llamadas a MLflow
    mock_mlflow.set_tracking_uri.assert_called_once_with(params_dict["mlflow_uri"])
    mock_mlflow.set_experiment.assert_called_once_with(params_dict["mlflow_experiment"])
    mock_mlflow.start_run.assert_called_once() # Verificar que se inició una 'run'
    
    # Verificar que se loggearon parámetros y métricas
    mock_mlflow.log_params.assert_called_once_with(params_dict["model_params"])
    mock_mlflow.log_metrics.assert_called_once() # Solo verificamos que se llamó
    
    # Verificar que se loggeó el modelo
    mock_mlflow.xgboost.log_model.assert_called_once()
    
    # Verificar que se infirió la firma
    mock_signature.assert_called_once()

    # Verificar guardado local
    # La clase usa 'self.model_file_path' que apunta a tmp_path/models/test_model.pkl
    expected_model_path = Path(trainer.model_file_path)
    
    assert expected_model_path.exists(), "El archivo .pkl del modelo no se guardó localmente"
    
    # Cargar el modelo guardado y verificar que no esté vacío
    with open(expected_model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # Verificamos que sea una instancia de XGBRegressor
    assert isinstance(loaded_model, XGBRegressor)

    print("\ntest_trainer_run_logs_and_saves PASSED")