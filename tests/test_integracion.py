import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import json


# Se agrega la carpeta de 'scripts' al sys.path para que Python pueda encontrar las clases
script_dir = Path(__file__).parent.parent / 'scripts'
sys.path.append(str(script_dir))

try:
    from prepare import DataPreparer
    from featurization import FeatureEngineer
    from train import ModelTrainer
except ImportError as e:
    print(f"\nError importando clases desde 'scripts/': {e}")
    print("Asegúrate de que 'prepare.py', 'featurization.py' y 'train.py' existan.")
    sys.exit(1)


@pytest.fixture
def full_params_dict():
    """
    Fixture que simula un params.yaml completo para todas las etapas.
    """
    return {
        "prepare": {
            "split": 0.2,
            "seed": 42,
            "categorical_columns": ["Month of absence", "Day of the week", "Seasons", 
                                    "Disciplinary failure", "Education", "Social drinker", 
                                    "Social smoker"],
            "numerical_columns": ["Transportation expense", "Distance from Residence to Work", 
                                  "Service time", "Age", "Work load Average/day", "Hit target", 
                                  "Son", "Pet", "Weight", "Height", "Body mass index", 
                                  "Reason for absence"]
        },
        "featurize": {
            "target_col": "Absenteeism time in hours",
            "n_components": 3, # 3 componentes para la prueba
            "numerical_cols": [
                "Transportation expense", "Distance from Residence to Work", 
                "Service time", "Age", "Work load Average/day", "Hit target", 
                "Son", "Pet", "Weight", "Height", "Body mass index"
            ],
            "nominal_cols": [
                "Disciplinary failure", "Social drinker", "Social smoker", 
                "Reason_Category" # Esta es generada en 'prepare'
            ]
        },
        "train": {
            "model_params": {
                "n_estimators": 5, # 5 para una prueba rápida
                "max_depth": 2,
                "random_state": 42
            },
            "mlflow_uri": "http://fake-mlflow-uri-for-test.com",
            "mlflow_experiment": "/test-integration-experiment",
            "model_output_name": "integration_test_model.pkl",
            "reports": "reports" # El nombre de la carpeta de reportes
        }
    }

@pytest.fixture
def fake_raw_data(tmp_path):
    """
    Crea un archivo CSV falso en un directorio temporal (tmp_path).
    (Versión simplificada del fixture de test_prepare.py)
    """
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    fake_csv_path = raw_dir / "integration_test_data.csv"
    
    # Creamos 20 filas de datos válidos para asegurar que pasen la limpieza
    data = {
        "Month of absence": np.random.randint(1, 13, size=20),
        "Day of the week": np.random.randint(2, 7, size=20),
        "Seasons": np.random.randint(1, 5, size=20),
        "Disciplinary failure": np.random.randint(0, 2, size=20),
        "Education": np.random.randint(1, 5, size=20),
        "Social drinker": np.random.randint(0, 2, size=20),
        "Social smoker": np.random.randint(0, 2, size=20),
        "Transportation expense": np.random.randint(100, 300, size=20),
        "Distance from Residence to Work": np.random.randint(5, 50, size=20),
        "Service time": np.random.randint(1, 15, size=20),
        "Age": np.random.randint(25, 50, size=20),
        "Work load Average/day": np.random.randint(200, 300, size=20),
        "Hit target": np.random.randint(90, 100, size=20),
        "Son": np.random.randint(0, 4, size=20),
        "Pet": np.random.randint(0, 3, size=20),
        "Weight": np.random.randint(60, 100, size=20),
        "Height": np.random.randint(160, 190, size=20),
        "Body mass index": np.random.randint(20, 30, size=20),
        "Reason for absence": np.random.randint(1, 28, size=20),
        "mixed_type_col": ["a"] * 20, # Será eliminada
        "Absenteeism time in hours": np.random.randint(1, 8, size=20) # Nuestro target
    }
    
    df = pd.DataFrame(data)
    df.to_csv(fake_csv_path, index=False)
    
    return str(fake_csv_path)


def test_full_pipeline_integration(full_params_dict, fake_raw_data, tmp_path, mocker):
    """
    Prueba de integración que ejecuta las 3 etapas del pipeline en secuencia.
    - Simula (mockea) MLflow para no requerir conexión.
    - Usa un sistema de archivos temporal (tmp_path) para todas las E/S.
    """
    
    # Simulamos todas las llamadas a 'mlflow' dentro del script 'train.py'
    mocker.patch("train.mlflow")
    mocker.patch("train.infer_signature")

    # DEFINIR RUTAS TEMPORALES
    raw_data_path = fake_raw_data # Ruta al CSV falso
    prepared_dir = tmp_path / "prepared"
    features_dir = tmp_path / "features"
    model_dir = tmp_path / "model"
    reports_dir = tmp_path / "reports" # Carpeta de reportes
    
    # --- ETAPA 1: PREPARE ---
    # Actualizar el diccionario de params de train para que apunte a 'reports_dir'
    full_params_dict['train']['reports'] = str(reports_dir)
    
    # Crear directorios que las clases esperan que existan
    prepared_dir.mkdir()
    features_dir.mkdir()
    model_dir.mkdir()

    preparer = DataPreparer(
        params=full_params_dict['prepare'], 
        input_file=raw_data_path
    )
    # Sobreescribir rutas de salida para que usen tmp_path
    preparer.outout_dir = str(prepared_dir)
    preparer.output_train_path = str(prepared_dir / "train.tsv")
    preparer.output_test_path = str(prepared_dir / "test.tsv")
    
    preparer.run()
    
    # Verificar salidas de 'prepare'
    assert (prepared_dir / "train.tsv").exists()
    assert (prepared_dir / "test.tsv").exists()

    # --- ETAPA 2: FEATURIZE ---
    print("\n--- INICIANDO ETAPA FEATURIZE ---")
    engineer = FeatureEngineer(
        in_path=str(prepared_dir), 
        out_path=str(features_dir), 
        params=full_params_dict['featurize']
    )
    
    engineer.run()

    # Verificar salidas de 'featurize'
    assert (features_dir / "train.pkl").exists()
    assert (features_dir / "test.pkl").exists()
    assert (features_dir / "preprocessing_pipeline.pkl").exists()
    
    # --- ETAPA 3: TRAIN ---
    print("\n--- INICIANDO ETAPA TRAIN ---")
    trainer = ModelTrainer(
        features_dir=str(features_dir), 
        output_dir=str(model_dir), 
        params=full_params_dict['train']
    )
    
    trainer.run()

    # Verificar salida de 'train' (modelo)
    model_output_name = full_params_dict['train']['model_output_name']
    assert (model_dir / model_output_name).exists()
    
    # Verificar salida de 'train' (métricas DVC)
    metrics_path = reports_dir / "metrics.json"
    assert metrics_path.exists()
    
    # Comprobar el contenido del JSON de métricas
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    assert "r2" in metrics
    assert "mae" in metrics
    assert isinstance(metrics['r2'], float)
    
    print("\nPrueba de integración completada con éxito.")
