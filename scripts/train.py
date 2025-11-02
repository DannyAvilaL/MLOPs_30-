import os
import sys
import pickle
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature

from dotenv import load_dotenv
load_dotenv()

class ModelTrainer:
    """
    Clase para la carga de features, el entrenamiento del modelo,
    la evaluación y el registro en MLflow.
    """
    
    def __init__(self, features_dir, output_dir, params):
        
        self.features_dir = features_dir
        self.output_dir = output_dir
        
        try:
            self.model_params = params["model_params"]
            self.mlflow_uri = params["mlflow_uri"]
            self.mlflow_experiment = params["mlflow_experiment"]
            self.model_output_name = params.get("model_output_name", "model.pkl")
            self.reports_dir = params["reports"]
            
        except KeyError as e:
            sys.stderr.write(f"Error: Parámetro '{e.key}' no encontrado en params.yaml sección 'train'\n")
            sys.exit(1)

        self.train_file = os.path.join(features_dir, "train.pkl")
        self.test_file = os.path.join(features_dir, "test.pkl")
        self.model_file_path = os.path.join(output_dir, self.model_output_name)

        
        os.makedirs(output_dir, exist_ok=True)           

    def load_data(self):
        # Carga los datos de features (train y test) desde archivos pickle.
        print(f"Cargando datos de train desde: {self.train_file}")
        with open(self.train_file, "rb") as f:
            X_train, y_train = pickle.load(f)
            
        print(f"Cargando datos de test desde: {self.test_file}")
        with open(self.test_file, "rb") as f:
            X_test, y_test = pickle.load(f)
            
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train):
        # Entrena el modelo XGBoost con los parámetros especificados.
        print("Entrenando el modelo XGBRegressor...")
        model = XGBRegressor(**self.model_params) #pasando como kwargs/dict
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        #Evalúa el modelo y devuelve un diccionario de métricas
        
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
        
        # Guarando las métricas
        os.makedirs(self.reports_dir, exist_ok=True)
        metrics_path = os.path.join(self.reports_dir, "metrics.json")
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Métricas guardadas en: {metrics_path}")
        
        print("Métricas del modelo:")
        print(f"  MAE  (Error Absoluto Medio): {mae:.2f}")
        print(f"  MSE  (Error Cuadrático Medio): {mse:.2f}")
        print(f"  RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
        print(f"  R2 (Coeficiente de determinación): {r2:.3f}")
        
        return metrics
    
    def log_to_mlflow(self, model, metrics, X_train, y_train):
        # Configura MLflow, inicia una run y registra params, métricas y el modelo.
        print(f"Registrando en MLflow (URI: {self.mlflow_uri})...")
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.mlflow_experiment)
        
        with mlflow.start_run():
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            # Registrar Parámetros
            mlflow.log_params(self.model_params)
            
            # Registrar Métricas
            mlflow.log_metrics(metrics)
            
            # Inferir firma del modelo
            signature = infer_signature(X_train, y_train)
            
            # Registrar el modelo
            mlflow.xgboost.log_model(
                xgb_model=model,
                name="Dream team model",
                signature=signature
            )
            print("Modelo, parámetros y métricas registrados en MLflow.")
            
            
    def save_model_pickle(self, model):
        # Guarda el modelo entrenado como un archivo pickle
        print(f"\nGuardando modelo en (pickle): {self.model_file_path}")
        with open(self.model_file_path, "wb") as f:
            pickle.dump(model, f)
        print("Modelo guardado localmente.")
    
    
    def run(self):
        # Orquesta la ejecución de todo el proceso de entrenamiento
        X_train, y_train, X_test, y_test = self.load_data()
        model = self.train_model(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Registrar en MLflow (parámetros, métricas y modelo)
        self.log_to_mlflow(model, metrics, X_train, y_train)
        
        # Guardar también una copia local del modelo (opcional pero bueno para DVC)
        self.save_model_pickle(model)

def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    Maneja la carga de argumentos y parámetros.
    """
    if len(sys.argv) != 3:
        sys.stderr.write("Error en argumentos. Uso:\n")
        sys.stderr.write("\tpython train_oop.py features-dir-path output-dir-path\n")
        sys.exit(1)
        
    features_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        # Carga la sección 'train' del params.yaml
        raw_params = yaml.safe_load(open("params.yaml"))["train"]
        # Expande variables de entorno como ${MLFLOW_TRACKING_URI}
        params = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in raw_params.items()}
    except Exception as e:
        sys.stderr.write(f"Error cargando 'params.yaml' (sección 'train'): {e}\n")
        sys.exit(1)

    # Instanciar y ejecutar el entrenador
    trainer = ModelTrainer(features_dir=features_dir, output_dir=output_dir, params=params)
    trainer.run()

if __name__ == "__main__":
    main()