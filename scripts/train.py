

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

import mlflow
from xgboost import XGBRegressor
from mlflow.models import infer_signature





def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage:\n")
        sys.stderr.write("\tpython train.py features-dir train-dir\n")
        sys.exit(1)

    features_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Archivos de features
    train_file = os.path.join(features_dir, "train.pkl")
    test_file = os.path.join(features_dir, "test.pkl")
    model_file = os.path.join(output_dir, "linear_model.pkl")

    # --- 1️⃣ Cargar los datos ---
    with open(train_file, "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(test_file, "rb") as f:
        X_test, y_test = pickle.load(f)

    # --- 2️⃣ Entrenar el modelo ---
    mlflow.set_tracking_uri("http://34.209.6.113:80")

    mlflow.set_experiment("/equipo30-xgboost")
    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        mlflow.log_params(params)
        # ---  Entrenar modelo XGBoost ---
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # ---  Predicciones ---
        y_pred = model.predict(X_test)

        # ---  Evaluar el modelo ---
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        print("📊 Métricas del modelo:")
        print(f"MAE  (Error Absoluto Medio): {mae:.2f}")
        print(f"MSE  (Error Cuadrático Medio): {mse:.2f}")
        print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
        print(f"R²   (Coeficiente de determinación): {r2:.3f}")


        # --- 6️⃣ Guardar el modelo ---
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        print(f"\n✅ Modelo XGBoost guardado en {model_file}")


if __name__ == "__main__":
    main()
