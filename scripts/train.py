

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

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
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- 3️⃣ Predicciones ---
    y_pred = model.predict(X_test)

    # --- 4️⃣ Evaluar el modelo ---
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("📊 Métricas del modelo:")
    print(f"MAE  (Error Absoluto Medio): {mae:.2f}")
    print(f"MSE  (Error Cuadrático Medio): {mse:.2f}")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
    print(f"R²   (Coeficiente de determinación): {r2:.3f}")

    # --- 5️⃣ Principales coeficientes ---
    coef_df = pd.DataFrame({
        "Variable": X_train.columns,
        "Coeficiente": model.coef_
    }).sort_values(by="Coeficiente", ascending=False)

    print("\n🔍 Principales variables que afectan el absentismo:")
    print(coef_df.head(10))

    # --- 6️⃣ Guardar el modelo ---
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Modelo guardado en {model_file}")


if __name__ == "__main__":
    main()
