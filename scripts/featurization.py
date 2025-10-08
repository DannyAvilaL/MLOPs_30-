import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def main():
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    train_input = os.path.join(in_path, "train.tsv")
    test_input = os.path.join(in_path, "test.tsv")
    train_output = os.path.join(out_path, "train.pkl")
    test_output = os.path.join(out_path, "test.pkl")

    os.makedirs(out_path, exist_ok=True)

    # Leer los archivos TSV
    df_train = pd.read_csv(train_input, sep="\t", encoding="utf-8")
    df_test = pd.read_csv(test_input, sep="\t", encoding="utf-8")

    # Columnas numéricas y categóricas
    numerical_cols = ["Transportation expense", "Distance from Residence to Work", "Service time",
                      "Age", "Work load Average/day", "Hit target", "Son", "Pet",
                      "Weight", "Height", "Body mass index"]

    nominal_vars = ["Disciplinary failure", "Social drinker", "Social smoker", "Reason_Category"]

    # --- Log-transform y normalización ---
    df_train[numerical_cols] = df_train[numerical_cols].apply(lambda x: np.log1p(x))
    df_test[numerical_cols] = df_test[numerical_cols].apply(lambda x: np.log1p(x))

    scaler = MinMaxScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    # --- PCA sobre columnas numéricas ---
    n_components = 5
    pca = PCA(n_components=n_components)
    df_train_pca = pca.fit_transform(df_train[numerical_cols])
    df_test_pca = pca.transform(df_test[numerical_cols])

    # Crear DataFrame solo con columnas PCA
    df_train_pca = pd.DataFrame(df_train_pca, columns=[f"PCA_{i+1}" for i in range(n_components)])
    df_test_pca = pd.DataFrame(df_test_pca, columns=[f"PCA_{i+1}" for i in range(n_components)])

    # --- Codificación de variables categóricas ---
    df_train_cat = pd.get_dummies(df_train[nominal_vars], drop_first=True)
    df_test_cat = pd.get_dummies(df_test[nominal_vars], drop_first=True)

    # Alinear columnas categóricas train/test
    df_test_cat = df_test_cat.reindex(columns=df_train_cat.columns, fill_value=0)

    # Concatenar PCA + categóricas
    df_train_final = pd.concat([df_train_pca, df_train_cat], axis=1)
    df_test_final = pd.concat([df_test_pca, df_test_cat], axis=1)

    # Separar X / y
    y_train = df_train["Absenteeism time in hours"]
    y_test = df_test["Absenteeism time in hours"]

    # Guardar como .pkl
    with open(train_output, "wb") as f_out:
        pickle.dump((df_train_final, y_train), f_out)

    with open(test_output, "wb") as f_out:
        pickle.dump((df_test_final, y_test), f_out)

    print(f"✅ Datos guardados en:\n{train_output}\n{test_output}")


if __name__ == "__main__":
    main()
