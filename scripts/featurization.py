

import os
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder




def main():


    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3 and len(sys.argv) != 5:
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

    #Se determinan las variables categoricas como sigue:
    categorical_cols = ["Reason for absence", "Month of absence", "Day of the week", "Seasons", "Disciplinary failure", "Education", "Social drinker", "Social smoker", "Reason_Category"]

    #Se determinan las variables numericas como sigue:
    numerical_cols = ["Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day", "Hit target", "Son", "Pet", "Weight", "Height", "Body mass index" ]

    # Aplicar transformación logarítmica solo a las columnas numéricas
    df_train[numerical_cols] = df_train[numerical_cols].apply(lambda x: np.log1p(x))
    df_test[numerical_cols] = df_test[numerical_cols].apply(lambda x: np.log1p(x))

    #Se aplica normalización
    scaler = MinMaxScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])



    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])

    #Se codifican las variables categóricas
    # Variables nominales (One-Hot Encoding)
    nominal_vars = [
    "Disciplinary failure",
    "Social drinker",
    "Social smoker",
    "Reason_Category"
    ]
    #One-Hot Encoding seguro
    df_train = pd.get_dummies(df_train, columns=nominal_vars, drop_first=True)
    train_cols = df_train.columns
    df_test = pd.get_dummies(df_test, columns=nominal_vars, drop_first=True)
    df_test = df_test.reindex(columns=train_cols, fill_value=0)


        # Separar variables predictoras (X) y objetivo (y)
    X_train = df_train.drop(columns=["Absenteeism time in hours"])
    y_train = df_train["Absenteeism time in hours"]

    X_test = df_test.drop(columns=["Absenteeism time in hours"])
    y_test = df_test["Absenteeism time in hours"]

    # Guardar los datos procesados como .pkl
    with open(train_output, "wb") as f_out:
        pickle.dump((X_train, y_train), f_out)

    with open(test_output, "wb") as f_out:
        pickle.dump((X_test, y_test), f_out)

    print(f"✅ Datos guardados en:\n{train_output}\n{test_output}")


if __name__ == "__main__":
    main()
