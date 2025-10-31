import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def main():
    np.set_printoptions(suppress=True)

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization_pipeline.py data-dir-path features-dir-path\n")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    train_input = os.path.join(in_path, "train.tsv")
    test_input = os.path.join(in_path, "test.tsv")
    train_output = os.path.join(out_path, "train.pkl")
    test_output = os.path.join(out_path, "test.pkl")

    os.makedirs(out_path, exist_ok=True)

    # --- Leer los archivos TSV ---
    df_train = pd.read_csv(train_input, sep="\t", encoding="utf-8")
    df_test = pd.read_csv(test_input, sep="\t", encoding="utf-8")

    # --- Columnas numéricas y categóricas ---
    numerical_cols = [
        "Transportation expense",
        "Distance from Residence to Work",
        "Service time",
        "Age",
        "Work load Average/day",
        "Hit target",
        "Son",
        "Pet",
        "Weight",
        "Height",
        "Body mass index",
    ]

    nominal_vars = ["Disciplinary failure", "Social drinker", "Social smoker", "Reason_Category"]

    target_col = "Absenteeism time in hours"

    # --- Separar X / y ---
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # --- Pipeline numérico ---
    numeric_pipeline = Pipeline(steps=[
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=5))
    ])

    # --- Pipeline categórico ---
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # --- Combinar en ColumnTransformer ---
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, nominal_vars)
    ])

    # --- Aplicar el pipeline ---
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # --- Nombres de columnas ---
    pca_features = [f"PCA_{i+1}" for i in range(preprocessor.named_transformers_["num"]["pca"].n_components_)]
    cat_features = list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(nominal_vars))
    all_features = pca_features + cat_features

    # --- Convertir a DataFrame ---
    df_train_final = pd.DataFrame(X_train_processed, columns=all_features)
    df_test_final = pd.DataFrame(X_test_processed, columns=all_features)

    # --- Guardar resultados ---
    with open(train_output, "wb") as f_out:
        pickle.dump((df_train_final, y_train), f_out)

    with open(test_output, "wb") as f_out:
        pickle.dump((df_test_final, y_test), f_out)

    # --- Guardar el pipeline completo para uso posterior ---
    pipeline_path = os.path.join(out_path, "preprocessing_pipeline.pkl")
    with open(pipeline_path, "wb") as f_pipe:
        pickle.dump(preprocessor, f_pipe)

    print(f"✅ Archivos generados:")
    print(f"  - Train features: {train_output}")
    print(f"  - Test features:  {test_output}")
    print(f"  - Pipeline guardado: {pipeline_path}")


if __name__ == "__main__":
    main()
