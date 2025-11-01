import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class FeatureEngineer:
    
    """
    Clase que crea los features y el preprocesamiento de
    los datos de entrenamiento y prueba
    """
    
    def __init__(self, in_path, out_path, params):
        self.in_path = in_path
        self.out_path = out_path
        self.params = params
        
        self.train_input = os.path.join(in_path, "train.tsv")
        self.test_input = os.path.join(in_path, "test.tsv")
        self.train_output = os.path.join(out_path, "train.pkl")
        self.test_output = os.path.join(out_path, "test.pkl")
        self.pipeline_output_path = os.path.join(out_path, "preprocessing_pipeline.plk")
        
        os.makedirs(out_path, exist_ok=True)
        
        # Parámetros de featuring
        try:
            self.target_col = params["target_col"]
            self.numerical_cols = params["numerical_cols"]
            self.nominal_vars = params["nominal_cols"]
            self.pca_components = params["n_components"]
        except KeyError as e:
            sys.stderr.write(f"Error: Parámetro '{e.key}' no encontrado en params.yaml sección 'featurize'\n")
            sys.exit(1)
            
        
    def load_data(self):
        # Lee los archivos TSV de train y test.
        df_train = pd.read_csv(self.train_input, sep="\t", encoding="utf-8")
        df_test = pd.read_csv(self.test_input, sep="\t", encoding="utf-8")
        
        return df_train, df_test
    
    def pipeline(self):
        # Construye el ColumnTransformer con los pipelines de preprocesamiento.
        
        # pipeline numerico
        numeric_pipeline = Pipeline(steps=[
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=self.pca_components))
        ])
        
        # pipeline categorico
        categorical_pipeline = Pipeline(steps=[
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
        
        # combinar en un ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_pipeline, self.numerical_cols),
            ("cat", categorical_pipeline, self.nominal_vars)
        ])
        
        return preprocessor

    
    def _get_feature_names(self, preprocessor):
        # Para generar los nombres de las columnas después del preprocesamiento.
        pca_features = [f"PCA_{i+1}" for i in range(self.pca_components)]
        
        categorical_features = list(
            preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(self.nominal_vars)
        )
        
        return pca_features + categorical_features

    def split_xy(self, df_train, df_test):
        # Separa los DataFrames en features (X) y target (y).
        x_train = df_train.drop(columns=[self.target_col])
        y_train = df_train[self.target_col]
        
        x_test = df_test.drop(columns=[self.target_col])
        y_test = df_test[self.target_col]
        
        return x_train, y_train, x_test, y_test


    def save_artifacts(self, df_train, y_train, df_test, y_test, pipeline):
        # Guardar los datos procesados y el pipeline como archivos pickle
        with open(self.train_output, "wb") as f_out:
            pickle.dump((df_train, y_train), f_out)
            
        with open(self.test_output, "wb") as f_out:
            pickle.dump((df_test, y_test), f_out)
            
        with open(self.pipeline_output_path, "wb") as f_out:
            pickle.dump(pipeline, f_out)
            
    
    def run(self):
        # Orquesta la ejecución de todo el proceso de entrenamiento
        
        # cargar datos y separar
        df_train, df_test = self.load_data()
        x_train, y_train, x_test, y_test = self.split_xy(df_train, df_test)
        
        # construir el pipeline
        preprocessor = self.pipeline()
        
        # aplicar el pipeline
        x_train_processed = preprocessor.fit_transform(x_train)
        x_test_processed = preprocessor.transform(x_test)
        all_features = self._get_feature_names(preprocessor)
        
        # convertir a dataframes
        df_train_final = pd.DataFrame(x_train_processed, columns=all_features)
        df_test_final = pd.DataFrame(x_test_processed, columns=all_features)
        
        # guardar los resultados
        self.save_artifacts(df_train_final, y_train, df_test_final, y_test, preprocessor)

def main():
    """
    Función principal para ejecutar el script desde la línea de comandos.
    Maneja la carga de argumentos y parámetros.
    """
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
        sys.exit(1)
        
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    try:
        # Carga la seccion 'featurize' del params.yaml
        params = yaml.safe_load(open("params.yaml"))["featurize"]
    except Exception as e:
        sys.stderr.write(f"Error cargando 'params.yaml' (sección 'featurize'): {e}\n")
        sys.exit(1)
        
    # se crea la instacia y se ejecuta
    engineer = FeatureEngineer(in_path=in_path, out_path=out_path, params=params)
    engineer.run()
    

if __name__ == "__main__":
    main()