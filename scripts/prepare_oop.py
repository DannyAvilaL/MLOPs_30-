import os
import sys
import yaml
import pandas as pd
#import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreparer:
    """
    Clase que carga, limpia, procesa
    y divide los datos para el pipeline de ML.
    """
    
    def __init__(self, params:dict, input_file:str):
        self.params = params
        self.input_file = input_file
        
        self.split_ratio = params["split"]
        self.seed = params["seed"]
        self.outout_dir = os.path.join("data", "prepared")
        self.output_train_path = os.path.join(self.outout_dir, "train.tsv")
        self.output_test_path = os.path.join(self.outout_dir, "test.tsv")
        
        #definiendo las columnas desde el params
        self.categorical_columns = params["categorical_columns"]
        self.numerical_columns = params["numerical_columns"]
        
        # asegurando el path de salida
        os.makedirs(self.outout_dir, exist_ok=True)
        
    
    def load_data(self):
        return pd.read_csv(self.input_file)
    
    
    def _limpia_datos(self, df):
        df_clean = df.copy()
        
        # Convirtiendo a columns numéricas
        for col in df_clean.select_dtypes(include="object").columns:
            df_clean[col] = pd.to_numeric(df_clean[col].str.replace(",", "").str.strip(),
            errors="coerce")
        
        if "mixed_type_col" in df_clean.columns:
            df_clean.drop("mixed_type_col", axis=1, inplace=True)
            
        # Elimiando valores nulos
        df_clean = df_clean.dropna()
        
        print("_limpia_datos()", df_clean.shape)
        return df_clean

    def _limpia_categoricos(self, df):
        df_clean = df.copy()
        
        # Limpiando razon de ausencia
        conditions = [
            (df_clean['Reason for absence'] >= 0) & (df_clean['Reason for absence'] <= 6),
            (df_clean['Reason for absence'] > 6) & (df_clean['Reason for absence'] <= 18),
            (df_clean['Reason for absence'] > 18) & (df_clean['Reason for absence'] <= 28)
        ]
        
        choices = [1, 2, 3]
        df_clean["Reason_Category"] = np.select(conditions, choices, default=np.nan)
        
        df_clean = df_clean[df_clean["Reason for absence"] <=50]
        self.categorical_columns.append("Reason_Category")
        
        # Quitando outliers para columnas Month of absence, Day of the week, Discplinary failure,
        # Education, Social drinker, Social smoker, Seasons,
        df_clean = df_clean[
            (df_clean['Month of absence'] >= 1) & (df_clean['Month of absence'] <= 12) &
            (df_clean['Day of the week'] <= 7) &
            (df_clean['Disciplinary failure'] <= 1) &
            (df_clean['Education'] <= 4) &
            (df_clean['Social drinker'] >= 0) & (df_clean['Social drinker'] <= 1) &
            (df_clean['Social smoker'] >= 0) & (df_clean['Social smoker'] <= 1) &
            (df_clean['Seasons'] <= 4)
        ]
        
        return df_clean
    
    def rango_intercuartil(self, df, column):
        # Devuelve un DataFrame filtrado por el rango IQR para una columna dada.
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Devuelve el DataFrame completo, filtrado
        return df[(df[column] >= limite_inferior) & (df[column] <= limite_superior)]
    
    def _limpia_numericos(self, df):
        # Limpia outliers de variables numéricas usando el método IQR.
        df_clean = df.copy()
        
        rows_before = len(df_clean)
        
        for col in self.numerical_columns:
            if col in df_clean.columns:
                # Reasigna el DataFrame completo con la versión filtrada
                df_clean = self.rango_intercuartil(df_clean, col)
            else:
                print(f"Advertencia: Columna numérica '{col}' no encontrada en el DataFrame.")

        rows_after = len(df_clean)
        print(f"Eliminadas {rows_before - rows_after} filas por outliers numéricos (IQR).")

        return df_clean
                
    def _limpia_data(self, df):
        df_clean = self._limpia_datos(df)
        df_clean = self._limpia_categoricos(df_clean)
        df_clean = self._limpia_numericos(df_clean)
        
        return df_clean
    
    def separa_data(self, df):
        df_train, df_test = train_test_split(
            df,
            test_size=self.split_ratio,
            random_state=self.seed
        )
        
        return df_train, df_test
    
    def guardar_data(self, df_train, df_test):
        df_train.to_csv(self.output_train_path, sep="\t", index=False)
        df_test.to_csv(self.output_test_path, sep="\t", index=False)
        
        
    def run(self):
        # Orquesta la ejecución de todo el proceso de entrenamiento
        df = self.load_data()
        df_clean = self._limpia_data(df)
        df_train, df_test = self.separa_data(df_clean)
        self.guardar_data(df_train, df_test)
        

def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Error en argumentos. Uso: \n")
        sys.stderr.write("\tpython model.py data-file\n")
        sys.exit(1)
        
    archivo_entrada = sys.argv[1]
    
    try:
        params = yaml.safe_load(open("params.yaml"))["prepare"]
    except Exception as e:
        sys.stderr.write(f"Error cargando 'params.yaml")
        sys.exit(1)
        
    preparador_datos = DataPreparer(params=params, input_file=archivo_entrada)
    preparador_datos.run()
    
    
if __name__ == "__main__":
    main()
            
        
