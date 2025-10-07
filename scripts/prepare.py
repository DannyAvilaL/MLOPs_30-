

import os
import random
import re
import sys
import xml.etree.ElementTree

import yaml

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    # Test data set split ratio
    split = params["split"]



    input = sys.argv[1]
    output_train = os.path.join("data", "prepared", "train.tsv")
    output_test = os.path.join("data", "prepared", "test.tsv")

    
    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    """
    Lectura del archivo csv y se prepara el dataframe
    realizando limpieza de datos
    """
    df = pd.read_csv(input)

    #Convierte a variables numericas aquellas que contienen numeros
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.to_numeric(df[col].str.replace(',', '').str.strip(), errors='coerce')

    #Se genera un dataframe para preparar
    df_prepared = df

    #Eliminamos la columna mixed_type_col, ya que carece de sentido
    df_prepared.drop("mixed_type_col", axis=1, inplace=True)

    #Se eliminan valores nulos
    df_prepared = df_prepared.dropna()

    #Se determinan las variables categoricas como sigue:
    categorical_cols = ["Reason for absence", "Month of absence", "Day of the week", "Seasons", "Disciplinary failure", "Education", "Social drinker", "Social smoker"]

    #Se determinan las variables numericas como sigue:
    numerical_cols = ["Transportation expense", "Distance from Residence to Work", "Service time", "Age", "Work load Average/day", "Hit target", "Son", "Pet", "Weight", "Height", "Body mass index", "Absenteeism time in hours"  ]

    
    """
    Se limpian variables categoricas
    """

    #Reason for absence
    #Crea una columna Reason_Category asignando 1 a valores de Reason for absence de 0 a 6
    #Asigna a Reason_Category 2 a valores entre 6 y 18
    #Asigna a Reason_Category 3 a valores entre 18 y 28
    #Elimina renglones con valores mayores a 50
    # Crear la columna Reason_Category según los rangos
    conditions = [
        (df_prepared['Reason for absence'] >= 0) & (df_prepared['Reason for absence'] <= 6),
        (df_prepared['Reason for absence'] > 6) & (df_prepared['Reason for absence'] <= 18),
        (df_prepared['Reason for absence'] > 18) & (df_prepared['Reason for absence'] <= 28)
    ]

    choices = [1, 2, 3]

    df_prepared['Reason_Category'] = np.select(conditions, choices, default=np.nan)

    # Eliminar renglones con valores mayores a 50
    df_prepared = df_prepared[df_prepared['Reason for absence'] <= 50]
    categorical_cols.append("Reason_Category")

    # Para Month of absence se eliminan los meses con valor 0 o mayor a 12

    df_prepared = df_prepared[(df_prepared['Month of absence'] >= 1) & 
                          (df_prepared['Month of absence'] <= 12)]
    
    # Mantener solo filas donde Day of the week esté entre 1 y 7
    df_prepared = df_prepared[df_prepared['Day of the week'] <= 7]

    # Mantener solo filas donde Disciplinary failure sea 0 o 1
    df_prepared = df_prepared[df_prepared['Disciplinary failure'] <= 1]

    # Mantener solo filas donde Education sea 1, 2, 3 o 4
    df_prepared = df_prepared[df_prepared['Education'] <= 4]

    # Mantener solo filas donde Social drinker sea 0 o 1
    df_prepared = df_prepared[(df_prepared['Social drinker'] >= 0) & 
                          (df_prepared['Social drinker'] <= 1)]
    
    # Mantener solo filas donde Social smoker sea 0 o 1
    df_prepared = df_prepared[(df_prepared['Social smoker'] >= 0) & 
                          (df_prepared['Social smoker'] <= 1)]
    
    # Mantener solo filas donde Seasons sea menor o igual a 4
    df_prepared = df_prepared[df_prepared['Seasons'] <= 4]


    """
    Se limpian datos numericos
    """
    #Se eliminan outliers con metodo IQR para cada uno de las columnas en numerical_cols
    for col in numerical_cols:
        Q1 = df_prepared[col].quantile(0.25)
        Q3 = df_prepared[col].quantile(0.75)
        IQR = Q3 - Q1
    
    # Filtrar filas dentro del rango [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    df_prepared = df_prepared[(df_prepared[col] >= Q1 - 1.5*IQR) & 
                              (df_prepared[col] <= Q3 + 1.5*IQR)]
    

    # División según el parámetro "split" definido en params.yaml
    df_train, df_test = train_test_split(
        df_prepared,
        test_size=split,
        random_state=params["seed"]
    )



    # --- Escritura en archivos TSV ---
    df_train.to_csv(output_train, sep="\t", index=False)
    df_test.to_csv(output_test, sep="\t", index=False)



    print(f"Datos preparados y divididos exitosamente:")
    print(f"Train → {output_train} ({len(df_train)} filas)")
    print(f"Test  → {output_test} ({len(df_test)} filas)")


if __name__ == "__main__":
    main()
