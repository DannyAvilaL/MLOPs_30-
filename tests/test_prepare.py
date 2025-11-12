import pytest
import pandas as pd
from pathlib import Path
import sys
import numpy as np


# Se agrega la carpeta de 'scripts' al sys.path para que Python pueda encontrar 'prepare'
script_dir = Path(__file__).parent.parent / 'scripts'
sys.path.append(str(script_dir))

# Importando la clase
try:
    from prepare import DataPreparer 
except ImportError:
    print("\nError: No se pudo importar DataPreparer desde 'scripts/prepare.py'")
    print(f"Asegúrate de que 'scripts/prepare.py' exista y que sys.path sea correcto.")
    print(f"sys.path actual incluye: {script_dir}")
    sys.exit(1)


@pytest.fixture
def params_dict():
    """Una fixture que simula el diccionario de params.yaml para 'prepare'."""
    params =  {
        "split": 0.2,
        "seed": 42,
        "categorical_columns": ["Month of absence", "Day of the week", "Seasons", 
                                "Disciplinary failure", "Education", "Social drinker", 
                                "Social smoker"],
        "numerical_columns": ["Transportation expense", "Distance from Residence to Work", 
                              "Service time", "Age", "Work load Average/day", "Hit target", 
                              "Son", "Pet", "Weight", "Height", "Body mass index", 
                              "Reason for absence"]
    }
    
    return params

@pytest.fixture
def fake_raw_data(tmp_path):
    """
    Crea un archivo CSV falso en un directorio temporal (tmp_path).
    tmp_path también es una fixture mágica de pytest.
    
    Datos (7 filas):
    - Fila 0: Válida
    - Fila 1: Válida
    - Fila 2: Válida
    - Fila 3: Outlier categórico (Month 0) -> Debería ser eliminada
    - Fila 4: Outlier numérico (Transport 5000) -> Debería ser eliminada
    - Fila 5: NaN (col_con_nan) -> Debería ser eliminada
    - Fila 6: Outlier categórico (Reason 60) -> Debería ser eliminada
    
    Resultado esperado: 3 filas limpias (0, 1, 2)
    Split (0.2): 3 filas * 0.2 test = 0.6 -> 1 fila en test, 2 en train
    """
    # Creando un directorio 'data/raw' falso dentro del tmp_path
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    fake_csv_path = raw_dir / "test_data.csv"
    
    # Creando los datos de prueba
    data = {
        # Columnas categóricas
        "Month of absence": [1, 5, 12, 0, 3, 6, 7],  # Fila 3 es outlier (0)
        "Day of the week": [2, 3, 4, 5, 6, 7, 2],
        "Seasons": [1, 2, 3, 4, 1, 2, 3],
        "Disciplinary failure": [0, 1, 0, 1, 0, 1, 0],
        "Education": [1, 2, 3, 4, 1, 2, 3],
        "Social drinker": [0, 1, 0, 1, 0, 1, 0],
        "Social smoker": [0, 0, 1, 1, 0, 1, 0],
        # Columnas numéricas
        "Transportation expense": [100, 200, 150, 220, 5000, 180, 190], # Fila 4 es outlier (5000)
        "Distance from Residence to Work": [10, 20, 15, 30, 25, 12, 18],
        "Service time": [5, 10, 8, 12, 3, 7, 9],
        "Age": [30, 40, 35, 45, 50, 33, 42],
        "Work load Average/day": [200, 210, 220, 230, 240, 205, 215],
        "Hit target": [90, 92, 95, 98, 99, 91, 93],
        "Son": [1, 2, 0, 1, 3, 2, 1],
        "Pet": [0, 1, 2, 0, 1, 0, 1],
        "Weight": [70, 80, 75, 85, 90, 78, 82],
        "Height": [170, 180, 175, 185, 190, 178, 182],
        "Body mass index": [24, 25, 26, 27, 28, 24.5, 25.5],
        "Reason for absence": [1, 10, 25, 5, 8, 12, 60],    # Fila 6 es outlier (60)
        # Columnas para limpieza general
        "mixed_type_col": ["a", "b", "c", "d", "e", "f", "g"], # Debería ser eliminada
        "col_con_nan": [1, 2, 3, 4, 5, np.nan, 7]        # Fila 5 es outlier (NaN)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(fake_csv_path, index=False)
    
    print(f"CSV falso creado en: {fake_csv_path}")
    return fake_csv_path

def test_run_limpia_y_divide(params_dict, fake_raw_data, tmp_path):
    """
    Prueba unitaria completa del método .run()
    1. Prepara un entorno falso (params, csv falso, directorio de salida falso)
    2. Ejecuta el DataPreparer
    3. Verifica que los archivos de salida (train.tsv, test.tsv) se hayan creado
    4. Verifica que los datos se hayan limpiado y dividido correctamente
    """
    # 1. Preparar entorno falso
    # El 'input_file' es la ruta al CSV falso que creó la fixture 'fake_raw_data'
    input_file = fake_raw_data
    
    # Le decimos a la clase que guarde la salida en el directorio temporal 'tmp_path'
    output_dir = tmp_path / "data" / "prepared"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Creamos una instancia de la clase
    preparer = DataPreparer(params=params_dict, input_file=input_file)
    
    #! Sobreescribimos las rutas de salida para que apunten a tmp_path
    preparer.outout_dir = str(output_dir) 
    preparer.output_train_path = str(output_dir / "train.tsv")
    preparer.output_test_path = str(output_dir / "test.tsv")

    # 2. Ejecutar el DataPreparer
    preparer.run()

    # 3. Verificar que los archivos de salida existan
    train_out = Path(preparer.output_train_path)
    test_out = Path(preparer.output_test_path)
    
    assert train_out.exists(), "El archivo train.tsv no se creó"
    assert test_out.exists(), "El archivo test.tsv no se creó"

    # 4. Verificar el contenido
    df_train = pd.read_csv(train_out, sep="\t")
    df_test = pd.read_csv(test_out, sep="\t")
    
    # Datos originales: 7 filas
    # Filas eliminadas (deberían ser 4):
    # - Fila 3 (Month 0)
    # - Fila 4 (Transport 5000)
    # - Fila 5 (NaN)
    # - Fila 6 (Reason 60)
    # Filas restantes: 3 (Filas 0, 1, 2)
    
    # Split 0.2 en 3 filas (seed 42) -> 2 train, 1 test
    
    assert len(df_train) == 2, f"Se esperaban 2 filas en train, pero se obtuvieron {len(df_train)}"
    assert len(df_test) == 1, f"Se esperaban 1 fila en test, pero se obtuvo {len(df_test)}"

    # Se combinan los dataframes de salida para revisar la limpieza total
    df_full = pd.concat([df_train, df_test])
    
    assert "mixed_type_col" not in df_full.columns, "La columna 'mixed_type_col' no fue eliminada"
    
    # Verificar que no hay NaNs en ninguna parte
    assert df_full.isna().sum().sum() == 0, "Se encontraron NaNs después de la limpieza"
    
    # Verificar que las filas que sobrevivieron son las correctas (Age 30, 40, 35)
    sobrevivientes_age = df_full["Age"].tolist()
    sobrevivientes_age.sort()
    assert sobrevivientes_age == [30, 35, 40], "Las filas sobrevivientes no son las esperadas"

    print("\ntest_run_limpia_y_divide PASSED")