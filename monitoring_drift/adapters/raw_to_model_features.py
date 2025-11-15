#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapter: de CSV crudo (Absenteeism) a las FEATURES que exige el modelo en MLflow.

Salidas de columnas EXACTAS que el modelo espera:
- PCA_1 .. PCA_5
- Disciplinary failure_1.0
- Social drinker_1.0
- Social smoker_1.0
- Reason_Category_2.0
- Reason_Category_3.0
- __y__ (si el CSV trae el target)

Uso:
  python monitoring_drift/adapters/raw_to_model_features.py \
    --in /ruta/al/raw.csv \
    --out monitoring_drift/base_validation_features.csv
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def map_reason_to_category(val):
    """
    Mapea 'Reason for absence' (1..28) en 2 categorías (2 y 3) usadas por el modelo.
    Retorna 0 cuando es NaN o fuera de rango. Acepta strings tipo '26.0'.
    """
    if pd.isna(val):
        return 0
    try:
        v = int(float(val))
    except (ValueError, TypeError):
        return 0
    if 1 <= v <= 14:
        return 2
    elif 15 <= v <= 28:
        return 3
    else:
        return 0


def safe_get_numeric(series):
    """Convierte a numérico con errors='coerce' y devuelve float Series."""
    return pd.to_numeric(series, errors="coerce")


def ensure_binary_onehot(df_raw, col_name, out_name):
    """
    Crea una columna one-hot binaria *_1.0 a partir de un campo binario/booleano.
    Si no existe la col original, genera ceros.
    """
    if col_name in df_raw.columns:
        s = safe_get_numeric(df_raw[col_name])
        s = (s == 1).astype(float)
    else:
        s = pd.Series(0.0, index=df_raw.index)
    return pd.Series(s, name=out_name)


def build_features(df_raw: pd.DataFrame, n_pca_components: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Construye el dataframe de features EXACTAS requeridas por el modelo:
      ['PCA_1','PCA_2','PCA_3','PCA_4','PCA_5',
       'Disciplinary failure_1.0','Social drinker_1.0','Social smoker_1.0',
       'Reason_Category_2.0','Reason_Category_3.0']
    y agrega '__y__' si existe target en el CSV.
    """
    df = df_raw.copy()
    # Normaliza encabezados (espacios, etc.)
    df.columns = df.columns.str.strip()

    feats = pd.DataFrame(index=df.index)

    # ----- One-hot de categorías derivadas de "Reason for absence"
    reason_num = safe_get_numeric(df.get("Reason for absence"))
    cat_series = reason_num.map(map_reason_to_category).fillna(0).astype(int)
    feats["Reason_Category_2.0"] = (cat_series == 2).astype(float)
    feats["Reason_Category_3.0"] = (cat_series == 3).astype(float)

    # ----- One-hot binario para columnas esperadas por el modelo
    feats["Disciplinary failure_1.0"] = ensure_binary_onehot(df, "Disciplinary failure", "Disciplinary failure_1.0")
    feats["Social drinker_1.0"] = ensure_binary_onehot(df, "Social drinker", "Social drinker_1.0")
    feats["Social smoker_1.0"] = ensure_binary_onehot(df, "Social smoker", "Social smoker_1.0")

    # ----- PCA_1..PCA_5 desde un set razonable de numéricas del CSV
    # Selecciona numéricas candidatas (excluimos las ya creadas para no mezclarlas)
    # Puedes ampliar/reducir esta lista según tus columnas reales:
    candidate_numeric = []
    for col in df.columns:
        if col in ["Absenteeism time in hours", "__y__", "Reason for absence",
                   "Disciplinary failure", "Social drinker", "Social smoker"]:
            continue
        # intenta numeric
        s = safe_get_numeric(df[col])
        # criterio: si al menos 50% son numéricos -> lo usamos
        valid_frac = s.notna().mean()
        if valid_frac >= 0.5:
            candidate_numeric.append(col)

    if len(candidate_numeric) == 0:
        # fallback: si no encontramos nada, usa zeros para PCA
        pca_array = np.zeros((len(df), n_pca_components), dtype=float)
    else:
        X_num = df[candidate_numeric].apply(pd.to_numeric, errors="coerce")

        # Imputa NaNs con mediana
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_num.values)

        # Estandariza de forma simple (media 0, var 1) para que PCA sea estable
        # Evitamos dependencia de StandardScaler para mantener simple
        means = np.nanmean(X_imp, axis=0)
        stds = np.nanstd(X_imp, axis=0)
        stds[stds == 0] = 1.0
        X_std = (X_imp - means) / stds

        # PCA
        n_comp = min(n_pca_components, X_std.shape[1]) if X_std.shape[1] > 0 else 0
        if n_comp == 0:
            pca_array = np.zeros((len(df), n_pca_components), dtype=float)
        else:
            pca = PCA(n_components=n_comp, random_state=random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z = pca.fit_transform(X_std)

            # Si n_comp < requerido, rellena con ceros
            if n_comp < n_pca_components:
                pad = np.zeros((Z.shape[0], n_pca_components - n_comp), dtype=float)
                pca_array = np.concatenate([Z, pad], axis=1)
            else:
                pca_array = Z[:, :n_pca_components]

    for i in range(n_pca_components):
        feats[f"PCA_{i+1}"] = pca_array[:, i] if pca_array.shape[1] > i else 0.0

    # ----- Target como __y__ (si existe)
    # Ajusta el nombre correctamente a tu CSV:
    possible_targets = ["Absenteeism time in hours", "Absenteeism.time.in.hours"]
    target_col = None
    for cand in possible_targets:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is not None:
        feats["__y__"] = safe_get_numeric(df[target_col])

    # Ordena columnas para hacerlas 100% reproducibles y legibles
    ordered_cols = [
        "PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5",
        "Disciplinary failure_1.0", "Social drinker_1.0", "Social smoker_1.0",
        "Reason_Category_2.0", "Reason_Category_3.0"
    ]
    if "__y__" in feats.columns:
        ordered_cols.append("__y__")

    # Asegura que todas existan (si faltó alguna, la crea con cero)
    for c in ordered_cols:
        if c not in feats.columns:
            feats[c] = 0.0

    feats = feats[ordered_cols]
    return feats


def main(in_csv: str, out_csv: str):
    in_path = Path(in_csv)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(in_path)
    feats = build_features(df_raw)
    feats.to_csv(out_path, index=False)
    print(f"[OK] Features guardadas en: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_csv", required=True, help="Ruta al CSV crudo")
    parser.add_argument("--out", dest="out_csv", required=True, help="Ruta de salida CSV de features")
    args = parser.parse_args()
    main(args.in_csv, args.out_csv)
