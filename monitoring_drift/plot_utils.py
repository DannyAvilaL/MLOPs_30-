import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_feature_hist_pair(df_base: pd.DataFrame, df_drift: pd.DataFrame,
                           features: List[str], outdir: str, bins: int = 30):
    ensure_dir(outdir)
    for col in features:
        if col not in df_base.columns or col not in df_drift.columns:
            continue
        plt.figure(figsize=(7,5))
        plt.hist(df_base[col].dropna(), bins=bins, alpha=0.6, label="base")
        plt.hist(df_drift[col].dropna(), bins=bins, alpha=0.6, label="drift")
        plt.title(f"Distribución: {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close()

def plot_error_bars(base_metrics: dict, drift_metrics: dict, outpath: str):
    labels = ["MAE", "RMSE"]
    base_vals = [base_metrics["mae"], base_metrics["rmse"]]
    drift_vals = [drift_metrics["mae"], drift_metrics["rmse"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6,5))
    plt.bar(x - width/2, base_vals, width, label="Base")
    plt.bar(x + width/2, drift_vals, width, label="Drift")
    plt.xticks(x, labels)
    plt.ylabel("Valor")
    plt.title("Comparación de error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
