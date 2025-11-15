import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml
import mlflow
from typing import Dict

from metrics import regression_metrics, compare_metrics
from plot_utils import ensure_dir, plot_feature_hist_pair, plot_error_bars
def _to_float_safe(series):
    """Convierte a float de forma robusta: coerce a NaN y rellena con la mediana (o 0.0 si no existe)."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    if pd.isna(med):
        med = 0.0
    return s.fillna(med).astype(float)

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_validation(csv_path, target_col):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    df = df.dropna(subset=[target_col])
    df = df.apply(pd.to_numeric, errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def simulate_drift(X: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    Xd = X.copy(deep=True)

    # 0) Normaliza columnas "numéricas problemáticas" (con '?', 'unknown', 'error', etc.)
    #    Si una columna es object pero tiene valores numéricos posibles, la convertimos y rellenamos NaN con mediana.
    for col in Xd.columns:
        if Xd[col].dtype == object:
            # Intento de parseo numérico conservador
            parsed = pd.to_numeric(Xd[col], errors="coerce")
            if parsed.notna().sum() > 0:           # hay al menos algún valor numérico
                med = parsed.median()
                if pd.isna(med):
                    med = 0.0
                Xd[col] = parsed.fillna(med).astype(float)

    # 1) mean shifts (desplazamiento de medias)
    for col, shift in (cfg.get("mean_shifts") or {}).items():
        if col in Xd.columns:
            # conversión robusta antes de sumar
            Xd[col] = _to_float_safe(Xd[col]) + float(shift)

    # 2) zero-out features (apagar columnas completas)
    for col in (cfg.get("zero_out_features") or []):
        if col in Xd.columns:
            # si no es numérica, la forzamos a 0.0
            if not np.issubdtype(Xd[col].dtype, np.number):
                Xd[col] = _to_float_safe(Xd[col])
            Xd[col] = 0.0

    # 3) ruido gaussiano proporcional al std (solo en columnas numéricas)
    std_fraction = (cfg.get("noise") or {}).get("std_fraction", 0.0)
    if std_fraction > 0:
        # Asegura que todas las columnas numéricas sean float
        for col in Xd.columns:
            if np.issubdtype(Xd[col].dtype, np.number):
                Xd[col] = _to_float_safe(Xd[col])

        numeric_cols = [c for c in Xd.columns if np.issubdtype(Xd[c].dtype, np.number)]
        for col in numeric_cols:
            std = Xd[col].std()
            scale = std_fraction * (std if (std is not None and std > 0) else 1.0)
            noise = np.random.normal(loc=0.0, scale=scale, size=len(Xd))
            Xd[col] = Xd[col] + noise

    # 4) sample fraction
    frac = float(cfg.get("sample_fraction", 1.0))
    if 0 < frac < 1.0:
        Xd = Xd.sample(frac=frac, random_state=42).reset_index(drop=True)

    return Xd


def get_model(mlflow_cfg: Dict):
    import os, mlflow

    # Configura conexión
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    # Carga credenciales (si existen)
    if "username" in mlflow_cfg and "password" in mlflow_cfg:
        os.environ["MLFLOW_TRACKING_USERNAME"] = str(mlflow_cfg["username"])
        os.environ["MLFLOW_TRACKING_PASSWORD"] = str(mlflow_cfg["password"])

    # Establece experimento, si existe
    if "experiment" in mlflow_cfg:
        mlflow.set_experiment(mlflow_cfg["experiment"])

    # Carga modelo (desde Registry o run_id)
    model_uri = mlflow_cfg["model_uri"]
    print(f"🔗 Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def main(config_path: str):
    cfg = load_config(config_path)
    outdir = cfg["outputs_dir"]
    ensure_dir(outdir)

    # reproducibilidad
    np.random.seed(42)

    # 1) cargar base (validación)
    X_base, y_base = load_validation(cfg["validation_csv"], cfg["target_col"])

    # 2) simular drift
    X_drift = simulate_drift(X_base, cfg["drift"])

    # 3) cargar modelo desde MLflow
    model = get_model(cfg["mlflow"])

    # 4) predecir y métricas
    y_pred_base = model.predict(X_base)
    y_pred_drift = model.predict(X_drift)

    base_m = regression_metrics(y_base, y_pred_base)

    # Alinear y_true con X_drift si hubo muestreo
    if len(X_drift) != len(X_base):
        y_base_aligned = y_base.iloc[X_drift.index].reset_index(drop=True)
    else:
        y_base_aligned = y_base.reset_index(drop=True)

    drift_m = regression_metrics(y_base_aligned, y_pred_drift)

    # Comparación de métricas (produce *_abs)
    # compare_metrics DEBE devolver: {'mae_delta_abs', 'rmse_delta_abs', 'r2_drop_abs'}
    deltas = compare_metrics(base_m, drift_m)

    # 5) umbrales / alertas (asegura nombres consistentes)
    th = cfg["thresholds"]
    alerts = {
        "mae_delta_alert": float(deltas.get("mae_delta_abs", 0.0)) > float(th.get("mae_delta_abs", 0.0)),
        "rmse_delta_alert": float(deltas.get("rmse_delta_abs", 0.0)) > float(th.get("rmse_delta_abs", 0.0)),
        "r2_drop_alert": float(deltas.get("r2_drop_abs", 0.0)) > float(th.get("r2_drop_abs", 0.0)),
    }
    any_alert = any(alerts.values())

    # 6) guardar artefactos locales
    X_drift_out = X_drift.copy()
    X_drift_out["y_true"] = y_base_aligned.values
    X_drift_out["y_pred_base"] = np.array(y_pred_base[:len(X_drift_out)])
    X_drift_out["y_pred_drift"] = np.array(y_pred_drift)
    X_drift_out.to_csv(os.path.join(outdir, "drift_dataset_predictions.csv"), index=False)

    with open(os.path.join(outdir, "metrics_base.json"), "w") as f:
        json.dump(base_m, f, indent=2)
    with open(os.path.join(outdir, "metrics_drift.json"), "w") as f:
        json.dump(drift_m, f, indent=2)
    with open(os.path.join(outdir, "metrics_deltas.json"), "w") as f:
        json.dump(deltas, f, indent=2)
    with open(os.path.join(outdir, "alerts.json"), "w") as f:
        json.dump({"alerts": alerts, "any_alert": any_alert}, f, indent=2)

    # 7) gráficos
    feat_altered = list((cfg["drift"].get("mean_shifts") or {}).keys()) + list((cfg["drift"].get("zero_out_features") or []))
    plot_feature_hist_pair(
        X_base.reset_index(drop=True),
        X_drift.reset_index(drop=True),
        features=feat_altered,
        outdir=os.path.join(outdir, "plots", "features")
    )
    plot_error_bars(base_m, drift_m, os.path.join(outdir, "plots", "error_comparison.png"))

    # 8) logging en MLflow (usa .get(...) para evitar KeyError)
    mlflow.set_experiment("absenteeism_drift_monitoring")
    with mlflow.start_run(run_name="drift_simulation") as run:
        mlflow.log_params({
            "mean_shifts": str(cfg["drift"].get("mean_shifts")),
            "zero_out_features": str(cfg["drift"].get("zero_out_features")),
            "noise_std_fraction": cfg["drift"].get("noise", {}).get("std_fraction", 0),
            "sample_fraction": cfg["drift"].get("sample_fraction", 1.0),
        })
        mlflow.log_metrics({
            "base_mae": float(base_m.get("mae", 0.0)),
            "base_rmse": float(base_m.get("rmse", 0.0)),
            "base_r2": float(base_m.get("r2", 0.0)),
            "drift_mae": float(drift_m.get("mae", 0.0)),
            "drift_rmse": float(drift_m.get("rmse", 0.0)),
            "drift_r2": float(drift_m.get("r2", 0.0)),
            "delta_mae": float(deltas.get("mae_delta_abs", 0.0)),
            "delta_rmse": float(deltas.get("rmse_delta_abs", 0.0)),
            "drop_r2": float(deltas.get("r2_drop_abs", 0.0)),
        })
        mlflow.set_tags({
            "mae_delta_alert": str(alerts.get("mae_delta_alert", False)),
            "rmse_delta_alert": str(alerts.get("rmse_delta_alert", False)),
            "r2_drop_alert": str(alerts.get("r2_drop_alert", False)),
            "any_alert": str(any_alert),
        })
        mlflow.log_artifacts(outdir)

    print("\n== Baseline metrics ==", base_m)
    print("== Drift metrics ==", drift_m)
    print("== Deltas ==", deltas)
    print("== Alerts ==", alerts, "| any_alert =", any_alert)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="monitoring_drift/drift_config.yaml")
    args = parser.parse_args()
    main(args.config)
