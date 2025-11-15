from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def regression_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def compare_metrics(base_m: dict, drift_m: dict) -> dict:
    return {
        "mae_delta_abs": float(drift_m["mae"] - base_m["mae"]),
        "rmse_delta_abs": float(drift_m["rmse"] - base_m["rmse"]),
        "r2_drop_abs": float(base_m["r2"] - drift_m["r2"]),
    }

    """
    Compara métricas base vs. métricas con drift y evalúa umbrales.

    thresholds (opcional) puede contener:
      - "mae_delta_abs": alerta si (drift.mae - base.mae) > umbral
      - "rmse_delta_abs": alerta si (drift.rmse - base.rmse) > umbral
      - "r2_drop_abs": alerta si (base.r2 - drift.r2) > umbral
    """
    thresholds = thresholds or {}
    mae_delta = float(drift.get("mae", np.nan) - base.get("mae", np.nan))
    rmse_delta = float(drift.get("rmse", np.nan) - base.get("rmse", np.nan))
    r2_drop = float(base.get("r2", np.nan) - drift.get("r2", np.nan))

    alerts = {
        "mae_delta_abs": bool(
            not np.isnan(mae_delta) and mae_delta > thresholds.get("mae_delta_abs", np.inf)
        ),
        "rmse_delta_abs": bool(
            not np.isnan(rmse_delta) and rmse_delta > thresholds.get("rmse_delta_abs", np.inf)
        ),
        "r2_drop_abs": bool(
            not np.isnan(r2_drop) and r2_drop > thresholds.get("r2_drop_abs", np.inf)
        ),
    }
    return {
        "base": base,
        "drift": drift,
        "deltas": {
            "mae_delta_abs": mae_delta,
            "rmse_delta_abs": rmse_delta,
            "r2_drop_abs": r2_drop,
        },
        "alerts": alerts,
        "alert_any": any(alerts.values()),
    }
