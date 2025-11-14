from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

def compare_metrics(base: Dict[str, float], drift: Dict[str, float]) -> Dict[str, float]:
    return {
        "delta_mae": drift["mae"] - base["mae"],
        "delta_rmse": drift["rmse"] - base["rmse"],
        "drop_r2": base["r2"] - drift["r2"]
    }
