# Simulación y Monitoreo de Data Drift

Este módulo:
1) Genera un dataset de monitoreo con **drift** (cambios de media, features anuladas, ruido).
2) Evalúa el modelo (MLflow `pyfunc`) en **base** y **drift** y compara métricas.
3) Registra métricas, gráficos y alertas en **MLflow** y en `artifacts/drift_run`.

## Configuración
Edita `monitoring_drift/drift_config.yaml`:
- `validation_csv`: CSV de validación con las columnas de features y el `target_col`.
- `mlflow.tracking_uri` y `mlflow.model_uri`: a tu servidor y modelo.

## Ejecución
```bash
python monitoring_drift/run_drift.py --config monitoring_drift/drift_config.yaml
