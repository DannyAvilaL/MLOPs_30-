#!/usr/bin/env bash
set -e

# ==============================
#   MLFLOW INFERENCE SERVER
# ==============================

# --- CARGA VARIABLES DESDE .env ---
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "No se encontró el archivo .env"
  exit 1
fi

# --- CONSTRUYE LA URI DEL MODELO ---
MODEL_URI="models:/${MODEL_NAME}@${MODEL_STAGE}"

# --- VARIABLES DE ENTORNO ---
export MLFLOW_TRACKING_URI="http://${MLFLOW_USER}:${MLFLOW_PASS}@${MLFLOW_HOST}"
export MLFLOW_REGISTRY_URI="${MLFLOW_TRACKING_URI}"

# --- MOSTRAR CONFIGURACIÓN ---
echo "=============================="
echo "  MLFLOW INFERENCE SERVER"
echo "=============================="
echo "Servidor: ${MLFLOW_HOST}"
echo "Modelo:   ${MODEL_URI}"
echo "Puerto:   ${PORT}"
echo "=============================="
echo

# --- SERVIR EL MODELO ---
mlflow models serve \
  -m "${MODEL_URI}" \
  -h 0.0.0.0 \
  -p "${PORT}" \
  --no-conda \
  --env-manager=local