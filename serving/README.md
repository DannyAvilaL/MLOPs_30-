# Serving del modelo con MLflow y Caddy

Este servicio despliega el modelo de predicción de ausentismo laboral utilizando **MLflow Serving** detrás de un proxy **Caddy**, integrando la documentación **Swagger UI** en un único contenedor accesible desde el puerto **8000**.

---

## Estructura general

- **MLflow** corre internamente en el puerto `8080` y expone el endpoint `/invocations`.
- **Caddy** actúa como proxy inverso, gestionando rutas y exponiendo todas las solicitudes por el puerto `8000`.
- **Swagger UI** permite probar el endpoint `/predict` a través de la especificación definida en `openapi/openapi.yml`.

---

## Endpoints disponibles

| Método | Ruta           | Descripción |
|:-------|:----------------|:-------------|
| `POST` | `/predict`      | Redirige internamente a `/invocations` de MLflow para inferencias |
| `POST` | `/invocations`  | Endpoint nativo de MLflow para predicciones directas |
| `GET`  | `/ping`         | Health check del servidor MLflow |
| `GET`  | `/version`      | Devuelve la versión activa de MLflow |
| `GET`  | `/health`       | Health check rápido del proxy Caddy |
| `GET`  | `/docs`         | Interfaz Swagger UI integrada |
| `GET`  | `/openapi.yml`  | Especificación OpenAPI para Swagger |

---

## Construcción y ejecución

### Opción 1. Docker directo

```bash
cd serving
docker build -t absenteeism-serving:latest .
docker run -p 8000:8000 absenteeism-serving:latest
```

### Opción 2. Variables de entorno

Puedes definir las variables `MODEL_NAME` y `MODEL_STAGE` en un archivo `.env` para personalizar qué modelo se sirve desde MLflow:

```env
MLFLOW_HOST=34.209.6.113
MLFLOW_USER=admin
MLFLOW_PASS=**************
MODEL_NAME=absenteeism-predic
MODEL_STAGE=champion
```

---

## Flujo interno

1. El script `serve_remote.sh` carga el archivo `.env` y configura `MLFLOW_TRACKING_URI`.
2. Inicia `mlflow models serve` en `0.0.0.0:8080`.
3. Caddy escucha en el puerto `8000` y:
   - Redirige `/predict` → `/invocations`.
   - Sirve Swagger UI en `/docs` y `/openapi.yml`.
   - Reenvía el resto de las rutas (`/ping`, `/version`, etc.) hacia MLflow.
   - Responde `"ok"` en `/health`.

El resultado es un único servicio accesible externamente desde el puerto **8000**.

---

## Beneficios del diseño

- Cumple los requerimientos de la **Fase Final – MNA** (serving reproducible en contenedor).
- Usa **un solo puerto expuesto**, simplificando despliegue y pruebas locales.
- Mantiene **compatibilidad con las rutas nativas de MLflow**, útiles en CI/CD.
- No requiere dependencias como FastAPI o Uvicorn.
- Permite **documentar y probar el modelo directamente desde Swagger UI**.
- Listo para publicarse en un registro de imágenes:

```bash
docker tag absenteeism-serving:latest team30/absenteeism-serving:v1.0
docker push team30/absenteeism-serving:v1.0
```

---

## Ejemplos de uso

### Usando `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "PCA_1": 0.82,
      "PCA_2": -0.17,
      "PCA_3": 0.59,
      "PCA_4": 1.04,
      "PCA_5": -0.42,
      "Disciplinary failure_1.0": 0.0,
      "Social drinker_1.0": 1.0,
      "Social smoker_1.0": 0.0,
      "Reason_Category_2.0": 1.0,
      "Reason_Category_3.0": 0.0
    }]
  }'
```

### Usando Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "inputs": [{
        "PCA_1": 0.82,
        "PCA_2": -0.17,
        "PCA_3": 0.59,
        "PCA_4": 1.04,
        "PCA_5": -0.42,
        "Disciplinary failure_1.0": 0.0,
        "Social drinker_1.0": 1.0,
        "Social smoker_1.0": 0.0,
        "Reason_Category_2.0": 1.0,
        "Reason_Category_3.0": 0.0
    }]
}
response = requests.post(url, json=payload)
print(response.json())
```

---

## Notas adicionales

- El archivo `openapi/openapi.yml` define la estructura de entrada para Swagger UI (`http://localhost:8000/docs`).
- El contenedor usa el archivo `.env` para conectarse al servidor remoto de MLflow.
- Si no se define un modelo, se cargará un **modelo por defecto (`best_model@test`)**.
- Puede integrarse fácilmente en CI/CD, despliegues en AWS, Render o Fly.io.