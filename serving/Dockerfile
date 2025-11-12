# ==============================
# 1. Etapa base: MLflow con dependencias del modelo
# ==============================
FROM ghcr.io/mlflow/mlflow:v3.6.0 AS mlflow-base

WORKDIR /app

ENV MODEL_NAME=best_model \
    MODEL_STAGE=test \
    PORT=8080

# Instalar solo dependencias requeridas
RUN pip install --no-cache-dir -q xgboost==3.1.1 psutil==7.0.0

# Copiar archivos de configuración y scripts
COPY serve_remote.sh ./
COPY openapi ./openapi
COPY .env .env


# ==============================
# 2. Etapa Swagger + Caddy
# ==============================
FROM caddy:2.9.1-alpine AS caddy-base

WORKDIR /usr/share/caddy

# Instalar solo lo necesario
RUN apk add --no-cache curl tar

# Descargar e instalar Swagger UI (versión fija)
ARG SWAGGER_VERSION=v5.17.14
RUN mkdir -p swagger && \
    curl -sSL "https://github.com/swagger-api/swagger-ui/archive/refs/tags/${SWAGGER_VERSION}.tar.gz" \
    | tar xz --strip-components=2 -C swagger "swagger-ui-${SWAGGER_VERSION#v}/dist"

# Copiar OpenAPI YAML y configuración personalizada
COPY openapi/openapi.yml swagger/openapi.yml
COPY Caddyfile /etc/caddy/Caddyfile

# Configurar inicialización de Swagger UI
RUN echo 'window.onload = () => { \
  window.ui = SwaggerUIBundle({ \
    url: "openapi.yml", \
    dom_id: "#swagger-ui", \
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset], \
    layout: "StandaloneLayout" \
  }); \
};' > swagger/swagger-initializer.js


# ==============================
# 3. Etapa final: imagen integrada MLflow + Caddy + Swagger
# ==============================
FROM ghcr.io/mlflow/mlflow:v3.6.0

WORKDIR /app

# Copiar dependencias y scripts de MLflow
COPY --from=mlflow-base /app /app

# Copiar Caddy y Swagger desde la etapa anterior
COPY --from=caddy-base /usr/bin/caddy /usr/bin/caddy
COPY --from=caddy-base /etc/caddy /etc/caddy
COPY --from=caddy-base /usr/share/caddy/swagger /usr/share/caddy/swagger

EXPOSE 8000

# Iniciar MLflow y Caddy
CMD ["bash", "-c", "\
if [ -f /app/.env ]; then export $(grep -v '^#' /app/.env | xargs); fi && \
bash /app/serve_remote.sh & \
exec caddy run --config /etc/caddy/Caddyfile --adapter caddyfile"]