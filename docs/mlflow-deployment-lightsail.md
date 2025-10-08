# MLflow Deployment en AWS Lightsail con Caddy

Este documento describe la estructura final y configuración recomendada para desplegar un servidor de MLflow remoto en AWS Lightsail o EC2, protegido con Caddy (reverse proxy + autenticación) y listo para recibir experimentos desde notebooks o pipelines.

---

## 1. Arquitectura general

```
[Local Jupyter / Scripts / Pipelines]
        │
        ▼
   (HTTP + Auth)
http://<IP_PUBLICA>:80
        │
        ▼
┌──────────────────────────┐
│         Caddy           │  ← Reverse Proxy + Basic Auth
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│       MLflow Server      │  ← Tracking UI + API
│  sqlite:///mlflow.db     │
│  artifacts/ (local)      │
└──────────────────────────┘
```

- Caddy se encarga de recibir las peticiones HTTP, proteger el acceso con autenticación básica y redirigirlas a MLflow.  
- MLflow expone la UI y la API REST para registrar experimentos, parámetros, métricas y artefactos.  
- SQLite y `artifacts/` funcionan como almacenamiento local por defecto (pueden migrarse a S3/Azure más adelante).

---

## 2. Estructura de archivos en el servidor

```
/home/ec2-user/
├── mlflow-data/
│   ├── mlflow.db                 # Base de datos de runs y experimentos
│   └── artifacts/               # Modelos, métricas, logs, plots, etc.
├── .local/bin/mlflow            # Binario de MLflow (instalado con pip)
└── /etc/
    └── caddy/
        └── Caddyfile            # Configuración del reverse proxy
```

---

## 3. Configuración del servicio de MLflow (systemd)

Crear el archivo `/etc/systemd/system/mlflow.service`:

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
User=ec2-user
Group=ec2-user
ExecStart=/home/ec2-user/.local/bin/mlflow server \
  --backend-store-uri sqlite:////home/ec2-user/mlflow-data/mlflow.db \
  --default-artifact-root file:///home/ec2-user/mlflow-data/artifacts \
  --host 127.0.0.1 --port 5000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Activar y habilitar el servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
sudo systemctl status mlflow
```

---

## 4. Configuración de Caddy como reverse proxy

Crear el archivo `/etc/caddy/Caddyfile`:

```caddyfile
:80 {
    encode gzip
    basic_auth {
        admin <HASH_GENERADO_AQUI>
    }
    reverse_proxy 127.0.0.1:5000
}
```

Generar el hash de la contraseña:

```bash
caddy hash-password --plaintext "tu_password_seguro"
```

Reiniciar el servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable caddy
sudo systemctl start caddy
sudo systemctl reload caddy
```

---

## 5. Verificar acceso

En el navegador:

```
http://<IP_PUBLICA>/
```

Debe aparecer un prompt de autenticación básica.  
Después de iniciar sesión, se mostrará la interfaz de MLflow UI.

---

## 6. Conexión desde notebooks o scripts

Ejemplo de uso con credenciales en la URL:

```python
import mlflow

mlflow.set_tracking_uri("http://admin:tu_password_seguro@<IP_PUBLICA>:80")
mlflow.set_experiment("data-cleaning-demo")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("accuracy", 0.95)
```

---

## 7. Variables de entorno (opción más segura)

En lugar de poner la contraseña en el código:

```bash
export MLFLOW_TRACKING_URI=http://<IP_PUBLICA>:80
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=tu_password_seguro
```

Y en el notebook:

```python
import mlflow
mlflow.set_tracking_uri("http://<IP_PUBLICA>:80")
mlflow.set_experiment("phase1-lab")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("loss", 0.32)
```

---

## 8. Buenas prácticas recomendadas

- Separar `artifacts/`: si planeas almacenar muchos modelos, considera moverlo a un volumen EBS más grande o a S3.  
- HTTPS: si usas un dominio (`mlflow.tudominio.com`), Caddy puede emitir certificados SSL automáticamente.  
- Control de acceso: puedes combinar `basic_auth` con restricciones por IP o integrarlo con OAuth2 si necesitas mayor seguridad.  
- Convención de experimentos: usa nombres como `playground-<nombre>` para pruebas y `lab-<fase>` para ejecuciones oficiales.

---

## 9. Resumen del flujo completo

1. Instalar MLflow en Lightsail  
2. Crear carpeta de datos `~/mlflow-data/`  
3. Configurar servicio `mlflow.service`  
4. Instalar y configurar Caddy con autenticación básica  
5. Abrir puerto 80 en el firewall / Security Group  
6. Verificar acceso desde navegador  
7. Conectarse desde notebooks con la URI remota  

Con esta estructura, el entorno de MLflow queda preparado para:

- Recibir experimentos de todo el equipo desde notebooks, pipelines o scripts.  
- Servir como backend centralizado para registrar parámetros, métricas, artefactos y modelos.  
- Escalar en el futuro con almacenamiento en la nube o HTTPS sin cambiar la arquitectura base.