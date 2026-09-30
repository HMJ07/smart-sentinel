# Smart Sentinel

Sistema de videovigilancia inteligente basado en visión por computador e inteligencia artificial multimodal ejecutada localmente.

Smart Sentinel captura vídeo mediante una cámara, detecta actividad relevante y utiliza un modelo de visión local para analizar el contexto de las imágenes. El objetivo es construir un sistema de monitorización capaz de tomar decisiones a partir de eventos visuales sin depender de servicios externos de inferencia.

El proyecto combina Computer Vision, Edge AI y procesamiento asíncrono en un entorno local.

## Features

* Captura y procesamiento de vídeo en tiempo real.
* Detección de movimiento mediante OpenCV.
* Análisis visual mediante modelos multimodales locales.
* Inferencia mediante Ollama.
* Interacción mediante reconocimiento de gestos.
* Interfaz de línea de comandos.
* Procesamiento local de las imágenes.
* Detector de anomalías con severidad (armas, caídas, merodeo, aglomeraciones, objetos abandonados...).
* Alertas sonoras, banner en pantalla y avisos por Telegram (opcional).
* Control sin teclado por gestos: puño para salir, 👍 para reconocer alertas.
* Registro de eventos en SQLite con capturas y dashboard web.

## Architecture

```text
                         Camera
                            |
                            v
                    +---------------+
                    |    OpenCV     |
                    | Video Capture |
                    +-------+-------+
                            |
                            v
                    +---------------+
                    |    Motion     |
                    |   Detection   |
                    +-------+-------+
                            |
                    +-------+-------+
                    |               |
                  No Activity    Activity
                    |               |
                    |               v
                    |       +---------------+
                    |       | Frame Capture |
                    |       +-------+-------+
                    |               |
                    |               v
                    |       +---------------+
                    |       |    Ollama     |
                    |       | Local VLM     |
                    |       +-------+-------+
                    |               |
                    |               v
                    |       +---------------+
                    |       | Visual        |
                    |       | Analysis      |
                    |       +-------+-------+
                    |               |
                    +---------------+
                                    |
                                    v
                           Event / Alert
```

The current architecture is designed around a local processing pipeline:

1. Capture frames from the camera.
2. Detect relevant movement.
3. Capture a frame when an event is detected.
4. Send the frame to the local vision model.
5. Process the model response.
6. Generate an event or alert.

## Technology Stack

| Technology       | Purpose                           |
| ---------------- | --------------------------------- |
| Python 3.10+     | Core application                  |
| OpenCV           | Video capture and computer vision |
| Ollama           | Local model inference             |
| Llama 3.2 Vision | Multimodal visual analysis        |
| NumPy            | Numerical and image processing    |
| asyncio          | Asynchronous processing           |
| Rich             | CLI interface                     |
| Telegram Bot API | Event notifications               |

## Instalación rápida (doble clic)

Descarga el instalador de tu sistema desde la pestaña **Releases** de GitHub (no hace falta Python ni terminal):

| Sistema | Archivo | Cómo instalarlo |
| ------- | ------- | --------------- |
| macOS (Apple Silicon) | `SmartSentinel-macOS.dmg` | Ábrelo y arrastra **Smart Sentinel** a *Aplicaciones* |
| Windows 10/11 | `SmartSentinel-Setup.exe` (o el `.zip` portátil) | Doble clic → Siguiente → Instalar |

**Primer arranque**

* **macOS:** la app no está firmada con un Developer ID de Apple (requiere una cuenta de pago), así que macOS mostrará
  *«Apple no ha podido verificar que Smart Sentinel no contenga software malicioso»* la primera vez. Es normal.
  Arrastra la app a *Aplicaciones* (no la abras desde el `.dmg`) y elige **una** de estas dos vías:
  * **Terminal (funciona en todas las versiones):**
    `xattr -dr com.apple.quarantine "/Applications/Smart Sentinel.app"` y ábrela con normalidad.
  * **Ajustes del Sistema:** intenta abrirla, pulsa *Aceptar* y ve a *Ajustes del Sistema → Privacidad y seguridad*;
    al final de la pantalla, en *Seguridad*, pulsa **Abrir igualmente** e introduce tu contraseña.
    (En macOS 15 Sequoia y posteriores ya no sirve el *clic derecho → Abrir* de versiones anteriores.)

  Después acepta el permiso de **cámara** cuando lo pida. Solo hay que hacerlo una vez.
* **Windows:** si SmartScreen avisa, pulsa *Más información → Ejecutar de todas formas*. Acepta el permiso de cámara.
* En la ventana aparece durante 20 s la dirección del dashboard y su **clave**. También queda guardada en la carpeta de datos.
* El análisis visual con IA (Ollama) es **opcional**: sin él, todo lo demás funciona. Para activarlo instala
  [Ollama](https://ollama.com) y ejecuta `ollama pull llama3.2-vision`.
* Si algo falla al arrancar, se muestra un mensaje de error y el detalle queda en `sentinel.log`.

**Dónde se guardan los datos** (base de datos, capturas, clave del dashboard, registro y `settings.env`):

| Sistema | Carpeta |
| ------- | ------- |
| macOS | `~/Library/Application Support/SmartSentinel` |
| Windows | `%APPDATA%\SmartSentinel` |

Para configurar Telegram, la clave del dashboard, la cámara, etc. edita `settings.env` en esa carpeta
(viene con ejemplos comentados) y reinicia la aplicación.

### Rendimiento y aceleración por hardware

La detección de objetos usa un modelo **ONNX** con *ONNX Runtime*, que elige solo el mejor acelerador:
**CoreML / Neural Engine** en Mac y **DirectML** (cualquier GPU DirectX 12: NVIDIA, AMD, Intel) en Windows,
con CPU como respaldo automático. Medido en un MacBook con M4: 11.9 ms/fotograma con CoreML frente a
18.3 ms de PyTorch en CPU, con las mismas detecciones, y un instalador de ~180 MB en vez de más de 1 GB
porque no incluye PyTorch. Fuerza la CPU con `SENTINEL_YOLO_PROVIDER=cpu` si tu GPU diera problemas.

### Generar los instaladores tú mismo

Cada sistema se compila en su propio sistema (PyInstaller no compila de un sistema a otro):

```bash
pip install -r requirements.txt pyinstaller
bash packaging/build_mac.sh            # macOS -> dist/SmartSentinel-macOS.dmg
```

```powershell
pip install -r requirements.txt pyinstaller
./packaging/build_windows.ps1          # Windows -> dist\SmartSentinel-Windows.zip y SmartSentinel-Setup.exe
```

La primera vez, si falta `yolov8n.onnx`: `pip install ultralytics onnx && python scripts/export_model.py`.
El flujo `.github/workflows/build.yml` compila los dos en GitHub y los publica al crear una etiqueta `v*`
(`git tag v1.0.0 && git push --tags`). Cada instalador se comprueba con `"Smart Sentinel" --selftest`.

> **Docker:** los contenedores de Docker Desktop en Mac y Windows no pueden acceder a la webcam, y la aplicación
> abre una ventana; por eso el instalador nativo es la vía recomendada. El `Dockerfile` solo tendría sentido
> para un servidor con cámara IP (no implementado).

## Seguridad del dashboard

Todo el dashboard (vídeo, eventos, capturas) exige una **clave**. Si no defines `SENTINEL_DASHBOARD_PASSWORD`,
se genera una aleatoria en el primer arranque y se guarda en `dashboard_password.txt` (permisos 600). Tras 5 intentos
fallidos se bloquea la IP durante un minuto; las sesiones caducan a las 12 h y hay botón de cerrar sesión.
Por defecto solo escucha en `127.0.0.1`. Antes de exponerlo a la red (`SENTINEL_DASHBOARD_HOST=0.0.0.0`) recuerda que
el tráfico va sin cifrar (HTTP): úsalo solo en redes de confianza o detrás de una VPN o proxy HTTPS.

## Requirements

### Operating System

* Windows 10/11 64-bit
* macOS 12+ (Intel o Apple Silicon; concede permiso de cámara a tu terminal/IDE)

### Python

Python 3.9–3.12 (solo si ejecutas desde el código fuente; los instaladores ya lo incluyen).
`mediapipe` está fijado a la versión 0.10.14: las 1.x abortan al iniciar el rastreador de manos.

### Local AI Runtime

Smart Sentinel uses Ollama for local model inference.

Install Ollama and download the vision model:

```powershell
ollama pull llama3.2-vision
```

Verify the installation:

```powershell
ollama list
```

The model should appear as:

```text
llama3.2-vision
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/HMJ07/smart-sentinel.git
cd smart-sentinel
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Ollama

Make sure Ollama is running and that the required vision model is available:

```powershell
ollama list
```

If necessary:

```powershell
ollama pull llama3.2-vision
```

### 5. Run the application

```powershell
python main.py
```

## Controls

| Input | Action |
| ----- | ------ |
| Puño cerrado 1.5 s | **Salir sin teclado** (aparece un anillo de progreso; abre la mano para cancelar) |
| Índice extendido | Dibujar en pantalla |
| Palma abierta (5 dedos) | Borrar el dibujo |
| Tres dedos (índice, corazón, anular) 2 s | **Pánico silencioso**: guarda el evento, envía foto por Telegram y lo marca en el dashboard, **sin sonido ni aviso en pantalla** |
| ✌️ (índice y corazón) 1.5 s | Pausar / reanudar la vigilancia |
| Pulgar arriba 0.6 s | **Reconocer la alerta** (silencia el sonido hasta que empeore o pasen `SENTINEL_ACK_SECONDS`) |
| `Q` / `ESC` / cerrar ventana | Salir (alternativa con teclado/ratón) |

El tiempo de salida se ajusta con `SENTINEL_EXIT_HOLD_SECONDS`. Otras variables: `SENTINEL_CAMERA_INDEX`,
`SENTINEL_OLLAMA_MODEL`, `SENTINEL_DASHBOARD_HOST` (por defecto `127.0.0.1`; el dashboard no tiene autenticación,
usa `0.0.0.0` solo en redes de confianza).

## Detección de anomalías

`core/anomaly_detector.py` fusiona YOLO, seguimiento de personas, movimiento y el análisis del VLM en una
evaluación con nivel **NORMAL / WARNING / CRITICAL**. Cada anomalía se confirma en varios ciclos seguidos
(sin falsos positivos de un solo fotograma), se notifica una vez y se reavisa cada `SENTINEL_REALERT_SECONDS`.

| Anomalía | Cómo se detecta | Nivel |
| -------- | --------------- | ----- |
| `ARMA` | cuchillo / bate (crítico), tijeras (advertencia) | CRITICAL / WARNING |
| `CAIDA` | persona que pasa de vertical a tumbada y baja en la imagen, y sigue así 1.5 s | CRITICAL |
| `MERODEO` | persona quieta en la misma zona `SENTINEL_LOITER_SECONDS` (30 s) | WARNING |
| `AGLOMERACION` | ≥ `SENTINEL_CROWD_THRESHOLD` personas (×2 = crítico) | WARNING / CRITICAL |
| `OBJETO_ABANDONADO` | mochila, maleta o bolso quieto 20 s sin ninguna persona cerca | WARNING |
| `PRESENCIA_NOCTURNA` | personas entre `SENTINEL_NIGHT_START` y `SENTINEL_NIGHT_END` (23–6 h) | WARNING |
| `MOV_BRUSCO` | movimiento que ocupa >35 % de la imagen durante 1 s con personas | WARNING |
| `VLM` | palabras clave en la respuesta del modelo, con normalización de acentos, negaciones ("no hay arma") y caducidad de 30 s; se rebaja un nivel si YOLO no ve a nadie | WARNING / CRITICAL |

Dos o más advertencias simultáneas **escalan a CRITICAL**. Las alertas críticas pueden enviarse con foto a Telegram
definiendo `SENTINEL_TELEGRAM_TOKEN` y `SENTINEL_TELEGRAM_CHAT_ID` (si no están definidos no se envía nada).
Los umbrales son ajustables con variables de entorno `SENTINEL_*` (ver `config/settings.py`).

Pruebas: `python -m unittest discover -s tests -v`

## Gesture Interaction

Smart Sentinel includes a computer-vision-based interaction layer for detecting hand gestures.

The processing pipeline is structured as follows:

```text
Camera
   |
   v
Hand Detection
   |
   v
Finger Detection
   |
   v
Gesture Classification
   |
   v
Application Action
```

This component is intended to provide an alternative interaction mechanism without requiring conventional input devices.

## Project Structure

```text
smart-sentinel/
|
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
|
├── config/
│   └── settings.py
|
├── core/
│   ├── anomaly_detector.py
│   ├── tracker.py
│   ├── alerts.py
│   └── event_logger.py
|
├── modules/
│   ├── camera.py
│   ├── motion.py
│   ├── vision.py
│   ├── gestures.py
│   └── overlay.py
|
├── dashboard/app.py
├── tests/
|
└── assets/
    └── ...
```

The structure may change as the application is modularized and additional components are introduced.

## Privacy

Image processing is designed to remain on the local machine.

```text
Camera
   |
   v
Local Machine
   |
   +-- OpenCV
   |
   +-- Motion Detection
   |
   +-- Ollama
          |
          v
       Local VLM
```

Frames used for visual analysis are processed locally and do not need to be uploaded to an external AI inference service.

External services, such as Telegram, are only required for features that explicitly use them for notifications.

## Roadmap

### Implemented

* [x] Camera video capture
* [x] Motion detection
* [x] OpenCV-based processing
* [x] Local Ollama integration
* [x] Multimodal visual analysis
* [x] Interactive CLI
* [x] Gesture-based interaction

### In Progress

* [ ] Event detection pipeline
* [x] Intelligent alerts
* [x] Telegram integration
* [x] Event logging
* [x] Environment-based configuration
* [ ] Performance optimization
* [x] Web monitoring interface

### Planned

* [x] Person detection
* [ ] Event classification
* [x] Anomaly detection
* [ ] Multi-camera support
* [x] Event history
* [x] Remote dashboard
* [ ] Docker deployment

## Demo

A demonstration of the system will be added once the core monitoring pipeline is stable.

## Contributing

Contributions are welcome.

To contribute:

```powershell
git checkout -b feature/new-feature
```

Make the required changes, then commit them:

```powershell
git add .
git commit -m "feat: add new feature"
```

Push the branch:

```powershell
git push origin feature/new-feature
```

Open a Pull Request describing the changes and their purpose.

## License

This project is distributed under the MIT License.

See [`LICENSE`](LICENSE) for the full license text.

## Author

**HMJ07**

Smart Sentinel is a personal project focused on exploring the practical application of:

* Computer Vision
* Edge AI
* Multimodal AI
* Python
* Automation
* Local inference
* Cybersecurity-oriented monitoring systems
