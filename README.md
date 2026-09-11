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
* Arquitectura preparada para notificaciones y registro de eventos.

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

## Requirements

### Operating System

* Windows 10/11 64-bit

### Python

Python 3.10–3.12 is recommended.

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

| Input         | Action                |
| ------------- | --------------------- |
| `Q`           | Exit application      |
| `ESC`         | Clean shutdown        |
| Hand gestures | Interface interaction |

Gesture mappings may change as the interaction system evolves.

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
├── modules/
│   ├── camera.py
│   ├── motion.py
│   ├── vision.py
│   └── gestures.py
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
* [ ] Intelligent alerts
* [ ] Telegram integration
* [ ] Event logging
* [ ] Environment-based configuration
* [ ] Performance optimization
* [ ] Web monitoring interface

### Planned

* [ ] Person detection
* [ ] Event classification
* [ ] Anomaly detection
* [ ] Multi-camera support
* [ ] Event history
* [ ] Remote dashboard
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
