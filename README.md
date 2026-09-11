# Smart Sentinel 🛡️👁️

**Smart Sentinel** es un sistema interactivo de visión por computador y análisis de contexto visual impulsado por IA local en tiempo real. 

Combina seguimiento gestual de manos (**MediaPipe Tasks**), interfaz interactiva (**OpenCV**) y análisis multimodal contextualizado con el modelo **Llama 3.2 Vision** a través de **Ollama**.

---

## 🚀 Características

* **Interacción Gestual en Tiempo Real:** Detección de estado de la mano (abierta/cerrada), trazado interactivo fluido mediante suavizado exponencial y borrado inteligente.
* **Interfaz Táctil en Pantalla:** Botón virtual interactivamente pulsable mediante gestos (*Glassmorphism UI*).
* **Análisis Visual Multimodal Edge:** Inferencia local asíncrona mediante Llama 3.2 Vision o LLaVA sin enviar fotogramas a la nube.
* **Filtrado de Movimiento:** Detección por sustracción de fondo MOG2 para optimizar las llamadas al modelo VLM.

---

## 🛠️ Requisitos Previos

1. **Python 3.10 o superior** (Probado en Windows 11).
2. **Ollama:** Descarga e instala [Ollama para Windows](https://ollama.com/download/windows).

---

## 💻 Instalación y Uso

### 1. Clonar el repositorio
\\\ash
git clone https://github.com/TU_USUARIO/smart-sentinel.git
cd smart-sentinel
\\\

### 2. Crear y activar entorno virtual
\\\powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
\\\

### 3. Instalar dependencias
\\\powershell
pip install -r requirements.txt
\\\

### 4. Iniciar el servicio VLM (Ollama)
Asegúrate de tener el modelo de visión descargado en tu sistema:
\\\ash
ollama run llama3.2-vision
\\\
*(Nota: Si usas una versión de Ollama sin soporte mllama, puedes cambiar el modelo a \llava\ en \config/settings.py\)*.

### 5. Ejecutar la aplicación
\\\powershell
python main.py
\\\

---

## 🎮 Controles Gestuales

| Gesto | Acción |
| :--- | :--- |
| **Dedo índice levantado** | Modo dibujo (pintar en pantalla) |
| **Mano abierta (5 dedos)** | Limpiar lienzo completo |
| **Mantener índice sobre SALIR** | Pulsar botón virtual de salida (0.8s) |
| **Tecla q** | Cerrar la aplicación |

---

## 📄 Licencia
Este proyecto se distribuye bajo la licencia **MIT**.
