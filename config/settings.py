import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
IS_FROZEN = getattr(sys, "frozen", False)                  # True dentro del instalador (PyInstaller)
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", ROOT_DIR))    # modelos empaquetados (solo lectura)


def _user_data_dir():
    """Carpeta de datos del usuario (base de datos, capturas, claves, ajustes)."""
    override = os.environ.get("SENTINEL_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if not IS_FROZEN:
        return ROOT_DIR                                     # en desarrollo todo queda en el proyecto
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SmartSentinel"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "SmartSentinel"
    return Path.home() / ".local" / "share" / "SmartSentinel"


DATA_DIR = _user_data_dir()
DATA_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_TEMPLATE = """# Ajustes de Smart Sentinel. Quita el '#' de una línea para activarla y reinicia la aplicación.
# SENTINEL_DASHBOARD_PASSWORD=mi-clave-segura
# SENTINEL_TELEGRAM_TOKEN=
# SENTINEL_TELEGRAM_CHAT_ID=
# SENTINEL_CAMERA_INDEX=0
# SENTINEL_OLLAMA_MODEL=llama3.2-vision
# SENTINEL_VLM_ENABLED=1
# SENTINEL_DASHBOARD_HOST=127.0.0.1
"""


def _load_settings_file():
    """Lee DATA_DIR/settings.env (CLAVE=valor). Las variables de entorno reales tienen prioridad."""
    path = DATA_DIR / "settings.env"
    if not path.exists():
        if IS_FROZEN:
            path.write_text(SETTINGS_TEMPLATE, encoding="utf-8")
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_settings_file()


def _env(name, default):
    """Lee una variable de entorno SENTINEL_<name>, casteando al tipo del valor por defecto."""
    raw = os.environ.get(f"SENTINEL_{name}")
    if raw is None:
        return default
    try:
        return type(default)(raw)
    except ValueError:
        return default


class Config:
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    IS_FROZEN = IS_FROZEN
    DB_PATH = str(DATA_DIR / "events.db")
    CAPTURES_DIR = str(DATA_DIR / "captures")
    LOG_PATH = str(DATA_DIR / "sentinel.log")
    YOLO_MODEL_PATH = str(RESOURCE_DIR / "yolov8n.onnx")
    HAND_MODEL_PATH = str(RESOURCE_DIR / "hand_landmarker.task"
                          if (RESOURCE_DIR / "hand_landmarker.task").exists()
                          else DATA_DIR / "hand_landmarker.task")   # si no viene empaquetado, se descarga aquí
    YOLO_PROVIDER = _env("YOLO_PROVIDER", "auto")           # auto | cpu
    VLM_ENABLED = _env("VLM_ENABLED", 1)

    CAMERA_INDEX = _env("CAMERA_INDEX", 0)
    FRAME_WIDTH = _env("FRAME_WIDTH", 1280)
    FRAME_HEIGHT = _env("FRAME_HEIGHT", 720)
    FPS = _env("FPS", 30)

    MOTION_THRESHOLD = 25
    MIN_CONTOUR_AREA = 2500

    # YOLO es lo más costoso: se ejecuta cada N fotogramas y se reutiliza el resultado.
    DETECT_EVERY_N_FRAMES = _env("DETECT_EVERY_N_FRAMES", 3)

    OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2-vision")
    OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")

    VLM_PROMPT = (
        "Analiza el fotograma minuciosamente en espanol. "
        "Si observas personas, objetos en la mano o situaciones anómalas, "
        "responde en una frase corta. Si no, responde: 'Escena normal'."
    )

    # Salir sin teclado: mantener el puño cerrado este tiempo.
    EXIT_HOLD_SECONDS = _env("EXIT_HOLD_SECONDS", 1.5)

    # Por defecto solo accesible desde este equipo. Usa SENTINEL_DASHBOARD_HOST=0.0.0.0
    # para abrirlo a la red local (el dashboard no tiene autenticación).
    DASHBOARD_HOST = _env("DASHBOARD_HOST", "127.0.0.1")
    DASHBOARD_PORT = _env("DASHBOARD_PORT", 5000)
    # Si no se define, se genera una clave aleatoria y se guarda en DATA_DIR/dashboard_password.txt
    DASHBOARD_PASSWORD = _env("DASHBOARD_PASSWORD", "")

    # Gestos de seguridad (mantener la pose): ✌️ pausa/reanuda la vigilancia; tres dedos = pánico silencioso.
    PANIC_HOLD_SECONDS = _env("PANIC_HOLD_SECONDS", 2.0)
    ARM_HOLD_SECONDS = _env("ARM_HOLD_SECONDS", 1.5)

    # --- Detector de anomalías ---
    CROWD_THRESHOLD = _env("CROWD_THRESHOLD", 4)            # personas a partir de las cuales hay aglomeración
    LOITER_SECONDS = _env("LOITER_SECONDS", 30.0)           # merodeo: persona quieta en la misma zona
    FALL_CONFIRM_SECONDS = _env("FALL_CONFIRM_SECONDS", 1.5)  # tiempo tumbado para confirmar caída
    ABANDON_SECONDS = _env("ABANDON_SECONDS", 20.0)         # objeto sin dueño (mochila, maleta...)
    NIGHT_START = _env("NIGHT_START", 23)                   # hora de inicio del horario "no esperado"
    NIGHT_END = _env("NIGHT_END", 6)                        # NIGHT_START == NIGHT_END lo desactiva
    VLM_TTL = _env("VLM_TTL", 30.0)                         # segundos de validez del último análisis del VLM
    REALERT_SECONDS = _env("REALERT_SECONDS", 60.0)         # reavisar si una anomalía persiste
    ACK_SECONDS = _env("ACK_SECONDS", 30.0)                 # silencio tras reconocer una alerta (👍)
    EVENT_COOLDOWN = _env("EVENT_COOLDOWN", 10.0)           # entre eventos informativos registrados

    WARMUP_SECONDS = _env("WARMUP_SECONDS", 4.0)             # arranque: sin alertas de movimiento

    # --- Avisos ---
    SOUND_ENABLED = _env("SOUND_ENABLED", 1)
    TELEGRAM_TOKEN = _env("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID", "")
