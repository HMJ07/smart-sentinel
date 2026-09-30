import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


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
    DB_PATH = str(ROOT_DIR / "events.db")
    CAPTURES_DIR = str(ROOT_DIR / "captures")
    HAND_MODEL_PATH = str(ROOT_DIR / "hand_landmarker.task")

    CAMERA_INDEX = _env("CAMERA_INDEX", 0)
    FRAME_WIDTH = _env("FRAME_WIDTH", 1280)
    FRAME_HEIGHT = _env("FRAME_HEIGHT", 720)
    FPS = _env("FPS", 30)

    MOTION_THRESHOLD = 25
    MIN_CONTOUR_AREA = 2500

    # YOLO es lo más costoso: se ejecuta cada N fotogramas y se reutiliza el resultado.
    DETECT_EVERY_N_FRAMES = _env("DETECT_EVERY_N_FRAMES", 3)

    OLLAMA_MODEL = _env("OLLAMA_MODEL", "llava")
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
