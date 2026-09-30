import os
import sys

from config.settings import Config

MAX_LOG_BYTES = 2_000_000


def setup_logging():
    """Sin consola (app empaquetada) print() no llega a ningún sitio y en Windows incluso falla
    si sys.stdout es None: se redirige todo a DATA_DIR/sentinel.log."""
    if not (Config.IS_FROZEN or sys.stdout is None or sys.stderr is None):
        return None
    try:
        if os.path.exists(Config.LOG_PATH) and os.path.getsize(Config.LOG_PATH) > MAX_LOG_BYTES:
            os.replace(Config.LOG_PATH, Config.LOG_PATH + ".1")
        log = open(Config.LOG_PATH, "a", buffering=1, encoding="utf-8")
    except OSError:
        return None
    sys.stdout = sys.stderr = log
    return Config.LOG_PATH


def show_error(title, message):
    """Diálogo nativo de error. Una app de doble clic sin consola no puede 'imprimir' el problema:
    sin esto se cerraría sin decir nada. Solo en la app empaquetada (en desarrollo basta la consola)."""
    print(f"❌ {message}")
    if not Config.IS_FROZEN:
        return
    try:
        if sys.platform == "darwin":
            import json
            import subprocess
            script = (f'display dialog {json.dumps(message)} with title {json.dumps(title)} '
                      'buttons {"OK"} default button "OK" with icon stop')
            subprocess.run(["osascript", "-e", script], timeout=300)
        elif sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        pass
