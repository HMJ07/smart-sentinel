import subprocess
import sys
import threading
import time

import cv2
import requests

from config.settings import Config
from core.anomaly_detector import CRITICAL

_MAC_SOUNDS = {CRITICAL: "/System/Library/Sounds/Sosumi.aiff", 1: "/System/Library/Sounds/Ping.aiff"}


def _play(level):
    """Sonido no bloqueante y sin dependencias extra en macOS, Windows y Linux."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", _MAC_SOUNDS.get(level, _MAC_SOUNDS[1])],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONHAND if level == CRITICAL else winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


class AlertNotifier:
    """Sonido repetido mientras haya una alerta sin reconocer + aviso por Telegram (opcional)."""

    SOUND_REPEAT = {CRITICAL: 2.5, 1: 6.0}

    def __init__(self):
        self._last_sound = 0.0
        self.telegram_enabled = bool(Config.TELEGRAM_TOKEN and Config.TELEGRAM_CHAT_ID)

    def update(self, assessment, frame):
        if assessment.level > 0 and not assessment.acknowledged and Config.SOUND_ENABLED:
            now = time.time()
            if now - self._last_sound >= self.SOUND_REPEAT.get(assessment.level, 6.0):
                self._last_sound = now
                _play(assessment.level)

        critical_new = [a for a in assessment.new if a.severity == CRITICAL]
        if critical_new and self.telegram_enabled:
            text = "🚨 Smart Sentinel\n" + "\n".join(a.message for a in critical_new)
            ok, buf = cv2.imencode(".jpg", frame)
            threading.Thread(target=self._send_telegram, args=(text, buf.tobytes() if ok else None),
                             daemon=True).start()

    @staticmethod
    def _send_telegram(text, jpeg):
        base = f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}"
        try:
            if jpeg:
                requests.post(f"{base}/sendPhoto", data={"chat_id": Config.TELEGRAM_CHAT_ID, "caption": text},
                              files={"photo": ("alerta.jpg", jpeg)}, timeout=15)
            else:
                requests.post(f"{base}/sendMessage", data={"chat_id": Config.TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        except requests.RequestException:
            pass  # sin red no debe tumbar la vigilancia
