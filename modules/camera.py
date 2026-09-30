import sys
import threading

import cv2
from config.settings import Config

READ_TIMEOUT = 0.5   # s esperando un fotograma nuevo antes de dar la cámara por parada


def _preferred_backend():
    # Backends nativos: arrancan mucho más rápido que el autodetectado en Windows/macOS.
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class Camera:
    """Un hilo lee la cámara sin parar y deja solo el último fotograma: así nunca se procesa una imagen vieja
    acumulada en el búfer del controlador (en Windows/DirectShow llegan a ser varios fotogramas de retraso)."""

    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX, _preferred_backend())
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara {Config.CAMERA_INDEX}. En macOS revisa "
                "Ajustes > Privacidad > Cámara; prueba SENTINEL_CAMERA_INDEX=1 si tienes varias."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cond = threading.Condition()
        self._frame, self._seq, self._seen = None, 0, 0
        self._running = True
        self._thread = threading.Thread(target=self._grab, name="camera-reader", daemon=True)
        self._thread.start()

    def _grab(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def read(self):
        """Espera al siguiente fotograma nuevo (o hasta READ_TIMEOUT) y lo devuelve espejado."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._seen, timeout=READ_TIMEOUT):
                return False, None
            self._seen = self._seq
            return True, self._frame

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()
