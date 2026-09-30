import sys
import cv2
from config.settings import Config


def _preferred_backend():
    # Backends nativos: arrancan mucho más rápido que el autodetectado en Windows/macOS.
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class Camera:
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

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.flip(frame, 1)

    def release(self):
        self.cap.release()
