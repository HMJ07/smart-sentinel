import cv2
from config.settings import Config

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, cv2.flip(frame, 1)

    def release(self):
        self.cap.release()
