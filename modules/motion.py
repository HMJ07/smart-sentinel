import cv2

ANALYSIS_WIDTH = 320   # el fondo se modela en baja resolución: ~5x menos coste y, de paso, menos ruido


class MotionDetector:
    def __init__(self, min_area=2500):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area   # en píxeles del fotograma original
        self.last_ratio = 0.0      # fracción real (0-1) de píxeles en movimiento

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, ANALYSIS_WIDTH / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        fg_mask = self.fgbg.apply(small)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        self.last_ratio = cv2.countNonZero(thresh) / thresh.size
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        bboxes = []
        min_area = self.min_area * scale * scale
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                motion_detected = True
                x, y, bw, bh = cv2.boundingRect(contour)
                bboxes.append((int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)))

        return motion_detected, bboxes
