import cv2

class MotionDetector:
    def __init__(self, min_area=2500):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area
        self.last_ratio = 0.0  # fracción real (0-1) de píxeles en movimiento

    def detect(self, frame):
        fg_mask = self.fgbg.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        self.last_ratio = cv2.countNonZero(thresh) / thresh.size
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        bboxes = []

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))

        return motion_detected, bboxes
