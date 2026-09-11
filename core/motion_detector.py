import cv2

class MotionDetector:
    def __init__(self, min_area=2500):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = min_area

    def detect(self, frame):
        fg_mask = self.fgbg.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        bounding_boxes = []

        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, w, h))

        return motion_detected, bounding_boxes
