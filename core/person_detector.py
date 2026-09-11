import cv2
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5):
        # Descarga automáticamente la versión nano de YOLOv8 la primera vez
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        person_boxes = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Clase 0 en COCO dataset es 'person'
            if cls_id == 0 and conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2, conf))

        return len(person_boxes) > 0, person_boxes
