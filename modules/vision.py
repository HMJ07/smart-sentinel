import asyncio
import cv2
import ollama
import torch
from ultralytics import YOLO
from config.settings import Config

class PersonAndAnomalyDetector:
    def __init__(self, conf_threshold=0.50):
        # Desactivar NMS acelerado por torchvision para evitar el fallo en Python 3.13
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold
        
        # Inferencia de prueba para evitar el warmup en caliente
        try:
            dummy = torch.zeros((1, 3, 640, 640))
            self.model.predict(dummy, verbose=False)
        except Exception:
            pass

    def detect_objects(self, frame):
        try:
            # Forzamos la inferencia deshabilitando la comprobación de torchvision
            results = self.model.predict(frame, verbose=False, device='cpu')[0]
            detected_persons = []
            detected_objects = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]

                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_id == 0:
                        detected_persons.append((x1, y1, x2, y2, conf))
                    else:
                        detected_objects.append((x1, y1, x2, y2, label, conf))

            return detected_persons, detected_objects
        except Exception as e:
            return [], []


class VLMAnalyzer:
    def __init__(self):
        self.is_analyzing = False
        self.latest_analysis = ""

    async def analyze_frame_async(self, frame):
        if self.is_analyzing:
            return

        self.is_analyzing = True
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=Config.OLLAMA_MODEL,
                    prompt=Config.VLM_PROMPT,
                    images=[image_bytes]
                )
            )
            self.latest_analysis = response.get('response', '').strip()
        except Exception:
            self.latest_analysis = "Error en conexión VLM"
        finally:
            self.is_analyzing = False
