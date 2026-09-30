import asyncio
import os
import time

import cv2
import ollama

from config.settings import Config
from modules.onnx_yolo import COCO_NAMES, OnnxYolo

VLM_RETRY_SECONDS = 30.0   # tras un fallo, no insistir contra Ollama en cada ciclo


def _ensure_onnx_model(path):
    """El instalador ya trae el .onnx. En desarrollo, se exporta una vez desde el .pt si hace falta."""
    if os.path.exists(path):
        return
    pt = os.path.join(os.path.dirname(path), "yolov8n.pt")
    try:
        from ultralytics import YOLO   # solo necesario para exportar, nunca en el instalador
        print("⏳ Exportando yolov8n.pt a ONNX (solo la primera vez)...")
        YOLO(pt).export(format="onnx", imgsz=640, simplify=True, opset=12)
    except Exception as e:
        raise RuntimeError(f"Falta el modelo {path}. Genera uno con: python scripts/export_model.py ({e})")


class PersonAndAnomalyDetector:
    def __init__(self, conf_threshold=0.50):
        _ensure_onnx_model(Config.YOLO_MODEL_PATH)
        self.model = OnnxYolo(Config.YOLO_MODEL_PATH, conf_threshold, provider_preference=Config.YOLO_PROVIDER)
        self.provider = self.model.provider
        self._warned = False

    def detect_objects(self, frame):
        try:
            persons, objects = [], []
            for x1, y1, x2, y2, cls_id, conf in self.model.detect(frame):
                if cls_id == 0:
                    persons.append((x1, y1, x2, y2, conf))
                else:
                    objects.append((x1, y1, x2, y2, COCO_NAMES[cls_id], conf))
            return persons, objects
        except Exception as e:
            if not self._warned:  # avisar una vez, no silenciar el fallo para siempre
                print(f"⚠️ Fallo en la detección YOLO: {e}")
                self._warned = True
            return [], []


class VLMAnalyzer:
    def __init__(self):
        self.is_analyzing = False
        self.latest_analysis = ""
        self.updated_at = None
        self._last_error = None
        self._retry_at = 0.0
        self._client = ollama.Client(host=Config.OLLAMA_HOST)

    async def analyze_frame_async(self, frame):
        if self.is_analyzing or not Config.VLM_ENABLED or time.time() < self._retry_at:
            return

        self.is_analyzing = True
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._client.generate(
                    model=Config.OLLAMA_MODEL,
                    prompt=Config.VLM_PROMPT,
                    images=[image_bytes],
                )
            )
            self.latest_analysis = response.get('response', '').strip()
            self.updated_at = time.time()
            self._last_error = None
        except Exception as e:
            if str(e) != self._last_error:  # mostrar la causa real, una vez por tipo de error
                self._last_error = str(e)
                print(f"⚠️ Análisis visual no disponible ({Config.OLLAMA_MODEL}): {e}. "
                      "El resto de funciones sigue activo.")
            self.latest_analysis = ""
            self.updated_at = None   # un error nunca cuenta como análisis válido para las anomalías
            self._retry_at = time.time() + VLM_RETRY_SECONDS
        finally:
            self.is_analyzing = False
