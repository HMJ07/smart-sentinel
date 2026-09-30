import asyncio
import time
import cv2
import ollama
from ultralytics import YOLO
from config.settings import Config


def _pick_device():
    # CUDA en Windows/Linux con NVIDIA; CPU en el resto (MPS da problemas con YOLO/NMS en algunas versiones).
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class PersonAndAnomalyDetector:
    def __init__(self, conf_threshold=0.50):
        self.model = YOLO(str(Config.ROOT_DIR / "yolov8n.pt"))
        self.conf_threshold = conf_threshold
        self.device = _pick_device()
        self._warned = False

    def detect_objects(self, frame):
        try:
            results = self.model.predict(frame, verbose=False, device=self.device,
                                         conf=self.conf_threshold)[0]
            persons, objects = [], []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == 0:
                    persons.append((x1, y1, x2, y2, conf))
                else:
                    objects.append((x1, y1, x2, y2, self.model.names[cls_id], conf))
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
        self._client = ollama.Client(host=Config.OLLAMA_HOST)

    async def analyze_frame_async(self, frame):
        if self.is_analyzing:
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
        except Exception:
            self.latest_analysis = "Error en conexión VLM (¿Ollama está en marcha?)"
            self.updated_at = None  # un error nunca cuenta como análisis válido para las anomalías
        finally:
            self.is_analyzing = False
