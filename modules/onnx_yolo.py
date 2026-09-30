"""YOLOv8 sobre ONNX Runtime, sin PyTorch ni ultralytics en tiempo de ejecución.

Un único modelo (.onnx) sirve para todo: ONNX Runtime elige el mejor acelerador disponible
(CoreML / Neural Engine en macOS, CUDA o DirectML en Windows) y cae a CPU si algo falla.
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def available_providers(preference="auto"):
    """Lista ordenada de proveedores de ONNX Runtime a intentar (siempre termina en CPU)."""
    have = ort.get_available_providers()
    if preference == "cpu":
        return ["CPUExecutionProvider"]
    if sys.platform == "darwin":
        order = ["CoreMLExecutionProvider"]
    else:
        order = ["CUDAExecutionProvider", "DmlExecutionProvider"]
    return [p for p in order if p in have] + ["CPUExecutionProvider"]


class OnnxYolo:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45, provider_preference="auto"):
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.provider = None
        self.session = None

        last_error = None
        for provider in available_providers(provider_preference):
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                if provider == "DmlExecutionProvider":     # requisitos de DirectML
                    opts.enable_mem_pattern = False
                    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
                self._input = session.get_inputs()[0].name
                size = session.get_inputs()[0].shape[2]
                self.size = size if isinstance(size, int) else 640
                self.session = session
                self._run(np.zeros((self.size, self.size, 3), np.uint8))   # calentamiento + validación
                self.provider = provider
                break
            except Exception as e:      # el acelerador puede no soportar el modelo: probar el siguiente
                last_error = e
                self.session = None
        if self.session is None:
            raise RuntimeError(f"No se pudo cargar {model_path}: {last_error}")

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.size / h, self.size / w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, scale, left, top

    def _run(self, frame):
        canvas, scale, left, top = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)   # BGR->RGB, NCHW, float32
        out = self.session.run(None, {self._input: blob})[0]
        return out, scale, left, top

    def detect(self, frame):
        """-> [(x1, y1, x2, y2, class_id, conf)] en coordenadas del fotograma original."""
        out, scale, left, top = self._run(frame)
        preds = out[0].T                                   # (8400, 84)
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(len(preds)), class_ids]
        keep = confs >= self.conf
        if not keep.any():
            return []
        preds, class_ids, confs = preds[keep], class_ids[keep], confs[keep]

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - w / 2 - left) / scale
        y1 = (cy - h / 2 - top) / scale
        boxes = np.stack([x1, y1, w / scale, h / scale], axis=1)

        # NMS por clase: se desplaza cada clase para que no compitan entre sí
        offset = class_ids[:, None] * 10000.0
        idx = cv2.dnn.NMSBoxes((boxes + np.hstack([offset, offset, 0 * offset, 0 * offset])).tolist(),
                               confs.tolist(), self.conf, self.iou)
        fh, fw = frame.shape[:2]
        results = []
        for i in np.array(idx).flatten():
            bx, by, bw, bh = boxes[i]
            results.append((int(max(0, bx)), int(max(0, by)), int(min(fw, bx + bw)), int(min(fh, by + bh)),
                            int(class_ids[i]), float(confs[i])))
        return results
