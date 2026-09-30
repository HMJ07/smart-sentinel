"""Exporta yolov8n.pt a yolov8n.onnx (solo necesario en desarrollo; requiere `pip install ultralytics onnx`)."""
from pathlib import Path

from ultralytics import YOLO

root = Path(__file__).resolve().parent.parent
print(YOLO(str(root / "yolov8n.pt")).export(format="onnx", imgsz=640, simplify=True, opset=12))
