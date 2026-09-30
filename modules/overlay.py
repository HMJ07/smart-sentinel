import math
import time
import unicodedata

import cv2

from core.anomaly_detector import CRITICAL, WARNING

COLORS = {WARNING: (0, 190, 255), CRITICAL: (40, 40, 240)}   # BGR
ACK_COLOR = (150, 150, 150)


def _ascii(text):
    # cv2.putText solo dibuja ASCII: quita acentos y símbolos.
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if ord(c) < 128)


def draw_alert(frame, assessment):
    """Banner superior + marco parpadeante según la severidad. Devuelve el frame."""
    if assessment.level == 0:
        return frame
    h, w = frame.shape[:2]
    color = ACK_COLOR if assessment.acknowledged else COLORS[assessment.level]
    title = {CRITICAL: "ALERTA CRITICA", WARNING: "ADVERTENCIA"}[assessment.level]
    if assessment.acknowledged:
        title += " (reconocida)"

    lines = [_ascii(a.message)[:110] for a in assessment.anomalies[:3]]
    banner_h = 40 + 24 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (20, 58 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if not assessment.acknowledged:
        pulse = 0.5 + 0.5 * math.sin(time.time() * (8 if assessment.level == CRITICAL else 4))
        thickness = 2 + int(6 * pulse)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
        cv2.putText(frame, "Pulgar arriba = reconocer", (w - 260, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
