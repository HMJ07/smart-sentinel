import cv2
import numpy as np
import urllib.request
import os
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.55):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.canvas = None
        
        self.smooth_x, self.smooth_y = 0, 0
        self.alpha = 0.40
        self.is_drawing_active = False

        self.connections = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (0,17), (17,18), (18,19), (19,20)
        ]
        self.exit_button_hover_start = None
        self.trigger_exit = False

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process(self, frame, timestamp_ms):
        h, w, _ = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        # 1. BOTÓN "SALIR" MÁS ACCESIBLE (Esquina inferior derecha, área grande)
        btn_w, btn_h = 160, 55
        btn_x1, btn_y1 = w - btn_w - 20, h - btn_h - 20
        btn_x2, btn_y2 = w - 20, h - 20
        
        # Renderizado estilo Glassmorphism HUD
        sub_img = frame[btn_y1:btn_y2, btn_x1:btn_x2]
        glass_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        glass_rect[:] = (20, 20, 35)
        res = cv2.addWeighted(sub_img, 0.4, glass_rect, 0.6, 0)
        frame[btn_y1:btn_y2, btn_x1:btn_x2] = res
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 220, 255), 1)
        cv2.putText(frame, "SALIR [X]", (btn_x1 + 32, btn_y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        # 2. Detección de Manos
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        finger_in_exit_btn = False

        if detection_result.hand_landmarks:
            for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                
                # Renderizado limpio y sutil del esqueleto
                for start_idx, end_idx in self.connections:
                    cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 180), 1, cv2.LINE_AA)
                for pt in coords:
                    cv2.circle(frame, pt, 2, (255, 255, 255), -1)

                # ALGORITMO VECTORIAL DE CONTEO DE DEDOS
                wrist = landmarks[0]
                fingers = []

                # Pulgar: distancia norma vectorial
                d_tip_thumb = self._dist(landmarks[4], landmarks[17])
                d_knuckle_thumb = self._dist(landmarks[2], landmarks[17])
                fingers.append(1 if d_tip_thumb > d_knuckle_thumb * 1.15 else 0)

                # Resto de 4 dedos: altura vectorial relativa a muñeca
                tips = [8, 12, 16, 20]
                pip_joints = [6, 10, 14, 18]
                for tip, pip in zip(tips, pip_joints):
                    fingers.append(1 if self._dist(landmarks[tip], wrist) > self._dist(landmarks[pip], wrist) * 1.08 else 0)

                total_fingers = sum(fingers)

                # Detección de gestos para la mano de interacción (Mano 1)
                if hand_idx == 0:
                    raw_idx_x, raw_idx_y = coords[8]

                    if not self.is_drawing_active:
                        self.smooth_x, self.smooth_y = raw_idx_x, raw_idx_y
                    else:
                        self.smooth_x = int(self.alpha * raw_idx_x + (1 - self.alpha) * self.smooth_x)
                        self.smooth_y = int(self.alpha * raw_idx_y + (1 - self.alpha) * self.smooth_y)

                    # Colisión con el botón flotante
                    if btn_x1 <= self.smooth_x <= btn_x2 and btn_y1 <= self.smooth_y <= btn_y2:
                        finger_in_exit_btn = True
                        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 0), 2)

                    index_up = fingers[1] == 1
                    others_down = (fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)

                    # Gesto DIBUJAR: Índice levantado
                    if index_up and others_down:
                        if self.is_drawing_active:
                            cv2.line(self.canvas, (self.smooth_x_prev, self.smooth_y_prev),
                                     (self.smooth_x, self.smooth_y), (0, 255, 120), 5, cv2.LINE_AA)
                        self.is_drawing_active = True
                        self.smooth_x_prev, self.smooth_y_prev = self.smooth_x, self.smooth_y
                        
                        # Retícula de precisión
                        cv2.circle(frame, (self.smooth_x, self.smooth_y), 6, (0, 255, 120), -1)
                        cv2.circle(frame, (self.smooth_x, self.smooth_y), 12, (255, 255, 255), 1)

                    # Gesto BORRAR: Mano abierta (5 dedos)
                    elif total_fingers == 5:
                        self.canvas = np.zeros_like(frame)
                        self.is_drawing_active = False
                    else:
                        self.is_drawing_active = False

                # Indicador discreto del recuento por mano en HUD
                cv2.putText(frame, f"Mano {hand_idx+1}: {total_fingers} dedos", (20, 35 + (hand_idx * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Feedback visual del botón al mantener pulsado
        if finger_in_exit_btn:
            if self.exit_button_hover_start is None:
                self.exit_button_hover_start = time.time()
            else:
                elapsed = time.time() - self.exit_button_hover_start
                progress_w = int((elapsed / 0.6) * btn_w)
                cv2.line(frame, (btn_x1, btn_y2 - 2), (btn_x1 + min(progress_w, btn_w), btn_y2 - 2), (0, 255, 0), 3)
                if elapsed >= 0.6:  # Cierre rápido a los 0.6s
                    self.trigger_exit = True
        else:
            self.exit_button_hover_start = None

        frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.9, 0)
        return frame
