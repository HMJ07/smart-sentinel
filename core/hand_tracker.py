import cv2
import numpy as np
import urllib.request
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.6):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Descargando modelo de MediaPipe Hands...")
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
        self.alpha = 0.35
        self.is_drawing_active = False

        self.finger_tips = [4, 8, 12, 16, 20]
        # Conexiones principales para dibujar el esqueleto
        self.connections = [
            (0,1), (1,2), (2,3), (3,4),       # Pulgar
            (0,5), (5,6), (6,7), (7,8),       # Índice
            (5,9), (9,10), (10,11), (11,12),  # Corazón
            (9,13), (13,14), (14,15), (15,16),# Anular
            (13,17), (0,17), (17,18), (18,19), (19,20) # Meñique
        ]
        self.exit_button_hover_start = None
        self.trigger_exit = False

    def process(self, frame, timestamp_ms):
        h, w, _ = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        # 1. UI: Botón de Salida discreto
        btn_x1, btn_y1 = w - 120, 20
        btn_x2, btn_y2 = w - 20, 60
        
        sub_img = frame[btn_y1:btn_y2, btn_x1:btn_x2]
        black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 40
        res = cv2.addWeighted(sub_img, 0.4, black_rect, 0.6, 0)
        frame[btn_y1:btn_y2, btn_x1:btn_x2] = res
        cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 1)
        cv2.putText(frame, "SALIR", (btn_x1 + 25, btn_y1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Detección en el frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        finger_in_exit_btn = False

        if detection_result.hand_landmarks:
            landmarks = detection_result.hand_landmarks[0]
            handedness = detection_result.handedness[0][0].category_name

            # Dibujar esqueleto estético (líneas muy finas y puntos discretos)
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for start_idx, end_idx in self.connections:
                cv2.line(frame, coords[start_idx], coords[end_idx], (0, 255, 200), 1, cv2.LINE_AA)
            for pt in coords:
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)

            # Detección de dedos
            fingers = []
            if handedness == 'Right':
                fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
            else:
                fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

            for tip in self.finger_tips[1:]:
                fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y - 0.015 else 0)

            total_fingers = sum(fingers)

            # Posición suavizada del índice
            raw_idx_x, raw_idx_y = coords[8]

            if not self.is_drawing_active:
                self.smooth_x, self.smooth_y = raw_idx_x, raw_idx_y
            else:
                self.smooth_x = int(self.alpha * raw_idx_x + (1 - self.alpha) * self.smooth_x)
                self.smooth_y = int(self.alpha * raw_idx_y + (1 - self.alpha) * self.smooth_y)

            # Botón salir
            if btn_x1 <= self.smooth_x <= btn_x2 and btn_y1 <= self.smooth_y <= btn_y2:
                finger_in_exit_btn = True
                cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 200), 1)

            index_up = fingers[1] == 1
            others_down = (fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)

            # Estado textual discreto en la esquina superior izquierda
            hand_state_txt = "Dibujando" if (index_up and others_down) else ("Abierta" if total_fingers == 5 else ("Cerrada" if total_fingers == 0 else f"{total_fingers} dedos"))
            cv2.putText(frame, f"Mano: {hand_state_txt}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

            # Modo DIBUJAR: Solo índice arriba
            if index_up and others_down:
                if self.is_drawing_active:
                    cv2.line(self.canvas, (self.smooth_x_prev, self.smooth_y_prev),
                             (self.smooth_x, self.smooth_y), (0, 255, 120), 5, cv2.LINE_AA)
                
                self.is_drawing_active = True
                self.smooth_x_prev, self.smooth_y_prev = self.smooth_x, self.smooth_y
                cv2.circle(frame, (self.smooth_x, self.smooth_y), 6, (0, 255, 120), -1)

            # Modo BORRAR: Mano abierta (5 dedos)
            elif total_fingers == 5:
                self.canvas = np.zeros_like(frame)
                self.is_drawing_active = False

            else:
                self.is_drawing_active = False

        else:
            self.is_drawing_active = False

        # Confirmación botón flotante
        if finger_in_exit_btn:
            if self.exit_button_hover_start is None:
                self.exit_button_hover_start = time.time()
            else:
                elapsed = time.time() - self.exit_button_hover_start
                progress_w = int((elapsed / 0.8) * (btn_x2 - btn_x1))
                cv2.line(frame, (btn_x1, btn_y2), (btn_x1 + min(progress_w, btn_x2 - btn_x1), btn_y2), (0, 255, 200), 2)
                
                if elapsed >= 0.8:
                    self.trigger_exit = True
        else:
            self.exit_button_hover_start = None

        frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.9, 0)
        return frame
