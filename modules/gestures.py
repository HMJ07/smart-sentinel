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
        self.timeout_hold_start = None
        self.trigger_exit = False

    def _dist(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def process(self, frame, timestamp_ms):
        h, w, _ = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        is_timeout_gesture = False

        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
            h1 = detection_result.hand_landmarks[0]
            h2 = detection_result.hand_landmarks[1]

            # Puntos clave: Muñeca (0), Nudillos (5, 17) y Puntales
            wrist1, wrist2 = h1[0], h2[0]
            idx_tip1, idx_tip2 = h1[8], h2[8]

            # Evaluar cuál mano está arriba (Horizontal) y cuál abajo (Vertical)
            # Calculamos inclinación del eje Wrist -> Middle Finger Tip (12)
            dir1_y = abs(h1[12].y - h1[0].y)
            dir1_x = abs(h1[12].x - h1[0].x)
            
            dir2_y = abs(h2[12].y - h2[0].y)
            dir2_x = abs(h2[12].x - h2[0].x)

            # Orientación: Es horizontal si desplaza más en X que en Y
            h1_isHorizontal = dir1_x > dir1_y
            h2_isHorizontal = dir2_x > dir2_y

            # Distancia entre la mano superior e inferior
            dist_hands = self._dist(wrist1, wrist2)

            # Condición de Gesto T: Una horizontal, otra vertical y cerca
            if ((h1_isHorizontal and not h2_isHorizontal) or (h2_isHorizontal and not h1_isHorizontal)) and dist_hands < 0.35:
                is_timeout_gesture = True

        if detection_result.hand_landmarks:
            for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                
                skel_color = (0, 0, 255) if is_timeout_gesture else (0, 255, 180)

                for start_idx, end_idx in self.connections:
                    cv2.line(frame, coords[start_idx], coords[end_idx], skel_color, 1, cv2.LINE_AA)
                for pt in coords:
                    cv2.circle(frame, pt, 2, (255, 255, 255), -1)

                wrist = landmarks[0]
                fingers = []

                d_tip_thumb = self._dist(landmarks[4], landmarks[17])
                d_knuckle_thumb = self._dist(landmarks[2], landmarks[17])
                fingers.append(1 if d_tip_thumb > d_knuckle_thumb * 1.15 else 0)

                tips = [8, 12, 16, 20]
                pip_joints = [6, 10, 14, 18]
                for tip, pip in zip(tips, pip_joints):
                    fingers.append(1 if self._dist(landmarks[tip], wrist) > self._dist(landmarks[pip], wrist) * 1.08 else 0)

                total_fingers = sum(fingers)

                if hand_idx == 0 and not is_timeout_gesture:
                    raw_idx_x, raw_idx_y = coords[8]

                    if not self.is_drawing_active:
                        self.smooth_x, self.smooth_y = raw_idx_x, raw_idx_y
                    else:
                        self.smooth_x = int(self.alpha * raw_idx_x + (1 - self.alpha) * self.smooth_x)
                        self.smooth_y = int(self.alpha * raw_idx_y + (1 - self.alpha) * self.smooth_y)

                    index_up = fingers[1] == 1
                    others_down = (fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0)

                    if index_up and others_down:
                        if self.is_drawing_active:
                            cv2.line(self.canvas, (self.smooth_x_prev, self.smooth_y_prev),
                                     (self.smooth_x, self.smooth_y), (0, 255, 120), 5, cv2.LINE_AA)
                        self.is_drawing_active = True
                        self.smooth_x_prev, self.smooth_y_prev = self.smooth_x, self.smooth_y
                        cv2.circle(frame, (self.smooth_x, self.smooth_y), 6, (0, 255, 120), -1)

                    elif total_fingers == 5:
                        self.canvas = np.zeros_like(frame)
                        self.is_drawing_active = False
                    else:
                        self.is_drawing_active = False

                cv2.putText(frame, f"Mano {hand_idx+1}: {total_fingers} dedos", (20, 35 + (hand_idx * 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        # Lógica y Feedback Visual del Gesto TIME-OUT
        if is_timeout_gesture:
            if self.timeout_hold_start is None:
                self.timeout_hold_start = time.time()
            else:
                elapsed = time.time() - self.timeout_hold_start
                remaining = max(0.0, 0.8 - elapsed)
                
                cv2.putText(frame, "GESTO TIME-OUT DETECTADO", (int(w/2) - 180, int(h/2) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"CERRANDO EN {remaining:.1f}s", (int(w/2) - 100, int(h/2) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                if elapsed >= 0.8:
                    self.trigger_exit = True
        else:
            self.timeout_hold_start = None

        frame = cv2.addWeighted(frame, 1.0, self.canvas, 0.9, 0)
        return frame
