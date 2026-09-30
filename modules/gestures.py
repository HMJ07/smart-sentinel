import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import Config
from modules.smoothing import OneEuroFilter

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# Un puño de menos de esto no muestra el anillo (evita ruido visual al bajar el dedo de dibujo).
RING_SHOW_AFTER = 0.35
# Pulgar arriba sostenido para reconocer una alerta, y pausa antes de poder repetirlo.
ACK_HOLD = 0.6
ACK_COOLDOWN = 2.0
# Si la mano se pierde un instante (parpadeo del tracker) no se reinicia la cuenta.
FIST_GRACE = 0.25


class HandTracker:
    """Seguimiento de manos: dibujo con el índice, palma abierta para borrar y
    puño cerrado sostenido para salir (sin teclado, igual en macOS y Windows)."""

    def __init__(self, max_hands=2, detection_confidence=0.55):
        model_path = Config.HAND_MODEL_PATH
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(MODEL_URL, model_path)

        options = vision.HandLandmarkerOptions(
            # CPU explícito: MediaPipe >= 1.0 intenta iniciar Metal/GPU por defecto y aborta el proceso.
            base_options=python.BaseOptions(model_asset_path=model_path,
                                            delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.canvas = None

        self.smooth_x = self.smooth_y = 0
        self.smooth_x_prev = self.smooth_y_prev = 0
        # One-Euro: sin temblor con la mano quieta y casi sin retraso al moverla (el EMA fijo hacía lo uno o lo otro).
        self.fx, self.fy = OneEuroFilter(1.0, 0.012), OneEuroFilter(1.0, 0.012)
        self.is_drawing_active = False

        self.fist_start = None
        self.fist_last_seen = 0.0
        self.trigger_exit = False

        self.thumb_start = None
        self.last_ack = 0.0
        self.ack_requested = False

        # Gestos de seguridad: cada uno es (inicio de la pose, último disparo, petición pendiente)
        self._holds = {"panic": [None, float("-inf")], "arm": [None, float("-inf")]}
        self._pending = {"panic": False, "arm": False}

    def consume_ack(self):
        """True una sola vez cuando el usuario ha hecho 👍 sostenido."""
        requested, self.ack_requested = self.ack_requested, False
        return requested

    def consume_panic(self):
        """True una sola vez tras mantener tres dedos: alerta silenciosa (sin sonido ni aviso en pantalla)."""
        requested, self._pending["panic"] = self._pending["panic"], False
        return requested

    def consume_arm_toggle(self):
        """True una sola vez tras mantener ✌️: pausar/reanudar la vigilancia."""
        requested, self._pending["arm"] = self._pending["arm"], False
        return requested

    def _hold(self, name, active, now, seconds, cooldown):
        """Pose continua durante `seconds` -> marca la petición (un parpadeo reinicia la cuenta)."""
        start, last_fire = self._holds[name]
        if not active:
            self._holds[name][0] = None
        elif now - last_fire >= cooldown:
            start = now if start is None else start
            self._holds[name][0] = start
            if now - start >= seconds:
                self._pending[name] = True
                self._holds[name] = [None, now]

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _finger_states(self, lm):
        """[pulgar, índice, corazón, anular, meñique] -> 1 si está extendido."""
        wrist = lm[0]
        thumb = self._dist(lm[4], lm[17]) > self._dist(lm[2], lm[17]) * 1.15
        states = [1 if thumb else 0]
        for tip, pip in zip((8, 12, 16, 20), (6, 10, 14, 18)):
            states.append(1 if self._dist(lm[tip], wrist) > self._dist(lm[pip], wrist) * 1.08 else 0)
        return states

    def _is_thumbs_up(self, lm, fingers):
        """Pulgar hacia arriba con el resto de dedos plegados (la mano no puede estar boca abajo)."""
        if any(fingers[1:]):
            return False
        hand_size = self._dist(lm[0], lm[9])
        return lm[4].y < lm[3].y < lm[2].y and (lm[0].y - lm[4].y) > 0.9 * hand_size

    def _draw_exit_ring(self, frame, lm, progress, w, h):
        cx, cy = int(lm[9].x * w), int(lm[9].y * h)
        radius = max(30, int(self._dist(lm[0], lm[9]) * max(w, h) * 1.1))
        cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 4, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, int(360 * progress),
                    (0, 200, 255), 6, cv2.LINE_AA)
        remaining = max(0.0, Config.EXIT_HOLD_SECONDS * (1 - progress))
        cv2.putText(frame, f"SALIENDO {remaining:.1f}s", (cx - 70, cy - radius - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "abre la mano para cancelar", (cx - 110, cy + radius + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def process(self, frame, timestamp_ms):
        h, w, _ = frame.shape
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        fist_landmarks = None
        thumbs_up_seen = False
        panic_pose = arm_pose = False

        for hand_idx, lm in enumerate(result.hand_landmarks or []):
            coords = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(frame, coords[a], coords[b], (0, 255, 180), 1, cv2.LINE_AA)
            for pt in coords:
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)

            fingers = self._finger_states(lm)
            total_fingers = sum(fingers)
            # Puño: índice, corazón, anular y meñique plegados (el pulgar es poco fiable),
            # pero un pulgar arriba es otro gesto (reconocer alerta), no una salida.
            thumbs_up = self._is_thumbs_up(lm, fingers)
            thumbs_up_seen = thumbs_up_seen or thumbs_up
            is_fist = sum(fingers[1:]) == 0 and not thumbs_up
            # Tres dedos (índice, corazón, anular; meñique plegado) = pánico. ✌️ (índice y corazón) = pausa.
            # El pulgar se ignora en ambos: es el dedo menos fiable de detectar.
            panic_pose = panic_pose or fingers[1:] == [1, 1, 1, 0]
            arm_pose = arm_pose or fingers[1:] == [1, 1, 0, 0]
            if is_fist and fist_landmarks is None:
                fist_landmarks = lm

            if hand_idx == 0:
                raw_x, raw_y = coords[8]
                if not self.is_drawing_active:
                    self.fx.reset()
                    self.fy.reset()
                self.smooth_x, self.smooth_y = int(self.fx(raw_x, now)), int(self.fy(raw_y, now))

                index_only = fingers[1] == 1 and not any(fingers[2:])
                if index_only:
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

            cv2.putText(frame, f"Mano {hand_idx + 1}: {total_fingers} dedos",
                        (20, 35 + hand_idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 200), 1, cv2.LINE_AA)

        # --- Salida por puño sostenido ---
        if fist_landmarks is not None:
            self.fist_last_seen = now
            if self.fist_start is None:
                self.fist_start = now
            held = now - self.fist_start
            if held >= RING_SHOW_AFTER:
                self._draw_exit_ring(frame, fist_landmarks,
                                     min(1.0, held / Config.EXIT_HOLD_SECONDS), w, h)
            if held >= Config.EXIT_HOLD_SECONDS:
                self.trigger_exit = True
        elif self.fist_start is not None and now - self.fist_last_seen > FIST_GRACE:
            self.fist_start = None

        # --- Gestos de seguridad (sin ningún aviso visual: pensados para no delatarse) ---
        self._hold("panic", panic_pose, now, Config.PANIC_HOLD_SECONDS, cooldown=30.0)
        self._hold("arm", arm_pose, now, Config.ARM_HOLD_SECONDS, cooldown=3.0)

        # --- Reconocer alerta con 👍 sostenido ---
        if thumbs_up_seen and now - self.last_ack > ACK_COOLDOWN:
            self.thumb_start = self.thumb_start or now
            if now - self.thumb_start >= ACK_HOLD:
                self.ack_requested = True
                self.last_ack = now
                self.thumb_start = None
        elif not thumbs_up_seen:
            self.thumb_start = None

        cv2.putText(frame, f"Puno cerrado {Config.EXIT_HOLD_SECONDS:g}s = salir",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)

        return cv2.addWeighted(frame, 1.0, self.canvas, 0.9, 0)
