import asyncio
import os
import threading
import time

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2

import dashboard.app as dash_app
from config.settings import Config
from core.alerts import AlertNotifier
from core.anomaly_detector import CRITICAL, LEVEL_NAMES, NORMAL, AnomalyDetector
from core.event_logger import EventLogger
from modules.camera import Camera
from modules.gestures import HandTracker
from modules.motion import MotionDetector
from modules.overlay import draw_alert
from modules.vision import PersonAndAnomalyDetector, VLMAnalyzer


def draw_detections(frame, persons, objects):
    for (x1, y1, x2, y2, conf) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.putText(frame, f"PERSONA {int(conf * 100)}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
    for (x1, y1, x2, y2, label, conf) in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"{label.upper()} {int(conf * 100)}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)


async def main():
    threading.Thread(target=dash_app.run_dashboard, daemon=True).start()

    try:
        camera = Camera()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    motion_detector = MotionDetector(min_area=Config.MIN_CONTOUR_AREA)
    object_detector = PersonAndAnomalyDetector(conf_threshold=0.50)
    hand_tracker = HandTracker(max_hands=2)
    vlm_analyzer = VLMAnalyzer()
    anomaly_detector = AnomalyDetector()
    notifier = AlertNotifier()
    logger = EventLogger()

    start_time = time.time()
    last_info_event = 0.0
    frame_count = 0
    persons, objects = [], []
    assessment = anomaly_detector.evaluate(start_time, (Config.FRAME_HEIGHT, Config.FRAME_WIDTH), [], [])

    window_name = "Smart Sentinel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)

    print("🚀 Smart Sentinel iniciado.")
    print(f"🌐 Dashboard disponible en: http://{Config.DASHBOARD_HOST}:{Config.DASHBOARD_PORT}")
    print(f"✊ Salir sin teclado: puño cerrado {Config.EXIT_HOLD_SECONDS:g}s  |  👍 reconocer alerta")
    if notifier.telegram_enabled:
        print("📨 Avisos por Telegram activados para alertas críticas.")

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("⚠️ Esperando señal de la cámara...")
            await asyncio.sleep(0.1)
            continue

        now = time.time()
        timestamp_ms = int((now - start_time) * 1000)
        motion_detected, bboxes = motion_detector.detect(frame)

        if frame_count % Config.DETECT_EVERY_N_FRAMES == 0:
            persons, objects = object_detector.detect_objects(frame)

            frame_area = frame.shape[0] * frame.shape[1]
            motion_ratio = sum(w * h for _, _, w, h in bboxes) / frame_area
            vlm_valid = vlm_analyzer.updated_at is not None
            assessment = anomaly_detector.evaluate(
                now, frame.shape, persons, objects, motion_ratio,
                vlm_text=vlm_analyzer.latest_analysis if vlm_valid else "",
                vlm_age=(now - vlm_analyzer.updated_at) if vlm_valid else None,
            )

            for anomaly in assessment.new:
                logger.log_event(f"ANOMALIA_{anomaly.code}", anomaly.message, frame.copy(),
                                 severity=LEVEL_NAMES[anomaly.severity])
                print(f"🚨 [{LEVEL_NAMES[anomaly.severity]}] {anomaly.message}")
            notifier.update(assessment, frame)

            # El VLM confirma o descarta: se lanza ante cualquier actividad o anomalía.
            active = persons or objects or motion_detected or assessment.level > NORMAL
            if active and not vlm_analyzer.is_analyzing:
                asyncio.create_task(vlm_analyzer.analyze_frame_async(frame.copy()))

            if (persons or objects or motion_detected) and now - last_info_event > Config.EVENT_COOLDOWN:
                last_info_event = now
                event_type = "PERSONA_DETECTADA" if persons else ("OBJETO_DETECTADO" if objects else "MOVIMIENTO")
                logger.log_event(event_type, f"Personas: {len(persons)}, Objetos: {len(objects)}", frame.copy())
        frame_count += 1

        draw_detections(frame, persons, objects)
        frame = hand_tracker.process(frame, timestamp_ms)

        if hand_tracker.consume_ack() and assessment.level > NORMAL:
            anomaly_detector.acknowledge(now)
            print("👍 Alerta reconocida por gesto.")

        if vlm_analyzer.latest_analysis:
            cv2.putText(frame, f"ANALISIS: {vlm_analyzer.latest_analysis[:75].encode('ascii', 'ignore').decode()}",
                        (20, Config.FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        frame = draw_alert(frame, assessment)

        dash_app.frame_buffer = frame.copy()
        dash_app.status.update(level=assessment.level_name, summary=assessment.summary,
                               persons=len(persons), acknowledged=assessment.acknowledged)

        cv2.imshow(window_name, frame)

        if hand_tracker.trigger_exit:
            print("🛑 Salida por gesto (puño) activada.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:  # cerrada con la X
            break

        await asyncio.sleep(0.001)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Smart Sentinel detenido limpiamente por el usuario.")
