import asyncio
import multiprocessing
import os
import sys
import threading
import time
import traceback

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import cv2

import dashboard.app as dash_app
from config.settings import Config
from core.alerts import AlertNotifier
from core.anomaly_detector import Assessment, LEVEL_NAMES, NORMAL, AnomalyDetector
from core.event_logger import EventLogger
from core.runtime import setup_logging, show_error
from core.worker import DetectionWorker
from modules.camera import Camera
from modules.gestures import HandTracker
from modules.motion import MotionDetector
from modules.overlay import draw_alert
from modules.vision import PersonAndAnomalyDetector, VLMAnalyzer

CREDENTIALS_ON_SCREEN_SECONDS = 20.0   # clave del dashboard visible al arrancar (si es automática)
SILENT_ALERT_DASHBOARD_SECONDS = 300.0


def draw_detections(frame, persons, objects):
    for (x1, y1, x2, y2, conf) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
        cv2.putText(frame, f"PERSONA {int(conf * 100)}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
    for (x1, y1, x2, y2, label, conf) in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(frame, f"{label.upper()} {int(conf * 100)}%", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)


def draw_text(frame, text, y, color=(255, 255, 255), scale=0.55):
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


async def main():
    log_path = setup_logging()

    password, automatic_password, password_file = dash_app.configure_auth()
    threading.Thread(target=dash_app.run_dashboard, daemon=True).start()

    try:
        camera = Camera()
    except RuntimeError as e:
        show_error("Smart Sentinel", str(e))
        return

    motion_detector = MotionDetector(min_area=Config.MIN_CONTOUR_AREA)
    object_detector = PersonAndAnomalyDetector(conf_threshold=0.50)
    detection_worker = DetectionWorker(object_detector)
    hand_tracker = HandTracker(max_hands=2)
    vlm_analyzer = VLMAnalyzer()
    anomaly_detector = AnomalyDetector()
    notifier = AlertNotifier()
    logger = EventLogger()

    start_time = time.time()
    last_info_event = 0.0
    frame_count = 0
    persons, objects = [], []
    armed = True
    toggle_notice_until = 0.0
    silent_alert_until = 0.0
    no_alert = Assessment()
    assessment = no_alert
    dashboard_url = f"http://{Config.DASHBOARD_HOST}:{Config.DASHBOARD_PORT}"

    window_name = "Smart Sentinel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)

    print("🚀 Smart Sentinel iniciado.")
    print(f"🧠 Detección de objetos: {object_detector.provider}")
    print(f"🌐 Dashboard: {dashboard_url} (protegido con clave)")
    if automatic_password:
        print(f"🔑 Clave del dashboard en: {password_file}")
    print(f"✊ Salir: puño {Config.EXIT_HOLD_SECONDS:g}s | 👍 reconocer alerta | ✌️ pausar/reanudar")
    if log_path:
        print(f"📄 Registro en: {log_path}")
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

        # YOLO corre en su propio hilo: aquí solo se le pasa el fotograma y se recogen resultados ya terminados.
        if frame_count % Config.DETECT_EVERY_N_FRAMES == 0:
            detection_worker.submit(frame.copy())
        fresh = detection_worker.poll()
        if fresh is not None:
            det_frame, persons, objects = fresh     # det_frame: el fotograma al que pertenecen esos recuadros

            evidence = det_frame.copy()      # fotograma con recuadros: lo que se guarda y se envía
            draw_detections(evidence, persons, objects)

            if armed:
                # Los primeros segundos MOG2 aprende el fondo y la cámara ajusta la exposición: no fiarse.
                motion_ratio = motion_detector.last_ratio if now - start_time > Config.WARMUP_SECONDS else 0.0
                vlm_valid = vlm_analyzer.updated_at is not None
                assessment = anomaly_detector.evaluate(
                    now, det_frame.shape, persons, objects, motion_ratio,
                    vlm_text=vlm_analyzer.latest_analysis if vlm_valid else "",
                    vlm_age=(now - vlm_analyzer.updated_at) if vlm_valid else None,
                )

                for anomaly in assessment.new:
                    logger.log_event(f"ANOMALIA_{anomaly.code}", anomaly.message, evidence,
                                     severity=LEVEL_NAMES[anomaly.severity])
                    print(f"🚨 [{LEVEL_NAMES[anomaly.severity]}] {anomaly.message}")
                notifier.update(assessment, evidence)

                # El VLM confirma o descarta: se lanza ante cualquier actividad o anomalía.
                active = persons or objects or motion_detected or assessment.level > NORMAL
                if active and not vlm_analyzer.is_analyzing:
                    asyncio.create_task(vlm_analyzer.analyze_frame_async(det_frame))

                if (persons or objects or motion_detected) and now - last_info_event > Config.EVENT_COOLDOWN:
                    last_info_event = now
                    event_type = "PERSONA_DETECTADA" if persons else ("OBJETO_DETECTADO" if objects else "MOVIMIENTO")
                    logger.log_event(event_type, f"Personas: {len(persons)}, Objetos: {len(objects)}", evidence)
            else:
                assessment = no_alert
        frame_count += 1

        clean_frame = frame.copy()   # sin dibujos: es lo que se guarda y se envía en los avisos
        draw_detections(frame, persons, objects)
        frame = hand_tracker.process(frame, timestamp_ms)

        # --- Gestos de seguridad ---
        if hand_tracker.consume_panic():
            # Alerta silenciosa: ni sonido, ni aviso en pantalla, ni mensaje en consola.
            logger.log_event("PANICO_SILENCIOSO", "Gesto de pánico silencioso activado", clean_frame,
                             severity="CRITICAL")
            notifier.send_telegram("🆘 Smart Sentinel: ALERTA SILENCIOSA (gesto de pánico)", clean_frame)
            silent_alert_until = now + SILENT_ALERT_DASHBOARD_SECONDS

        if hand_tracker.consume_arm_toggle():
            armed = not armed
            toggle_notice_until = now + 2.5
            if armed:
                anomaly_detector.reset()
            print("▶️ Vigilancia reanudada." if armed else "⏸️ Vigilancia en pausa.")

        if hand_tracker.consume_ack() and assessment.level > NORMAL:
            anomaly_detector.acknowledge(now)
            print("👍 Alerta reconocida por gesto.")

        # --- Interfaz ---
        if vlm_analyzer.latest_analysis:
            draw_text(frame, f"ANALISIS: {vlm_analyzer.latest_analysis[:75].encode('ascii', 'ignore').decode()}",
                      Config.FRAME_HEIGHT - 30, (0, 220, 255), 0.5)
        frame = draw_alert(frame, assessment)
        if not armed:
            draw_text(frame, "VIGILANCIA EN PAUSA  (2 dedos para reanudar)", 100, (0, 200, 255), 0.7)
        elif now < toggle_notice_until:
            draw_text(frame, "VIGILANCIA ACTIVA", 100, (0, 255, 120), 0.7)
        if automatic_password and now - start_time < CREDENTIALS_ON_SCREEN_SECONDS:
            draw_text(frame, f"Dashboard: {dashboard_url}   clave: {password}", frame.shape[0] - 60, (255, 255, 255))

        dash_app.frame_buffer = frame.copy()
        level_name, summary = assessment.level_name, assessment.summary
        if now < silent_alert_until:      # solo visible en el dashboard, nunca en la ventana local
            level_name, summary = "CRITICAL", "ALERTA SILENCIOSA: gesto de pánico"
        elif not armed:
            level_name, summary = "NORMAL", "Vigilancia en pausa"
        dash_app.status.update(level=level_name, summary=summary, persons=len(persons),
                               acknowledged=assessment.acknowledged)

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

    detection_worker.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()   # necesario en el ejecutable de Windows
    if "--selftest" in sys.argv:
        setup_logging()
        from core.selftest import run as selftest
        sys.exit(selftest())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Smart Sentinel detenido limpiamente por el usuario.")
    except Exception as e:
        traceback.print_exc()
        show_error("Smart Sentinel - error inesperado",
                   f"{type(e).__name__}: {e}\n\nDetalles en: {Config.LOG_PATH}")
        sys.exit(1)
