import asyncio
import cv2
import time
import threading
from config.settings import Config
from modules.camera import Camera
from modules.motion import MotionDetector
from modules.gestures import HandTracker
from modules.vision import PersonAndAnomalyDetector, VLMAnalyzer
from core.event_logger import EventLogger
import dashboard.app as dash_app

async def main():
    # Servidor Flask en hilo daemon
    dash_thread = threading.Thread(target=dash_app.run_dashboard, daemon=True)
    dash_thread.start()

    camera = Camera()
    motion_detector = MotionDetector(min_area=Config.MIN_CONTOUR_AREA)
    object_detector = PersonAndAnomalyDetector(conf_threshold=0.50)
    hand_tracker = HandTracker(max_hands=2)
    vlm_analyzer = VLMAnalyzer()
    logger = EventLogger()

    start_time = time.time()
    last_event_time = 0

    # Crear ventana explícita de OpenCV
    window_name = "Smart Sentinel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)

    print("🚀 Smart Sentinel iniciado.")
    print("🌐 Dashboard disponible en: http://localhost:5000")

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            print("⚠️ Esperando señal de la cámara...")
            await asyncio.sleep(0.1)
            continue

        timestamp_ms = int((time.time() - start_time) * 1000)

        motion_detected, bboxes = motion_detector.detect(frame)
        persons, objects = object_detector.detect_objects(frame)

        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(frame, f"PERSONA {int(conf*100)}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        for (x1, y1, x2, y2, label, conf) in objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"{label.upper()} {int(conf*100)}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        current_time = time.time()
        if (persons or objects or motion_detected) and (current_time - last_event_time > 4):
            last_event_time = current_time
            event_type = "PERSONA_DETECTADA" if persons else ("OBJETO_DETECTADO" if objects else "MOVIMIENTO")
            desc = f"Personas: {len(persons)}, Objetos: {len(objects)}. VLM: {vlm_analyzer.latest_analysis[:50]}"
            
            logger.log_event(event_type, desc, frame.copy())

            if not vlm_analyzer.is_analyzing:
                asyncio.create_task(vlm_analyzer.analyze_frame_async(frame.copy()))

        frame = hand_tracker.process(frame, timestamp_ms)

        if vlm_analyzer.latest_analysis:
            cv2.putText(frame, f"ANALISIS: {vlm_analyzer.latest_analysis[:75]}", 
                        (20, Config.FRAME_HEIGHT - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        dash_app.frame_buffer = frame.copy()

        cv2.imshow(window_name, frame)

        if hand_tracker.trigger_exit:
            print("🛑 Desconexión por gesto activada.")
            break

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # Tecla Q o ESC
            break

        await asyncio.sleep(0.001)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Smart Sentinel detenido limpiamente por el usuario.")
