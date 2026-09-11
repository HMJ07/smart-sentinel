import asyncio
import cv2
import time
from config.settings import Config
from modules.camera import Camera
from modules.motion import MotionDetector
from modules.gestures import HandTracker
from modules.vision import PersonAndAnomalyDetector, VLMAnalyzer

async def main():
    camera = Camera()
    motion_detector = MotionDetector(min_area=Config.MIN_CONTOUR_AREA)
    object_detector = PersonAndAnomalyDetector(conf_threshold=0.50)
    hand_tracker = HandTracker(max_hands=2)
    vlm_analyzer = VLMAnalyzer()

    start_time = time.time()
    print("🚀 Smart Sentinel iniciado. Presiona 'Q' o mantén el índice en 'SALIR'.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        timestamp_ms = int((time.time() - start_time) * 1000)

        # 1. Detección de Movimiento
        motion_detected, bboxes = motion_detector.detect(frame)

        # 2. Detección de Personas y Objetos
        persons, objects = object_detector.detect_objects(frame)

        # Renderizado de Bounding Boxes Limpias (Estilo HUD)
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
            cv2.putText(frame, f"PERSONA {int(conf*100)}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        for (x1, y1, x2, y2, label, conf) in objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"{label.upper()} {int(conf*100)}%", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # 3. Disparo VLM Asíncrono ante movimiento u objetos
        if (motion_detected or persons or objects) and not vlm_analyzer.is_analyzing:
            asyncio.create_task(vlm_analyzer.analyze_frame_async(frame.copy()))

        # 4. Motor de Gestos e Interacción Táctil
        frame = hand_tracker.process(frame, timestamp_ms)

        if hand_tracker.trigger_exit:
            print("🛑 Botón SALIR pulsado.")
            break

        # Renderizado de texto VLM elegante en pantalla
        if vlm_analyzer.latest_analysis:
            cv2.putText(frame, f"ANALISIS: {vlm_analyzer.latest_analysis[:75]}", 
                        (20, Config.FRAME_HEIGHT - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow("Smart Sentinel", frame)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

        await asyncio.sleep(0.005)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
