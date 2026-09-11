import asyncio
import cv2
import time
from config.settings import Config
from core.motion_detector import MotionDetector
from core.hand_tracker import HandTracker
from core.vlm_analyzer import VLMAnalyzer

async def main():
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    motion_detector = MotionDetector(min_area=Config.MIN_CONTOUR_AREA)
    hand_tracker = HandTracker()
    vlm_analyzer = VLMAnalyzer()

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Espejo aplicado antes de enviar al tracker
        frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - start_time) * 1000)

        # 1. Detección de movimiento
        motion_detected, bboxes = motion_detector.detect(frame)

        # 2. Tracking de mano y UI limpia
        frame = hand_tracker.process(frame, timestamp_ms)

        if hand_tracker.trigger_exit:
            break

        # 3. VLM Asíncrono
        if motion_detected and not vlm_analyzer.is_analyzing:
            asyncio.create_task(vlm_analyzer.analyze_frame_async(frame.copy()))

        # Subtítulo discreto e integrado del VLM en la parte inferior
        if vlm_analyzer.latest_analysis:
            txt = vlm_analyzer.latest_analysis[:70]
            cv2.putText(frame, txt, (20, Config.FRAME_HEIGHT - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Smart Sentinel", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.005)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
