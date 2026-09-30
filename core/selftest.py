"""`Smart Sentinel --selftest`: comprueba sin cámara ni ventana que todo lo empaquetado carga y funciona
(modelos, detector, gestos, anomalías, base de datos y dashboard con autenticación). Devuelve 0 si todo va bien."""
import os
import time
import traceback

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")


def _step(name, fn):
    result = fn()
    print(f"  ✔ {name}" + (f" — {result}" if result else ""))


def run():
    import numpy as np

    from config.settings import Config

    frame = np.zeros((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), np.uint8)
    state = {}
    print("SELFTEST: empezando")

    def detector():
        from modules.vision import PersonAndAnomalyDetector
        d = PersonAndAnomalyDetector()
        persons, objects = d.detect_objects(frame)
        assert isinstance(persons, list) and isinstance(objects, list)
        return f"acelerador {d.provider}"

    def hands():
        from modules.gestures import HandTracker
        out = HandTracker().process(frame.copy(), 100)
        assert out.shape == frame.shape

    def anomalies():
        from core.anomaly_detector import AnomalyDetector
        a = AnomalyDetector().evaluate(time.time(), frame.shape, [], [])
        assert a.level == 0

    def logging_db():
        from core.event_logger import EventLogger
        EventLogger().log_event("SELFTEST", "prueba automática", frame, severity="INFO")

    def dashboard():
        import dashboard.app as dash
        dash.configure_auth()
        client = dash.app.test_client()
        assert client.get("/api/status").status_code == 401, "el dashboard no exige clave"
        r = client.post("/login", data={"password": dash.app.config["SENTINEL_PASSWORD"]})
        assert r.status_code == 302 and client.get("/api/status").status_code == 200
        return "autenticación correcta"

    def vlm():
        from modules.vision import VLMAnalyzer
        VLMAnalyzer()

    for name, fn in (("detector de objetos (ONNX)", detector), ("rastreador de manos (MediaPipe)", hands),
                     ("motor de anomalías", anomalies), ("registro de eventos (SQLite)", logging_db),
                     ("dashboard + autenticación", dashboard), ("cliente del análisis visual", vlm)):
        try:
            _step(name, fn)
        except Exception:
            print(f"  ✘ {name}")
            traceback.print_exc()
            print("SELFTEST FALLIDO")
            return 1
    print("SELFTEST OK")
    return 0
