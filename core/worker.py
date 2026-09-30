import queue
import threading


class DetectionWorker:
    """Ejecuta la detección de objetos (YOLO) en un hilo aparte para que el bucle que dibuja y sigue las manos
    nunca espere a la red neuronal. Solo se guarda el fotograma más reciente: si la detección va más lenta que la
    cámara, se descartan los intermedios en lugar de acumular retraso. onnxruntime libera el GIL al inferir."""

    def __init__(self, detector):
        self._detector = detector
        self._inbox = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._busy = False
        self._result = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="detection-worker", daemon=True)
        self._thread.start()

    def submit(self, frame):
        """Encola `frame` si el hilo está libre; devuelve False (y no hace nada) si sigue ocupado."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self._inbox.put(frame)
        return True

    def poll(self):
        """(fotograma, personas, objetos) de la última detección terminada, una sola vez; None si no hay nueva."""
        with self._lock:
            result, self._result = self._result, None
        return result

    def stop(self):
        self._stop.set()
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            frame = self._inbox.get()
            if frame is None:
                break
            try:
                persons, objects = self._detector.detect_objects(frame)
                result = (frame, persons, objects)
            except Exception:                      # detect_objects ya avisa; el hilo no debe morir
                result = (frame, [], [])
            with self._lock:
                self._result = result
                self._busy = False
