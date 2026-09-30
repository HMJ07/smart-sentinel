import threading
import time
import unittest

import numpy as np

from core.worker import DetectionWorker
from modules.motion import MotionDetector
from modules.smoothing import OneEuroFilter


class SlowDetector:
    def __init__(self, delay=0.05):
        self.delay, self.calls = delay, 0
        self.thread = None

    def detect_objects(self, frame):
        self.thread = threading.current_thread().name
        self.calls += 1
        time.sleep(self.delay)
        return [("p", int(frame[0, 0, 0]))], []


def wait_for(worker, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.005)
    return None


class WorkerTests(unittest.TestCase):
    def test_runs_off_the_caller_thread_and_never_blocks_it(self):
        det = SlowDetector(0.1)
        worker = DetectionWorker(det)
        frame = np.full((4, 4, 3), 7, np.uint8)
        t0 = time.perf_counter()
        self.assertTrue(worker.submit(frame))
        self.assertLess(time.perf_counter() - t0, 0.03)            # submit no espera a la detección
        self.assertIsNone(worker.poll())                           # aún no hay resultado
        fr, persons, objects = wait_for(worker)
        self.assertEqual(persons, [("p", 7)])
        self.assertIs(fr, frame)                                   # el resultado trae SU fotograma
        self.assertEqual(det.thread, "detection-worker")
        worker.stop()

    def test_busy_worker_drops_frames_instead_of_queueing(self):
        det = SlowDetector(0.15)
        worker = DetectionWorker(det)
        results = [worker.submit(np.full((2, 2, 3), i, np.uint8)) for i in range(5)]
        self.assertEqual(results, [True, False, False, False, False])
        wait_for(worker)
        self.assertEqual(det.calls, 1)
        self.assertTrue(worker.submit(np.zeros((2, 2, 3), np.uint8)))   # libre otra vez
        worker.stop()

    def test_result_is_delivered_once(self):
        worker = DetectionWorker(SlowDetector(0.01))
        worker.submit(np.zeros((2, 2, 3), np.uint8))
        self.assertIsNotNone(wait_for(worker))
        self.assertIsNone(worker.poll())
        worker.stop()

    def test_detector_exception_does_not_kill_the_thread(self):
        class Boom:
            def detect_objects(self, frame):
                raise RuntimeError("boom")
        worker = DetectionWorker(Boom())
        worker.submit(np.zeros((2, 2, 3), np.uint8))
        self.assertEqual(wait_for(worker)[1:], ([], []))
        self.assertTrue(worker.submit(np.zeros((2, 2, 3), np.uint8)))
        self.assertIsNotNone(wait_for(worker))
        worker.stop()


class MotionTests(unittest.TestCase):
    def feed(self, detector, n_static=40, moving=False):
        rng = np.random.default_rng(1)
        base = (rng.random((720, 1280, 3)) * 30 + 100).astype(np.uint8)
        for _ in range(n_static):
            detector.detect(base)
        frame = base.copy()
        if moving:
            frame[200:500, 400:900] = 255                          # objeto grande entrando en escena
        return detector.detect(frame)

    def test_still_scene_is_quiet_and_big_object_is_detected(self):
        quiet, _ = self.feed(MotionDetector(2500))
        self.assertFalse(quiet)
        detected, boxes = self.feed(MotionDetector(2500), moving=True)
        self.assertTrue(detected)
        x, y, w, h = boxes[0]                                      # coordenadas en píxeles del fotograma original
        self.assertTrue(350 <= x <= 450 and 150 <= y <= 250 and 450 <= w <= 560 and 250 <= h <= 350, boxes)

    def test_small_blob_below_min_area_is_ignored(self):
        d = MotionDetector(2500)
        rng = np.random.default_rng(1)
        base = (rng.random((720, 1280, 3)) * 30 + 100).astype(np.uint8)
        for _ in range(40):
            d.detect(base)
        frame = base.copy()
        frame[300:330, 300:330] = 255                              # 900 px < 2500
        self.assertFalse(d.detect(frame)[0])


class OneEuroTests(unittest.TestCase):
    def test_removes_jitter_when_still(self):
        f, rng, out = OneEuroFilter(1.0, 0.012), np.random.default_rng(0), []
        for i in range(120):
            out.append(f(500 + rng.normal(0, 3), i / 30))
        self.assertLess(np.std(out[30:]), 1.5)                    # el ruido de 3 px queda a menos de la mitad

    def test_follows_fast_motion_with_little_lag(self):
        f = OneEuroFilter(1.0, 0.012)
        last = 0
        for i in range(30):
            last = f(i * 40.0, i / 30)                             # 1200 px/s
        self.assertLess(29 * 40 - last, 60)


if __name__ == "__main__":
    unittest.main()
