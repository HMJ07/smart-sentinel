import unittest

from core.anomaly_detector import AnomalyDetector, NORMAL, WARNING, CRITICAL, normalize

SHAPE = (720, 1280, 3)
DAY = 12


def person(x=500, y=200, w=120, h=300):
    return (x, y, x + w, y + h, 0.9)


def obj(label, x=100, y=500, w=80, h=80):
    return (x, y, x + w, y + h, label, 0.9)


def run(det, seconds, step=0.3, start=0.0, **kw):
    """Evalúa repetidamente durante `seconds`; devuelve la última evaluación y todas las nuevas."""
    t, last, new = start, None, []
    while t <= start + seconds:
        kw.setdefault("hour", DAY)
        last = det.evaluate(t, SHAPE, **{"persons": [], "objects": [], **kw})
        new += last.new
        t += step
    return last, new


class AnomalyTests(unittest.TestCase):
    def codes(self, a):
        return {x.code for x in a.anomalies}

    def test_empty_scene_is_normal(self):
        a, _ = run(AnomalyDetector(), 3)
        self.assertEqual(a.level, NORMAL)

    def test_knife_is_critical_after_confirmation(self):
        det = AnomalyDetector()
        first = det.evaluate(0, SHAPE, [person()], [obj("knife")], hour=DAY)
        self.assertEqual(first.level, NORMAL)            # un solo ciclo no basta
        a, new = run(det, 1, start=0.3, persons=[person()], objects=[obj("knife")])
        self.assertEqual(a.level, CRITICAL)
        self.assertEqual(len([n for n in new if n.code == "ARMA"]), 1)   # avisa una sola vez

    def test_single_frame_flicker_is_ignored(self):
        det = AnomalyDetector()
        det.evaluate(0, SHAPE, [person()], [obj("knife")], hour=DAY)
        a, _ = run(det, 2, start=0.3, persons=[person()])
        self.assertEqual(a.level, NORMAL)

    def test_crowd_and_escalation(self):
        det = AnomalyDetector()
        crowd = [person(x=100 + i * 200) for i in range(4)]
        a, _ = run(det, 2, persons=crowd)
        self.assertEqual(self.codes(a), {"AGLOMERACION"})
        self.assertEqual(a.level, WARNING)
        a, _ = run(det, 2, start=2.5, persons=crowd, hour=2)   # + presencia nocturna -> escala
        self.assertEqual(a.level, CRITICAL)

    def test_loitering(self):
        det = AnomalyDetector()
        a, _ = run(det, 130, step=0.5, persons=[person()])
        self.assertIn("MERODEO", self.codes(a))

    def test_walking_person_does_not_loiter(self):
        det = AnomalyDetector()
        t, last = 0.0, None
        while t <= 40:
            last = det.evaluate(t, SHAPE, [person(x=int(100 + (t * 40) % 900))], [], hour=DAY)
            t += 0.5
        self.assertNotIn("MERODEO", self.codes(last))

    def test_fall_detected(self):
        det = AnomalyDetector()
        run(det, 1.5, persons=[person(y=150, w=110, h=320)])              # de pie
        a, _ = run(det, 3, start=1.8, persons=[person(y=420, w=300, h=110)])  # tumbado, más abajo
        self.assertIn("CAIDA", self.codes(a))
        self.assertEqual(a.level, CRITICAL)

    def test_flapping_fall_notifies_once_and_survives_single_upright_frame(self):
        det = AnomalyDetector()
        run(det, 1.5, persons=[person(y=150, w=110, h=320)])
        _, new = run(det, 3, start=1.8, persons=[person(y=420, w=300, h=110)])
        notified = len([n for n in new if n.code == "CAIDA"])
        t = 5.0
        for _ in range(6):        # parpadeo: un fotograma "de pie" suelto y de nuevo tumbado
            a = det.evaluate(t, SHAPE, [person(y=150, w=110, h=320)], [], hour=DAY)
            notified += len([n for n in a.new if n.code == "CAIDA"])
            for _ in range(4):
                t += 0.3
                a = det.evaluate(t, SHAPE, [person(y=420, w=300, h=110)], [], hour=DAY)
                notified += len([n for n in a.new if n.code == "CAIDA"])
            t += 0.3
        self.assertEqual(notified, 1)
        self.assertIn("CAIDA", self.codes(a))

    def test_fall_clears_after_person_stands_up(self):
        det = AnomalyDetector()
        run(det, 1.5, persons=[person(y=150, w=110, h=320)])
        a, _ = run(det, 3, start=1.8, persons=[person(y=420, w=300, h=110)])
        self.assertIn("CAIDA", self.codes(a))
        a, _ = run(det, 4, start=5.0, persons=[person(y=150, w=110, h=320)])
        self.assertNotIn("CAIDA", self.codes(a))

    def test_lying_from_start_is_not_a_fall(self):
        a, _ = run(AnomalyDetector(), 4, persons=[person(y=420, w=300, h=110)])
        self.assertNotIn("CAIDA", self.codes(a))

    def test_abandoned_backpack_but_not_when_owner_nearby(self):
        det = AnomalyDetector()
        a, _ = run(det, 25, step=0.5, objects=[obj("backpack")])
        self.assertIn("OBJETO_ABANDONADO", self.codes(a))
        det = AnomalyDetector()
        a, _ = run(det, 25, step=0.5, objects=[obj("backpack")], persons=[person(x=150, y=380)])
        self.assertNotIn("OBJETO_ABANDONADO", self.codes(a))

    def test_night_presence(self):
        a, _ = run(AnomalyDetector(), 2, persons=[person()], hour=2)
        self.assertIn("PRESENCIA_NOCTURNA", self.codes(a))
        a, _ = run(AnomalyDetector(), 2, persons=[person()], hour=12)
        self.assertNotIn("PRESENCIA_NOCTURNA", self.codes(a))

    def test_sustained_motion_needs_persons_and_duration(self):
        a, _ = run(AnomalyDetector(), 3, persons=[person()], motion_ratio=0.5)
        self.assertIn("MOV_BRUSCO", self.codes(a))
        a, _ = run(AnomalyDetector(), 3, motion_ratio=0.5)
        self.assertEqual(a.level, NORMAL)

    def test_vlm_keywords_negation_and_normal(self):
        cases = [
            ("Una persona con un arma en la mano", [person()], CRITICAL),
            ("No se observa ningún arma", [person()], NORMAL),
            ("Escena normal", [person()], NORMAL),
            ("Hay un intruso sospechoso", [person()], WARNING),
            ("Parece haber violencia", [], WARNING),          # sin personas -> rebajado
            ("Un hombre CAÍDO en el suelo", [person()], CRITICAL),
        ]
        for text, persons, expected in cases:
            a, _ = run(AnomalyDetector(), 1, persons=persons, vlm_text=text, vlm_age=1)
            self.assertEqual(a.level, expected, text)

    def test_vlm_expires(self):
        a, _ = run(AnomalyDetector(), 1, persons=[person()], vlm_text="Hay un arma", vlm_age=999)
        self.assertEqual(a.level, NORMAL)

    def test_acknowledge_mutes_until_worse(self):
        det = AnomalyDetector()
        crowd = [person(x=100 + i * 200) for i in range(4)]
        a, _ = run(det, 2, persons=crowd)
        det.acknowledge(2.0)
        a, new = run(det, 3, start=2.3, persons=crowd)
        self.assertTrue(a.acknowledged)
        self.assertEqual(new, [])
        a, _ = run(det, 2, start=5.5, persons=crowd, objects=[obj("knife")])   # empeora
        self.assertEqual(a.level, CRITICAL)
        self.assertFalse(a.acknowledged)

    def test_realert_after_interval(self):
        det = AnomalyDetector()
        crowd = [person(x=100 + i * 200) for i in range(4)]
        _, new = run(det, 130, step=0.5, persons=crowd)
        self.assertEqual(len([n for n in new if n.code == "AGLOMERACION"]), 3)  # t=~1, ~61, ~121

    def test_escalation_does_not_accumulate_text(self):
        det = AnomalyDetector()
        crowd = [person(x=100 + i * 200) for i in range(4)]
        a, _ = run(det, 5, persons=crowd, hour=2)
        self.assertEqual(sum(x.message.count("escalada") for x in a.anomalies), 2)

    def test_reset_clears_state(self):
        det = AnomalyDetector()
        crowd = [person(x=100 + i * 200) for i in range(4)]
        a, _ = run(det, 2, persons=crowd)
        self.assertEqual(a.level, WARNING)
        det.reset()
        a = det.evaluate(3.0, SHAPE, [], [], hour=DAY)
        self.assertEqual(a.level, NORMAL)

    def test_normalize(self):
        self.assertEqual(normalize("CAÍDA Ñu"), "caida nu")


if __name__ == "__main__":
    unittest.main()
