import unittest
from types import SimpleNamespace as P

try:
    from modules.gestures import HandTracker
    HAVE_MEDIAPIPE = True
except ImportError:
    HAVE_MEDIAPIPE = False


def hand(thumb=0, index=0, middle=0, ring=0, pinky=0, thumb_up=False):
    """Mano sintética (coordenadas normalizadas, y crece hacia abajo) con la muñeca abajo."""
    lm = [P(x=0.5, y=0.5) for _ in range(21)]
    lm[0] = P(x=0.5, y=0.9)
    for base, pip, tip, x, up in ((5, 6, 8, 0.44, index), (9, 10, 12, 0.50, middle),
                                  (13, 14, 16, 0.56, ring), (17, 18, 20, 0.62, pinky)):
        lm[base] = P(x=x, y=0.65)
        lm[pip] = P(x=x, y=0.55)
        lm[tip] = P(x=x, y=0.30 if up else 0.70)      # extendido: lejos de la muñeca; plegado: cerca
    lm[2] = P(x=0.40, y=0.80)
    lm[3] = P(x=0.38, y=0.70 if thumb_up else 0.80)
    if thumb_up:
        lm[4] = P(x=0.38, y=0.55)                       # punta del pulgar por encima de todo
    else:
        lm[4] = P(x=0.22, y=0.72) if thumb else P(x=0.52, y=0.78)
    return lm


@unittest.skipUnless(HAVE_MEDIAPIPE, "requiere mediapipe")
class GestureTests(unittest.TestCase):
    def setUp(self):
        self.t = HandTracker.__new__(HandTracker)      # sin cargar el modelo ni abrir la cámara
        self.t._holds = {"panic": [None, float("-inf")], "arm": [None, float("-inf")]}
        self.t._pending = {"panic": False, "arm": False}

    def poses(self, lm):
        f = self.t._finger_states(lm)
        return dict(fist=sum(f[1:]) == 0 and not self.t._is_thumbs_up(lm, f),
                    thumbs_up=self.t._is_thumbs_up(lm, f),
                    panic=f[1:] == [1, 1, 1, 0], arm=f[1:] == [1, 1, 0, 0], index_only=f[1] == 1 and not any(f[2:]),
                    palm=sum(f) == 5)

    def test_poses_are_mutually_exclusive(self):
        cases = {
            "fist": hand(), "thumbs_up": hand(thumb_up=True), "panic": hand(index=1, middle=1, ring=1),
            "arm": hand(index=1, middle=1), "index_only": hand(index=1),
            "palm": hand(1, 1, 1, 1, 1),
        }
        for name, lm in cases.items():
            got = {k for k, v in self.poses(lm).items() if v}
            self.assertEqual(got, {name}, f"{name} -> {got}")

    def test_hold_requires_continuous_pose(self):
        t = self.t
        t._hold("panic", True, 0.0, 2.0, 30)
        t._hold("panic", True, 1.0, 2.0, 30)
        self.assertFalse(t.consume_panic())
        t._hold("panic", False, 1.5, 2.0, 30)            # se rompe la pose: reinicia
        t._hold("panic", True, 2.0, 2.0, 30)
        t._hold("panic", True, 3.9, 2.0, 30)
        self.assertFalse(t.consume_panic())
        t._hold("panic", True, 4.1, 2.0, 30)
        self.assertTrue(t.consume_panic())
        self.assertFalse(t.consume_panic())              # se consume una sola vez

    def test_cooldown_prevents_repeated_panic(self):
        t = self.t
        for now in (0.0, 2.1):
            t._hold("panic", True, now, 2.0, 30)
        self.assertTrue(t.consume_panic())
        for now in (3.0, 5.5, 8.0):                      # sigue con la pose: dentro del enfriamiento
            t._hold("panic", True, now, 2.0, 30)
        self.assertFalse(t.consume_panic())


if __name__ == "__main__":
    unittest.main()
