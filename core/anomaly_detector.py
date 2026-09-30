"""Motor de anomalías de Smart Sentinel.

Fusiona cinco fuentes de señal en una única evaluación con severidad:

* YOLO        -> armas, aglomeraciones, objetos abandonados, presencia nocturna
* Seguimiento -> merodeo (persona quieta mucho tiempo) y caídas (de pie -> tumbado)
* Movimiento  -> movimiento brusco y sostenido con personas presentes
* VLM         -> palabras clave con normalización, negación ("no hay arma") y TTL
* Fusión      -> varias advertencias simultáneas escalan a CRÍTICO

Las anomalías se confirman con histéresis (varios ciclos seguidos) para no saltar por un
falso positivo de un solo fotograma, y avisan una sola vez (con re-aviso periódico).
"""
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime

from config.settings import Config
from core.tracker import CentroidTracker

NORMAL, WARNING, CRITICAL = 0, 1, 2
LEVEL_NAMES = {NORMAL: "NORMAL", WARNING: "WARNING", CRITICAL: "CRITICAL"}

CRITICAL_OBJECTS = {"knife": "cuchillo", "baseball bat": "bate"}
WARNING_OBJECTS = {"scissors": "tijeras"}
LEFTABLE_OBJECTS = {"backpack": "mochila", "suitcase": "maleta", "handbag": "bolso"}

# Raíces de palabras (sin acentos, en minúsculas) que activan cada nivel en el texto del VLM.
VLM_CRITICAL = ("arma", "pistola", "cuchill", "fuego", "incendi", "humo", "violen", "agres",
                "pelea", "pelean", "robo", "robando", "hiriend", "herid", "sangre", "caid", "desplomad", "inconscient")
VLM_WARNING = ("sospech", "intruso", "peligro", "forzand", "escalando", "encapuchad", "mascara",
               "merodea", "amenaz", "anomal")
NEGATIONS = {"no", "sin", "ningun", "ninguna", "ni", "nada", "tampoco"}
VLM_NORMAL_MARKERS = ("escena normal",)


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


@dataclass
class Anomaly:
    code: str
    severity: int
    message: str


@dataclass
class Assessment:
    level: int = NORMAL
    anomalies: list = field(default_factory=list)   # todas las activas
    new: list = field(default_factory=list)         # las que hay que notificar ahora
    acknowledged: bool = False

    @property
    def level_name(self):
        return LEVEL_NAMES[self.level]

    @property
    def summary(self):
        return " | ".join(a.message for a in self.anomalies) or "Sin anomalías"


class AnomalyDetector:
    # Ciclos consecutivos necesarios para confirmar cada código (los temporales ya tienen duración).
    CONFIRM = {"ARMA": 2, "AGLOMERACION": 3, "MOV_BRUSCO": 1, "PRESENCIA_NOCTURNA": 3, "VLM": 1}
    MOTION_RATIO = 0.35
    MOTION_SECONDS = 1.0

    def __init__(self):
        self.persons = CentroidTracker(max_dist_ratio=0.2, max_age=1.5, move_radius=0.12)
        self.objects = CentroidTracker(max_dist_ratio=0.05, max_age=3.0, move_radius=0.03)
        self._hits = {}
        self._latest = {}          # último Anomaly visto por código (para el periodo de gracia)
        self._active = {}          # code -> Anomaly activa en el ciclo anterior
        self._last_alert = {}      # code -> instante del último aviso
        self._motion_high_since = None
        self._muted_until = 0.0
        self._acked_level = NORMAL

    # ------------------------------------------------------------------ API
    def acknowledge(self, now):
        """Reconoce las alertas actuales (p. ej. con 👍): silencia hasta que empeoren o pase el tiempo."""
        level = max((a.severity for a in self._active.values()), default=NORMAL)
        if level > NORMAL:
            self._muted_until = now + Config.ACK_SECONDS
            self._acked_level = level

    def evaluate(self, now, frame_shape, persons, objects, motion_ratio=0.0,
                 vlm_text="", vlm_age=None, hour=None):
        """persons: [(x1,y1,x2,y2,conf)], objects: [(x1,y1,x2,y2,label,conf)]."""
        hour = datetime.now().hour if hour is None else hour
        candidates = {}

        p_tracks = self.persons.update([(*p[:4], "person") for p in persons], frame_shape, now)
        o_tracks = self.objects.update([(*o[:4], o[4]) for o in objects], frame_shape, now)

        self._check_weapons(objects, candidates)
        self._check_crowd(len(persons), candidates)
        self._check_night(len(persons), hour, candidates)
        self._check_motion(now, len(persons), motion_ratio, candidates)
        self._check_loitering(now, p_tracks, candidates)
        self._check_falls(now, p_tracks, frame_shape, candidates)
        self._check_abandoned(now, o_tracks, p_tracks, candidates)
        self._check_vlm(vlm_text, vlm_age, len(persons), candidates)

        active = self._debounce(candidates)
        self._escalate(active)
        return self._build_assessment(now, active)

    # ------------------------------------------------------------ detectores
    def _check_weapons(self, objects, out):
        found = [(CRITICAL_OBJECTS[o[4]], CRITICAL) for o in objects if o[4] in CRITICAL_OBJECTS]
        found += [(WARNING_OBJECTS[o[4]], WARNING) for o in objects if o[4] in WARNING_OBJECTS]
        if found:
            severity = max(s for _, s in found)
            names = ", ".join(sorted({n for n, _ in found}))
            out["ARMA"] = Anomaly("ARMA", severity, f"Posible objeto peligroso: {names}")

    def _check_crowd(self, n, out):
        if n >= Config.CROWD_THRESHOLD:
            severity = CRITICAL if n >= Config.CROWD_THRESHOLD * 2 else WARNING
            out["AGLOMERACION"] = Anomaly("AGLOMERACION", severity, f"Aglomeración: {n} personas")

    def _check_night(self, n, hour, out):
        start, end = Config.NIGHT_START, Config.NIGHT_END
        if n == 0 or start == end:
            return
        in_night = (hour >= start or hour < end) if start > end else (start <= hour < end)
        if in_night:
            out["PRESENCIA_NOCTURNA"] = Anomaly("PRESENCIA_NOCTURNA", WARNING,
                                                f"Presencia fuera de horario ({hour:02d}h)")

    def _check_motion(self, now, n_persons, ratio, out):
        if n_persons > 0 and ratio >= self.MOTION_RATIO:
            self._motion_high_since = self._motion_high_since or now
            if now - self._motion_high_since >= self.MOTION_SECONDS:
                out["MOV_BRUSCO"] = Anomaly("MOV_BRUSCO", WARNING,
                                            f"Movimiento brusco sostenido ({ratio:.0%} de la imagen)")
        else:
            self._motion_high_since = None

    def _check_loitering(self, now, tracks, out):
        for t in tracks:
            if t.stationary_for(now) >= Config.LOITER_SECONDS:
                out["MERODEO"] = Anomaly("MERODEO", WARNING,
                                         f"Persona #{t.id} quieta en la misma zona {t.stationary_for(now):.0f}s")
                return

    def _check_falls(self, now, tracks, frame_shape, out):
        frame_h = frame_shape[0]
        for t in tracks:
            w, h = t.size
            if w <= 0 or h <= 0:
                continue
            _, cy = t.center
            if h / w >= 1.25:                       # de pie
                t.upright_seen_at, t.upright_cy, t.lying_since = now, cy, None
            elif w / h >= 1.1:                      # tumbado
                t.lying_since = t.lying_since or now
                lying_for = now - t.lying_since
                came_from_standing = (t.upright_seen_at is not None
                                      and t.lying_since - t.upright_seen_at <= 3.0
                                      and cy - t.upright_cy > 0.08 * frame_h)
                if lying_for >= Config.FALL_CONFIRM_SECONDS and came_from_standing:
                    out["CAIDA"] = Anomaly("CAIDA", CRITICAL, f"Posible caída de la persona #{t.id}")
                    return

    def _check_abandoned(self, now, obj_tracks, person_tracks, out):
        for t in obj_tracks:
            if t.label not in LEFTABLE_OBJECTS or t.stationary_for(now) < Config.ABANDON_SECONDS:
                continue
            ox, oy = t.center
            reach = max(t.size) * 1.5
            near_person = any(abs(p.center[0] - ox) < reach + p.size[0] / 2 and
                              abs(p.center[1] - oy) < reach + p.size[1] / 2 for p in person_tracks)
            if not near_person:
                name = LEFTABLE_OBJECTS[t.label]
                out["OBJETO_ABANDONADO"] = Anomaly("OBJETO_ABANDONADO", WARNING,
                                                   f"{name.capitalize()} sin dueño hace {t.stationary_for(now):.0f}s")
                return

    def _check_vlm(self, text, age, n_persons, out):
        if not text or (age is not None and age > Config.VLM_TTL):
            return
        norm = normalize(text)
        if any(m in norm for m in VLM_NORMAL_MARKERS):
            return
        words = re.findall(r"[a-zñ]+", norm)
        severity, hit = NORMAL, None
        for i, word in enumerate(words):
            level = (CRITICAL if word.startswith(VLM_CRITICAL) else
                     WARNING if word.startswith(VLM_WARNING) else NORMAL)
            if level == NORMAL or any(w in NEGATIONS for w in words[max(0, i - 3):i]):
                continue
            if level > severity:
                severity, hit = level, word
        if severity == NORMAL:
            return
        # Un VLM sin ninguna persona detectada que grite "violencia" suele alucinar: se rebaja.
        if severity == CRITICAL and n_persons == 0:
            severity = WARNING
        out["VLM"] = Anomaly("VLM", severity, f"Análisis visual: «{text.strip()[:80]}»")

    # ------------------------------------------------------------- fusión
    def _debounce(self, candidates):
        """Confirma con histéresis: sube al detectar, baja de uno en uno al dejar de detectar."""
        active = {}
        for code in set(self._hits) | set(candidates):
            need = self.CONFIRM.get(code, 1)
            hits = self._hits.get(code, 0)
            hits = min(hits + 1, need + 2) if code in candidates else hits - 1
            if code in candidates:
                self._latest[code] = candidates[code]
            if hits <= 0:
                self._hits.pop(code, None)
                self._latest.pop(code, None)
                continue
            self._hits[code] = hits
            if hits >= need:
                active[code] = replace(self._latest[code])   # copia: la escalada no toca el original
        return active

    @staticmethod
    def _escalate(active):
        """Dos o más advertencias distintas a la vez equivalen a una situación crítica."""
        warnings = [a for a in active.values() if a.severity == WARNING]
        if len(warnings) >= 2:
            for a in warnings:
                a.severity = CRITICAL
                a.message += " (escalada: varias señales)"

    def _build_assessment(self, now, active):
        level = max((a.severity for a in active.values()), default=NORMAL)
        new = []
        for code, a in active.items():
            first_time = code not in self._active
            stale = now - self._last_alert.get(code, -1e9) >= Config.REALERT_SECONDS
            if first_time or stale:
                new.append(a)
                self._last_alert[code] = now
        self._active = active

        if level > self._acked_level:                 # empeoró: rompe el silencio
            self._muted_until = 0.0
        acknowledged = now < self._muted_until and level <= self._acked_level and level > NORMAL
        if acknowledged:
            new = []
        if level == NORMAL:
            self._acked_level, self._muted_until = NORMAL, 0.0
        return Assessment(level, list(active.values()), new, acknowledged)
