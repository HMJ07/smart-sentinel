import math
from dataclasses import dataclass, field


@dataclass
class Track:
    id: int
    label: str
    bbox: tuple
    first_seen: float
    last_seen: float
    anchor: tuple            # centro donde "se quedó quieto"
    anchor_time: float       # desde cuándo permanece cerca del ancla
    # Estado usado por la detección de caídas
    upright_seen_at: float = None
    upright_cy: float = None
    lying_since: float = None
    extra: dict = field(default_factory=dict)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def size(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

    def dwell(self, now):
        """Segundos que lleva visible."""
        return now - self.first_seen

    def stationary_for(self, now):
        """Segundos que lleva prácticamente sin moverse."""
        return now - self.anchor_time


class CentroidTracker:
    """Seguimiento simple por centroides (asignación voraz por distancia, misma etiqueta).

    move_radius: fracción de la diagonal del fotograma que se considera 'quieto'.
    """

    def __init__(self, max_dist_ratio=0.2, max_age=1.5, move_radius=0.06):
        self.max_dist_ratio = max_dist_ratio
        self.max_age = max_age
        self.move_radius = move_radius
        self.tracks = {}
        self._next_id = 1

    def update(self, detections, frame_shape, now):
        """detections: [(x1, y1, x2, y2, label)] -> lista de tracks activos en este ciclo."""
        h, w = frame_shape[:2]
        diag = math.hypot(w, h)
        max_dist = self.max_dist_ratio * diag
        move_dist = self.move_radius * diag

        pairs = []
        for di, (x1, y1, x2, y2, label) in enumerate(detections):
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            for tid, t in self.tracks.items():
                if t.label != label:
                    continue
                d = math.dist(c, t.center)
                if d <= max_dist:
                    pairs.append((d, di, tid))
        pairs.sort()

        used_det, used_trk, active = set(), set(), []
        for d, di, tid in pairs:
            if di in used_det or tid in used_trk:
                continue
            used_det.add(di)
            used_trk.add(tid)
            t = self.tracks[tid]
            t.bbox = tuple(detections[di][:4])
            t.last_seen = now
            if math.dist(t.center, t.anchor) > move_dist:
                t.anchor, t.anchor_time = t.center, now
            active.append(t)

        for di, (x1, y1, x2, y2, label) in enumerate(detections):
            if di in used_det:
                continue
            t = Track(self._next_id, label, (x1, y1, x2, y2), now, now, anchor=((x1 + x2) / 2, (y1 + y2) / 2), anchor_time=now)
            self.tracks[t.id] = t
            self._next_id += 1
            active.append(t)

        for tid in [i for i, t in self.tracks.items() if now - t.last_seen > self.max_age]:
            del self.tracks[tid]
        return active
