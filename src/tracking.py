from dataclasses import dataclass, field
from collections import defaultdict
import cv2

from .vision_utils import iou, clamp_box


def _create_tracker():
    """
    Tenta criar um tracker disponível (CSRT > KCF > MOSSE).
    Em alguns OpenCV, trackers ficam em cv2.legacy.
    """
    candidates = [
        ("CSRT", "TrackerCSRT_create"),
        ("KCF", "TrackerKCF_create"),
        ("MOSSE", "TrackerMOSSE_create"),
    ]

    # cv2.*
    for _, fn in candidates:
        if hasattr(cv2, fn):
            return getattr(cv2, fn)()

    # cv2.legacy.*
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        for _, fn in candidates:
            if hasattr(legacy, fn):
                return getattr(legacy, fn)()

    raise RuntimeError("Nenhum tracker disponível. Instale opencv-contrib-python.")


@dataclass
class Track:
    id: int
    box: tuple  # (x,y,w,h)
    tracker: object = None
    age: int = 0  # frames desde o último refresh bom
    last_emotion: str = "unknown"
    emotion_counts: dict = field(default_factory=lambda: defaultdict(int))
    emotion_sum: dict = field(default_factory=lambda: defaultdict(float))


class TrackManager:
    def __init__(self, max_track_age: int, iou_threshold: float = 0.3):
        self.max_track_age = max_track_age
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update_trackers(self, frame_bgr, W: int, H: int):
        dead = []
        for tid, tr in self.tracks.items():
            if tr.tracker is None:
                continue
            ok, new_box = tr.tracker.update(frame_bgr)
            if ok:
                x, y, w, h = map(int, new_box)
                tr.box = clamp_box((x, y, w, h), W, H)
                tr.age += 1
            else:
                tr.age += 5  # penaliza falha
            if tr.age > self.max_track_age:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

    def _refresh_tracker(self, tr: Track, frame_bgr, box):
        import cv2

        def _py(v):
            # converte numpy scalar -> python nativo
            return v.item() if hasattr(v, "item") else v

        # garante bbox com tipos python puros
        x, y, w, h = [_py(v) for v in box]

        # primeiro tenta float (Rect2d)
        bbox_f = (float(x), float(y), float(w), float(h))
        bbox_i = (int(round(float(x))), int(round(float(y))),
                  int(round(float(w))), int(round(float(h))))

        tracker = _create_tracker()

        try:
            tracker.init(frame_bgr, bbox_f)
            tr.box = bbox_i  # internamente guardamos como int pra desenhar/consistência
        except cv2.error:
            # fallback: alguns builds do OpenCV preferem bbox inteiro
            tracker.init(frame_bgr, bbox_i)
            tr.box = bbox_i

        tr.tracker = tracker
        tr.age = 0


    def match_and_update(self, frame_bgr, detections: list, W: int, H: int):
        """
        detections: lista com dict {box, dominant_emotion, emotion}
        Faz matching por IOU e atualiza/ cria tracks. Re-inicializa tracker nos frames de detecção.
        Retorna lista de track IDs atualizados.
        """
        updated_ids = []

        for d in detections:
            box = clamp_box(d["box"], W, H)

            best_tid, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                v = iou(box, tr.box)
                if v > best_iou:
                    best_iou = v
                    best_tid = tid

            if best_tid is not None and best_iou >= self.iou_threshold:
                tr = self.tracks[best_tid]
            else:
                tid = self._next_id
                self._next_id += 1
                tr = Track(id=tid, box=box)
                self.tracks[tid] = tr

            # emoções
            dom = d.get("dominant_emotion") or "unknown"
            tr.last_emotion = dom
            tr.emotion_counts[dom] += 1

            emo_map = d.get("emotion", {}) or {}
            for k, v in emo_map.items():
                tr.emotion_sum[k] += float(v)

            # refresh tracker
            self._refresh_tracker(tr, frame_bgr, box)
            updated_ids.append(tr.id)

        return updated_ids
