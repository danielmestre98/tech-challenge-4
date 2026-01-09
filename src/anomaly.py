from collections import deque
import numpy as np
import cv2


class MotionAnomaly:
    """
    Anomalia por "explosão" de movimento (diferença entre frames).
    Agrupa frames consecutivos em eventos para contar "anomalias" como eventos.
    """
    def __init__(self, z_thresh=3.0, warmup=30, window=120, min_gap_frames=10):
        self.z_thresh = z_thresh
        self.warmup = warmup
        self.window = window
        self.min_gap_frames = min_gap_frames

        self.values = deque(maxlen=window)
        self.prev_gray = None

        self.events = []
        self._in_event = False
        self._event_start = None
        self._event_peak = 0.0
        self._last_event_end = -10**9

    def update(self, frame_bgr, frame_idx: int):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        motion = float(np.mean(diff))
        self.values.append(motion)

        if len(self.values) < self.warmup:
            return False, 0.0

        mu = float(np.mean(self.values))
        sigma = float(np.std(self.values)) + 1e-6
        z = (motion - mu) / sigma
        is_anom = z >= self.z_thresh

        # eventos
        if is_anom and (frame_idx - self._last_event_end) > self.min_gap_frames:
            if not self._in_event:
                self._in_event = True
                self._event_start = frame_idx
                self._event_peak = z
            else:
                self._event_peak = max(self._event_peak, z)

        if self._in_event and (not is_anom):
            self._in_event = False
            end = frame_idx
            self._last_event_end = end
            self.events.append({
                "start_frame": int(self._event_start),
                "end_frame": int(end),
                "peak_z": float(self._event_peak),
            })

        return is_anom, float(z)

    def finalize(self, last_frame_idx: int):
        # se acabou o vídeo no meio de um evento, fecha aqui
        if self._in_event:
            self._in_event = False
            self.events.append({
                "start_frame": int(self._event_start),
                "end_frame": int(last_frame_idx),
                "peak_z": float(self._event_peak),
            })
