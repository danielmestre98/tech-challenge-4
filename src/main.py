import os
from collections import defaultdict, deque
import time


import cv2

from .config import parse_args
from .face_emotion import deepface_detect_and_emotion
from .tracking import TrackManager
from .anomaly import MotionAnomaly
from .report import build_report, write_report_json, write_report_md, with_timestamps

from .activity import ActivityModel, torch_available, resolve_device


def main():
    cfg = parse_args()

    os.makedirs(cfg.outdir, exist_ok=True)
    out_video_path = os.path.join(cfg.outdir, "annotated.mp4")
    out_json_path = os.path.join(cfg.outdir, "report.json")
    out_md_path = os.path.join(cfg.outdir, "report.md")

    cap = cv2.VideoCapture(cfg.input)
    if not cap.isOpened():
        raise RuntimeError(f"Não consegui abrir: {cfg.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))
    start_time = time.time()
    print(f"Processando: {cfg.input} | {W}x{H} | fps={fps:.2f} | total_frames={total_frames}")


    # atividade (opcional)
    activity_enabled = (not cfg.no_activity)
    activity_model = None
    if activity_enabled:
        if not torch_available():
            print("Torch/torchvision não encontrado. Rodando sem atividade.")
            activity_enabled = False
        else:
            device = resolve_device(cfg.device)
            activity_model = ActivityModel(device=device)

    activity_buf = deque(maxlen=16)
    activity_label = "unknown"
    activity_conf = 0.0
    activity_counts = defaultdict(int)
    activity_every_frames = max(1, int(round(cfg.activity_every_sec * fps)))
    last_activity_frame = -10**9

    # tracking + emoções
    tracks = TrackManager(max_track_age=cfg.max_track_age)
    frames_with_deepface = 0
    emotion_counts_global = defaultdict(int)

    # anomalia
    anomaly = MotionAnomaly(z_thresh=cfg.anom_z)

    frame_idx = -1
    while True:
        if frame_idx % 100 == 0 and frame_idx > 0:
          elapsed = time.time() - start_time
          speed = frame_idx / elapsed if elapsed > 0 else 0
          tf = total_frames if total_frames > 0 else "?"
          print(f"[{frame_idx}/{tf}] ~{speed:.2f} fps | deepface_calls={frames_with_deepface} | anom_events={len(anomaly.events)}")
      

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # anomalia em todo frame
        is_anom, z = anomaly.update(frame, frame_idx)

        # atualiza trackers nos frames intermediários
        tracks.update_trackers(frame, W, H)

        # DeepFace a cada N frames
        if frame_idx % cfg.detect_every == 0:
            dets = deepface_detect_and_emotion(frame, detector_backend=cfg.detector)
            frames_with_deepface += 1

            updated = tracks.match_and_update(frame, dets, W, H)

            # atualiza contagem global de emoções (por detecção)
            for d in dets:
                dom = d.get("dominant_emotion") or "unknown"
                emotion_counts_global[dom] += 1

        # atividade (opcional)
        if activity_enabled:
            small = cv2.resize(frame, (112, 112))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            activity_buf.append(rgb)

            if (frame_idx - last_activity_frame) >= activity_every_frames and len(activity_buf) == 16:
                activity_label, activity_conf = activity_model.predict_top1(list(activity_buf))
                activity_counts[activity_label] += 1
                last_activity_frame = frame_idx

        # overlays
        act_text = f"Atividade: {activity_label} ({activity_conf:.2f})" if activity_enabled else "Atividade: (desligada)"
        cv2.putText(frame, act_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
        cv2.putText(frame, act_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if is_anom:
            cv2.putText(frame, f"ANOMALIA (z={z:.2f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)

        for tid, tr in tracks.tracks.items():
            x, y, w, h = tr.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), 2)
            label = f"ID {tid} | {tr.last_emotion}"
            cv2.putText(frame, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)

        vw.write(frame)

    cap.release()
    vw.release()

    anomaly.finalize(frame_idx)
    events = with_timestamps(anomaly.events, fps)

    frames_read = frame_idx + 1

    report = build_report(
        input_path=cfg.input,
        fps=fps,
        W=W,
        H=H,
        total_frames_in_video=total_frames,
        frames_read=frames_read,
        frames_with_deepface=frames_with_deepface,
        emotion_counts_global=emotion_counts_global,
        activity_enabled=activity_enabled,
        activity_every_sec=cfg.activity_every_sec if activity_enabled else None,
        activity_counts=activity_counts,
        anomaly_events=events,
        anom_z=cfg.anom_z,
        outputs={
            "annotated_video": out_video_path,
            "report_json": out_json_path,
            "report_md": out_md_path,
        },
    )

    write_report_json(out_json_path, report)
    write_report_md(out_md_path, report)

    print("OK!")
    print(f"Vídeo anotado: {out_video_path}")
    print(f"Relatório JSON: {out_json_path}")
    print(f"Relatório MD: {out_md_path}")


if __name__ == "__main__":
    main()
