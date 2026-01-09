import json
from collections import defaultdict

from .vision_utils import fmt_time


def build_report(
    input_path: str,
    fps: float,
    W: int,
    H: int,
    total_frames_in_video: int,
    frames_read: int,
    frames_with_deepface: int,
    emotion_counts_global: dict,
    activity_enabled: bool,
    activity_every_sec: float | None,
    activity_counts: dict,
    anomaly_events: list,
    anom_z: float,
    outputs: dict,
):
    emo_sorted = sorted(emotion_counts_global.items(), key=lambda x: x[1], reverse=True)
    act_sorted = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True) if activity_enabled else []

    return {
        "input": input_path,
        "fps": fps,
        "resolution": {"width": W, "height": H},
        "total_frames_in_video": total_frames_in_video,
        "frames_read": frames_read,
        "frames_with_deepface_inference": frames_with_deepface,
        "activity_enabled": activity_enabled,
        "activity_inference_every_sec": activity_every_sec if activity_enabled else None,
        "top_emotions": [{"emotion": k, "count": int(v)} for k, v in emo_sorted[:5]],
        "top_activities": [{"activity": k, "count": int(v)} for k, v in act_sorted[:10]],
        "anomalies": {
            "count_events": len(anomaly_events),
            "z_threshold": anom_z,
            "events": anomaly_events,
        },
        "outputs": outputs,
    }


def write_report_json(path: str, report: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def write_report_md(path: str, report: dict):
    fps = report["fps"]
    events = report["anomalies"]["events"]

    md = []
    md.append("# Relatório (gerado automaticamente)")
    md.append("")
    md.append("## Métricas principais")
    md.append(f"- Total de frames analisados (lidos): **{report['frames_read']}**")
    md.append(f"- Frames com inferência DeepFace (emoção/rostos): **{report['frames_with_deepface_inference']}**")
    md.append(f"- Número de anomalias (eventos): **{report['anomalies']['count_events']}**")
    md.append("")
    md.append("## Emoções (top)")
    for item in report["top_emotions"]:
        md.append(f"- {item['emotion']}: {item['count']}")
    md.append("")
    md.append("## Atividades (top)")
    if report["activity_enabled"]:
        for item in report["top_activities"]:
            md.append(f"- {item['activity']}: {item['count']}")
    else:
        md.append("- (atividade desligada)")
    md.append("")
    md.append("## Anomalias (eventos)")
    if not events:
        md.append("- Nenhum evento acima do limiar.")
    else:
        for e in events:
            md.append(f"- {fmt_time(e['start_time_sec'])} → {fmt_time(e['end_time_sec'])} | peak_z={e['peak_z']:.2f}")
    md.append("")
    md.append("## Saídas")
    md.append(f"- Vídeo anotado: `{report['outputs']['annotated_video']}`")
    md.append(f"- JSON: `{report['outputs']['report_json']}`")
    md.append(f"- MD: `{report['outputs']['report_md']}`")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


def with_timestamps(events: list, fps: float) -> list:
    out = []
    for e in events:
        out.append({
            **e,
            "start_time_sec": float(e["start_frame"] / fps),
            "end_time_sec": float(e["end_frame"] / fps),
        })
    return out
