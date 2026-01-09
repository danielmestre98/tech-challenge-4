from dataclasses import dataclass
import argparse

@dataclass
class AppConfig:
    input: str
    outdir: str
    detector: str
    detect_every: int
    max_track_age: int

    no_activity: bool
    activity_every_sec: float
    device: str

    anom_z: float


def parse_args() -> AppConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="caminho do vídeo de entrada")
    ap.add_argument("--outdir", default="outputs", help="pasta de saída")
    ap.add_argument("--detector", default="retinaface", help="backend do DeepFace (retinaface, opencv, mtcnn, mediapipe...)")

    ap.add_argument("--detect_every", type=int, default=5, help="rodar DeepFace (detecção+emoção) a cada N frames")
    ap.add_argument("--max_track_age", type=int, default=30, help="frames tolerados sem update de tracker")

    ap.add_argument("--no-activity", action="store_true", help="desliga atividade (torchvision)")
    ap.add_argument("--activity_every_sec", type=float, default=1.0, help="inferencia de atividade a cada X segundos")
    ap.add_argument("--device", default="cpu", help="cpu ou cuda (se disponível)")

    ap.add_argument("--anom_z", type=float, default=3.0, help="limiar z-score de anomalia")
    args = ap.parse_args()

    return AppConfig(
        input=args.input,
        outdir=args.outdir,
        detector=args.detector,
        detect_every=args.detect_every,
        max_track_age=args.max_track_age,
        no_activity=args.no_activity,
        activity_every_sec=args.activity_every_sec,
        device=args.device,
        anom_z=args.anom_z,
    )
