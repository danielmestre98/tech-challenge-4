import numpy as np
import cv2
from deepface import DeepFace


def deepface_detect_and_emotion(frame_bgr, detector_backend="retinaface"):
    """
    Retorna detecções reais de face + emoção.
    Se não houver face, retorna [] (evita "pessoas fantasmas").
    """
    H, W = frame_bgr.shape[:2]

    # 1) extrai faces + bbox + confidence
    faces = DeepFace.extract_faces(
        img_path=frame_bgr,
        detector_backend=detector_backend,
        enforce_detection=False,  # não explode quando não tem rosto
        align=True
    )

    detections = []
    for f in faces:
        conf = float(f.get("confidence", 0.0))  # extract_faces retorna confidence :contentReference[oaicite:1]{index=1}
        fa = f.get("facial_area", {}) or {}
        x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)

        # 2) FILTRO CRÍTICO:
        # quando não acha rosto e enforce_detection=False, ele gera "base_region" com confidence=0 e bbox = imagem toda :contentReference[oaicite:2]{index=2}
        if conf <= 0.0:
            continue
        if w <= 0 or h <= 0:
            continue
        if (w * h) >= 0.70 * (W * H):  # evita bbox gigante “inventado”
            continue
        if w < 30 or h < 30:  # evita ruído minúsculo
            continue

        face_rgb = f.get("face", None)
        if face_rgb is None:
            continue

        # extract_faces pode devolver face normalizada (float [0,1]) em RGB :contentReference[oaicite:3]{index=3}
        if face_rgb.dtype != np.uint8:
            face_rgb_u8 = np.clip(face_rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            face_rgb_u8 = face_rgb

        face_bgr = cv2.cvtColor(face_rgb_u8, cv2.COLOR_RGB2BGR)

        # 3) roda emoção no crop, sem redetectar (detector_backend="skip" para face já extraída) :contentReference[oaicite:4]{index=4}
        try:
            r = DeepFace.analyze(
                img_path=face_bgr,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True
            )
            if isinstance(r, list):
                r = r[0] if r else {}
        except Exception:
            continue

        detections.append({
            "box": (x, y, w, h),
            "dominant_emotion": r.get("dominant_emotion", "unknown"),
            "emotion": r.get("emotion", {}) or {}
        })

    return detections
