import numpy as np

USE_TORCH = True
try:
    import torch
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:
    USE_TORCH = False


class ActivityModel:
    """
    Modelo simples de action recognition (R3D-18 / Kinetics-400).
    """
    def __init__(self, device="cpu"):
        if not USE_TORCH:
            raise RuntimeError("Torch/torchvision não disponível.")
        self.device = device

        self.weights = R3D_18_Weights.DEFAULT
        self.categories = self.weights.meta.get("categories", [])

        self.model = r3d_18(weights=self.weights).to(self.device).eval()

        # Normalização Kinetics (RGB em [0,1])
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)

    def predict_top1(self, frames_rgb_112):
        """
        frames_rgb_112: lista com 16 frames RGB 112x112 (uint8)
        retorna (label, confidence)
        """
        if len(frames_rgb_112) < 16:
            return ("unknown", 0.0)

        arr = np.stack(frames_rgb_112[:16], axis=0)  # (T,H,W,C)
        t = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # (1,C,T,H,W)
        t = t.float().to(self.device) / 255.0
        t = (t - self.mean) / self.std

        with torch.no_grad():
            out = self.model(t)
            prob = torch.softmax(out, dim=1)[0]
            conf, idx = torch.max(prob, dim=0)
            label = self.categories[int(idx)] if self.categories else f"class_{int(idx)}"
            return (label, float(conf))


def torch_available():
    return USE_TORCH


def resolve_device(device: str) -> str:
    if not USE_TORCH:
        return "cpu"
    if device == "cuda" and (not torch.cuda.is_available()):
        return "cpu"
    return device
