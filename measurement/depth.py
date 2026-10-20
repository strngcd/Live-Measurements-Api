"""MiDaS monocular depth, kept for later comparison. The API does not use it.

MiDaS returns relative inverse depth (larger means nearer) with an unknown
scale and offset, so its values cannot be read as centimetres or turned into
body depth without a separate calibration.
"""
import cv2
import torch
import torch.nn.functional as F

_depth_model = None


def load_depth_model():
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval()
    return model


def get_depth_model():
    """Load MiDaS on first use (downloads code and weights if not cached)."""
    global _depth_model
    if _depth_model is None:
        _depth_model = load_depth_model()
    return _depth_model


def estimate_depth(image, model=None):
    """Relative inverse depth map (384 x 384) for a BGR image, preprocessed as in the original code.

    That preprocessing does not match MiDaS_small's official transform (256 px,
    aspect ratio kept, ImageNet normalisation); fix it before relying on the output.
    """
    if model is None:
        model = get_depth_model()
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Resize input to match MiDaS model input size
    input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth_map = model(input_tensor)

    return depth_map.squeeze().numpy()
