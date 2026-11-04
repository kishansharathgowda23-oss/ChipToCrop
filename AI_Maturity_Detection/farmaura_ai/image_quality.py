from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageFilter, ImageStat


def assess_image(image_path: Path, *, min_edge: int = 256,
                 min_brightness: float = 40.0, max_brightness: float = 220.0,
                 min_sharpness_variance: float = 60.0) -> dict:
    """Cheap capture gate. Thresholds are heuristics and must be tuned on phone photos."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    gray = image.convert("L")
    brightness = float(ImageStat.Stat(gray).mean[0])
    laplacian = gray.filter(ImageFilter.Kernel(
        (3, 3), (-1, -1, -1, -1, 8, -1, -1, -1, -1), scale=1, offset=128
    ))
    sharpness_variance = float(ImageStat.Stat(laplacian).var[0])
    flags = []
    if min(width, height) < min_edge:
        flags.append("low_resolution")
    if brightness < min_brightness:
        flags.append("low_light")
    elif brightness > max_brightness:
        flags.append("overexposed")
    if sharpness_variance < min_sharpness_variance:
        flags.append("blurry")
    return {
        "width": width,
        "height": height,
        "brightness_mean_0_255": round(brightness, 2),
        "sharpness_laplacian_variance": round(sharpness_variance, 2),
        "flags": flags,
        "usable": not flags,
        "method": "heuristic_v1",
    }
