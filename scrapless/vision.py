from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

@dataclass
class VisionAnalysis:
    image: Image.Image
    overlay: Image.Image
    detected_regions: int
    estimated_foreground_fraction: float
    note: str
    regions: list[dict]

def analyze_material_image(image: Image.Image, min_area: float = 500.0) -> VisionAnalysis:
    if image is None:
        raise ValueError("An image is required.")
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if height < 20 or width < 20: raise ValueError("Image too small.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2GRAY)
    _, thresh_a = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_b = cv2.bitwise_not(thresh_a)

    def score(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= min_area]
        return len(areas), sum(areas) / float(width * height), contours

    sa, sb = score(thresh_a), score(thresh_b)
    cands = [sa, sb]
    plaus = [c for c in cands if 0.02 <= c[1] <= 0.95]
    _, coverage, contours = min(plaus, key=lambda c: abs(c[1] - 0.5)) if plaus else max(cands, key=lambda c: c[1])

    overlay_bgr = bgr.copy()
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    regions = []
    cv2.drawContours(overlay_bgr, valid_contours, -1, (0, 180, 150), 3)
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay_bgr, (x, y), (x + w, y + h), (220, 120, 30), 2)
        regions.append({"x_px": x, "y_px": y, "width_px": w, "height_px": h,
                        "contour_area_px2": float(cv2.contourArea(c))})

    return VisionAnalysis(
        image=image.convert("RGB"), overlay=Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)),
        detected_regions=len(valid_contours), estimated_foreground_fraction=float(min(max(coverage, 0.0), 1.0)),
        note="Basic threshold demo. Not a trained detector. Lighting, shadows, touching parts, and perspective can reduce accuracy.",
        regions=regions,
    )
