"""On-device scene-text OCR via Apple Vision (ocrmac).

Reads signs, bus route numbers, door labels. VLMs tend to hallucinate / "fix"
route numbers, so exact strings come from OCR here; the LLM only reasons over
them. Import is guarded so logic/tests run on non-Mac / without ocrmac.
"""

from __future__ import annotations

from typing import List

import numpy as np

# Normalized x-center → bearing thresholds (Vision origin is bottom-left, but
# x grows left→right either way, so x alone gives horizontal side).
_LEFT_MAX = 0.34
_RIGHT_MIN = 0.66
# Drop OCR noise below this confidence.
MIN_OCR_CONF = 0.3


def _bearing_from_x(x_center: float) -> str:
    if x_center < _LEFT_MAX:
        return "left"
    if x_center > _RIGHT_MIN:
        return "right"
    return "center"


def recognize_text(rgb_bgr: np.ndarray) -> List[dict]:
    """Return [{text, conf, bearing, bbox}] from Apple Vision OCR.

    Raises if ocrmac/Vision is unavailable; callers degrade gracefully.
    """
    from ocrmac import ocrmac  # lazy: Mac-only dependency
    from PIL import Image

    # ocrmac wants RGB; our frames are BGR (OpenCV).
    rgb = rgb_bgr[:, :, ::-1]
    img = Image.fromarray(np.ascontiguousarray(rgb))
    raw = ocrmac.OCR(img, recognition_level="accurate", unit="line").recognize()

    out: List[dict] = []
    for text, conf, bbox in raw:
        x, y, w, h = bbox  # normalized (0..1)
        out.append(
            {
                "text": text,
                "conf": round(float(conf), 3),
                "bearing": _bearing_from_x(x + w / 2.0),
                "bbox": [round(v, 4) for v in (x, y, w, h)],
            }
        )
    return out
