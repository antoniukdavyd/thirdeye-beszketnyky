"""Depth in box prefers near surface; detection hold smooths gaps."""

from __future__ import annotations

import numpy as np

from assist.capture.stream import median_depth_in_box
from assist.perception.detect import DetectedObject, DetectionHold


def test_median_depth_prefers_near_over_far_background():
    h, w = 100, 100
    depth = np.full((h, w), 8.0, dtype=np.float32)
    conf = np.full((h, w), 2, dtype=np.uint8)
    # object occupies left-central part of box
    depth[30:70, 30:55] = 2.0
    z, c = median_depth_in_box(depth, conf, 20, 20, 80, 80)
    assert z is not None
    assert abs(z - 2.0) < 0.5


def test_detection_hold_keeps_last():
    hold = DetectionHold(ttl_frames=3)
    objs = [
        DetectedObject("person", 1.5, "center", 2, (10, 10, 40, 80), 0.9),
    ]
    assert hold.update(objs) == objs
    empty = hold.update([])
    assert len(empty) == 1
    assert empty[0].label == "person"
    hold.update([])
    hold.update([])
    assert hold.update([]) == []
