"""Minimal stand-ins for Ultralytics YOLO results, so tests need no weights."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np


class _Arr:
    def __init__(self, values):
        self._v = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._v


class _Scalar:
    def __init__(self, value):
        self._v = value

    def item(self):
        return self._v


def fake_results(boxes, names):
    """boxes: [(x1, y1, x2, y2, cls_id, score)], names: {cls_id: name}."""
    items = [
        SimpleNamespace(
            xyxy=[_Arr([x1, y1, x2, y2])], cls=[_Scalar(c)], conf=[_Scalar(s)]
        )
        for x1, y1, x2, y2, c, s in boxes
    ]
    return [SimpleNamespace(boxes=items, names=names)]


class FakeYolo:
    """Records set_classes / predict; returns canned boxes for the current classes."""

    def __init__(self, boxes=(), delay=0.0, fail_on=()):
        self.boxes = list(boxes)  # [(x1, y1, x2, y2, cls_id, score)]
        self.delay = delay
        self.fail_on = set(fail_on)  # devices whose predict raises
        self.classes = []
        self.set_classes_calls = 0
        self.devices = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def set_classes(self, classes):
        self.classes = list(classes)
        self.set_classes_calls += 1

    def predict(self, img, conf=0.25, verbose=False, imgsz=640, device="cpu"):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.devices.append(device)
            if device in self.fail_on:
                raise RuntimeError(f"predict failed on {device}")
            if self.delay:
                time.sleep(self.delay)
            names = {i: n for i, n in enumerate(self.classes)}
            return fake_results(self.boxes, names)
        finally:
            with self._lock:
                self.active -= 1
