"""YOLO_LOCK serialises GPU passes; the live detector can skip instead of wait."""

from __future__ import annotations

import numpy as np

from assist.perception.detect import YOLO_LOCK, ObjectDetector, iter_boxes
from tests.yolo_fakes import FakeYolo, fake_results


def _frame():
    rgb = np.zeros((100, 100, 3), np.uint8)
    depth = np.full((100, 100), 2.0, np.float32)
    conf = np.full((100, 100), 2, np.uint8)
    return rgb, depth, conf


def _detector(model):
    d = ObjectDetector()
    d._model = model
    d.device = "cpu"
    return d


def test_iter_boxes_parses_results():
    res = fake_results([(1, 2, 30, 40, 0, 0.8)], {0: "door"})
    assert iter_boxes(res) == [(1.0, 2.0, 30.0, 40.0, "door", 0.8)]


def test_iter_boxes_empty():
    assert iter_boxes([]) == []


def test_detect_non_blocking_skips_while_lock_held():
    model = FakeYolo(boxes=[(10, 10, 60, 60, 0, 0.9)])
    model.set_classes(["person"])
    det = _detector(model)
    with YOLO_LOCK:
        assert det.detect(*_frame(), block=False) is None
    assert model.devices == []  # never predicted while the finder held the lock


def test_detect_runs_when_lock_free():
    model = FakeYolo(boxes=[(10, 10, 60, 60, 0, 0.9)])
    model.set_classes(["person"])
    out = _detector(model).detect(*_frame(), block=False)
    assert [o.label for o in out] == ["person"]
    assert not YOLO_LOCK.locked()
