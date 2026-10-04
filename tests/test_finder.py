"""On-demand focused finder: vocabulary, selection, depth, locking, fallback."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from assist.perception.detect import YOLO_LOCK
from assist.perception.find import (
    FIND_VOCAB,
    FindHit,
    FinderUnavailable,
    ObjectFinder,
    prompts_for,
    select_hit,
)
from tests.yolo_fakes import FakeYolo


def _frame(depth_m=2.0, w=200, h=100):
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), depth_m, np.float32)
    conf = np.full((h, w), 2, np.uint8)
    return rgb, depth, conf


def _finder(model, device="cpu"):
    f = ObjectFinder(model_name="fake.pt", imgsz=1280)
    f._model = model
    f.device = device
    return f


def _cls(name):
    return FIND_VOCAB.index(name)


def _hit(score, dist, prompt="door"):
    return FindHit("door", prompt, score, dist, "center", (0, 0, 1, 1))


def test_prompts_for_synonyms_and_passthrough():
    assert prompts_for("door") == ["door", "doorway", "glass door", "entrance"]
    assert prompts_for("person")[0] == "person"
    assert prompts_for("  Trash   Can ") == ["trash can"]
    assert prompts_for("") == []


def test_select_hit_prefers_nearest_strong():
    pick = select_hit([_hit(0.9, 4.0), _hit(0.5, 1.5), _hit(0.2, 0.5)])
    assert pick.dist_m == 1.5


def test_select_hit_strong_without_distance_uses_score():
    pick = select_hit([_hit(0.5, None), _hit(0.8, None), _hit(0.2, 1.0)])
    assert pick.score == 0.8


def test_select_hit_only_weak_takes_best_score():
    pick = select_hit([_hit(0.12, 1.0), _hit(0.3, 3.0)])
    assert pick.score == 0.3 and not pick.certain


def test_select_hit_empty():
    assert select_hit([]) is None


def test_find_builds_hits_with_depth_and_bearing():
    model = FakeYolo(boxes=[(10, 10, 50, 90, _cls("glass door"), 0.6)])
    hits = _finder(model).find(*_frame(depth_m=2.0), "door")
    assert len(hits) == 1
    h = hits[0]
    assert h.label == "door" and h.prompt == "glass door"
    assert h.dist_m == pytest.approx(2.0)
    assert h.bearing == "left" and h.certain


def test_find_keeps_hit_without_valid_depth():
    model = FakeYolo(boxes=[(100, 10, 180, 90, _cls("door"), 0.2)])
    hits = _finder(model).find(*_frame(depth_m=0.0), "door")
    assert hits[0].dist_m is None and not hits[0].certain


def test_find_sorted_by_score():
    model = FakeYolo(boxes=[(0, 0, 20, 20, _cls("door"), 0.2), (30, 0, 60, 20, _cls("door"), 0.7)])
    hits = _finder(model).find(*_frame(), "door")
    assert [h.score for h in hits] == [0.7, 0.2]


def test_vocabulary_set_once_for_known_targets():
    # Any set_classes call makes the next MPS predict ~1 s slower (measured
    # 2026-10-04), so switching door <-> person must not touch it.
    model = FakeYolo()
    f = _finder(model)
    for target in ("door", "person", "door", "stairs", "car"):
        f.find(*_frame(), target)
    assert model.set_classes_calls == 1
    assert "wall" not in model.classes


def test_unknown_target_extends_vocabulary_once():
    model = FakeYolo()
    f = _finder(model)
    f.find(*_frame(), "door")
    f.find(*_frame(), "trash can")
    f.find(*_frame(), "trash can")
    f.find(*_frame(), "door")
    assert model.set_classes_calls == 2
    assert "trash can" in model.classes and "door" in model.classes


def test_hits_filtered_to_target_synonyms():
    model = FakeYolo(
        boxes=[(0, 0, 20, 20, _cls("person"), 0.9), (30, 0, 60, 20, _cls("doorway"), 0.4)]
    )
    hits = _finder(model).find(*_frame(), "door")
    assert [h.prompt for h in hits] == ["doorway"]


def test_find_holds_yolo_lock_during_predict():
    seen = []

    class Probe(FakeYolo):
        def predict(self, *a, **k):
            seen.append(YOLO_LOCK.locked())
            return super().predict(*a, **k)

    _finder(Probe()).find(*_frame(), "door")
    assert seen == [True]
    assert not YOLO_LOCK.locked()


def test_concurrent_finds_are_serialised():
    model = FakeYolo(boxes=[(0, 0, 20, 20, _cls("door"), 0.9)], delay=0.05)
    f = _finder(model)
    threads = [
        threading.Thread(target=f.find, args=(*_frame(), "door")) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(2.0)
    assert model.max_active == 1
    assert len(model.devices) == 2


def test_mps_failure_falls_back_to_cpu_once():
    model = FakeYolo(boxes=[(0, 0, 20, 20, _cls("door"), 0.9)], fail_on={"mps"})
    f = _finder(model, device="mps")
    assert len(f.find(*_frame(), "door")) == 1
    f.find(*_frame(), "door")
    assert model.devices == ["mps", "cpu", "cpu"]


def test_load_failure_raises_once_without_retrying(monkeypatch):
    calls = []

    def broken_yolo(name):
        calls.append(name)
        raise OSError("download failed")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", broken_yolo)
    f = ObjectFinder(model_name="missing.pt")
    with pytest.raises(FinderUnavailable):
        f.find(*_frame(), "door")
    with pytest.raises(FinderUnavailable):
        f.find(*_frame(), "door")
    assert calls == ["missing.pt"]
    assert not YOLO_LOCK.locked()


def test_env_overrides_model_and_imgsz(monkeypatch):
    monkeypatch.setenv("FIND_MODEL", "yolov8m-worldv2.pt")
    monkeypatch.setenv("FIND_IMGSZ", "960")
    f = ObjectFinder()
    assert f.model_name == "yolov8m-worldv2.pt" and f.imgsz == 960


def test_defaults(monkeypatch):
    monkeypatch.delenv("FIND_MODEL", raising=False)
    monkeypatch.delenv("FIND_IMGSZ", raising=False)
    f = ObjectFinder()
    assert f.model_name == "yolov8l-worldv2.pt" and f.imgsz == 1280


def test_model_load_does_not_hold_yolo_lock(monkeypatch):
    # A first-run download must not stall live labeling.
    seen = []

    def loader(name):
        seen.append(YOLO_LOCK.locked())
        return FakeYolo()

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", loader)
    ObjectFinder(model_name="x.pt").find(*_frame(), "door")
    assert seen == [False]


def test_bad_imgsz_env_falls_back(monkeypatch):
    monkeypatch.setenv("FIND_IMGSZ", "abc")
    assert ObjectFinder().imgsz == 1280
