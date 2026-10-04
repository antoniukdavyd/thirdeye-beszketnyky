# find_object Fresh Focused Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `find_object` is triggered (keys `p`/`c`/`f` or the voice tool), run a new focused `yolov8l-worldv2` pass on the newest frame instead of reading the last background labels, and speak weak hits with doubt.

**Architecture:** A new `ObjectFinder` (own YOLO-World instance, per-target synonym vocabulary, `imgsz=1280`, `conf ≥ 0.10`) is injected into `ToolRegistry` like the OCR callable. A module-level `YOLO_LOCK` serialises every YOLO predict; the finder holds it for its whole pass and the background detector uses a non-blocking acquire, so live labeling skips its passes while a find runs. `SceneStore` also carries the newest depth + confidence so the finder can measure meters.

**Tech Stack:** Python 3.12, Ultralytics 8.4 YOLO-World, PyTorch MPS, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-04-find-object-fresh-pass-design.md`

## Global Constraints

- Run everything with the repo venv: `.venv/bin/python -m pytest …` from `/Users/petromelnyk/Downloads/nekit`.
- **No git commits.** The user approves before anything is committed; each task ends with a checkpoint, not a commit.
- Finder model default `yolov8l-worldv2.pt` (`FIND_MODEL`), `imgsz` default `1280` (`FIND_IMGSZ`), score floor `0.10`, strong hit `score ≥ 0.35`.
- Spoken output is English. A weak hit is never spoken as certain ("Possibly …"); a missing distance is "distance unknown", never a guessed number.
- With `--no-yolo`, or a snapshot without depth, `find_object` behaves exactly as before (stored labels).
- Proximity beeps (zones) must not depend on the finder.

## Review Focus

1. Finder weights missing and no internet → the first load fails once, later finds do not retry the download, and every find still answers from stored labels.
2. Voice tool and a key press trigger find at the same moment → passes run one after the other on the GPU, both answer.
3. MPS predict error mid-session → finder retries on CPU once and keeps working (slower).
4. Messy or aliased target text ("  Trash Can ", "дверь", "people") → sensible vocabulary ("trash can"; door synonyms; person synonyms).
5. Wording for plural / empty targets → "I don't see stairs." (no "a stairs"), empty label → "label required" error.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `assist/perception/detect.py` (modify) | `YOLO_LOCK`, `iter_boxes()` shared box parsing, `ObjectDetector.detect(..., block=)` |
| `assist/perception/find.py` (create) | `ObjectFinder`, `FindHit`, `FinderUnavailable`, `prompts_for()`, `select_hit()` |
| `assist/perception/scene_store.py` (modify) | depth / conf in publish + snapshot |
| `assist/perception/__init__.py` (modify) | export finder names |
| `assist/tools/__init__.py` (modify) | `finder` dependency, fresh vs stored find path, wording, tool description |
| `assist/agent/__init__.py` (modify) | agent prompt: "possibly" / "distance unknown" |
| `assist/app.py` (modify) | worker skip on `None`, create + warm finder, publish depth/conf |
| `.env.example`, `README.md` (modify) | `FIND_MODEL`, `FIND_IMGSZ` |
| `tests/yolo_fakes.py` (create) | fake Ultralytics results / model shared by tests |
| `tests/test_yolo_lock.py` (create) | lock + `detect(block=False)` |
| `tests/test_finder.py` (create) | finder unit tests |
| `tests/test_find_object_fresh.py` (create) | tool-level tests |
| `tests/test_detect_pacing.py` (modify) | fake accepts `block`, skip + finder warmup tests |
| `tests/test_tools.py` (modify) | SceneStore depth round-trip |

---

### Task 1: YOLO_LOCK, shared box parsing, non-blocking detect

**Files:**
- Modify: `assist/perception/detect.py`
- Modify: `assist/app.py` (`_detect_watch`)
- Create: `tests/yolo_fakes.py`, `tests/test_yolo_lock.py`
- Modify: `tests/test_detect_pacing.py`

**Interfaces:**
- Produces: `YOLO_LOCK: threading.Lock`; `iter_boxes(results) -> list[tuple[float, float, float, float, str, float]]` (x1, y1, x2, y2, class name, score); `ObjectDetector.detect(rgb_bgr, depth, conf_map, block: bool = True) -> Optional[list[DetectedObject]]` (None only when `block=False` and the lock is busy).
- Produces (tests): `tests/yolo_fakes.py` with `fake_results(boxes, names)` and `FakeYolo`.

- [ ] **Step 1: Write the shared fakes**

`tests/yolo_fakes.py`:
```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_yolo_lock.py`:
```python
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
```

In `tests/test_detect_pacing.py`, change `FakeDetector.detect` to accept `block` and add a skip switch, and add the skip test:
```python
class FakeDetector:
    def __init__(self):
        self.seen = []
        self.done = threading.Event()
        self.skip = False

    def warmup(self):
        pass

    def detect(self, rgb, depth, conf, block=True):
        self.seen.append((rgb, time.monotonic()))
        self.done.set()
        if self.skip:
            return None
        return []
```
```python
def test_skipped_pass_keeps_objects_and_is_not_counted():
    app = worker_app(interval=0.05)
    counted = []
    app.framelog = SimpleNamespace(detected=lambda ms: counted.append(ms))
    app.objects = ["kept"]
    app.detector.skip = True
    app._start_detect_watch()
    try:
        app._submit_detect("A", None, None)
        assert app.detector.done.wait(1.0)
        time.sleep(0.05)
    finally:
        app._stop_detect_watch()
    assert app.objects == ["kept"]
    assert counted == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_yolo_lock.py tests/test_detect_pacing.py -q`
Expected: FAIL — `ImportError: cannot import name 'YOLO_LOCK'`, and the skip test fails (`objects` becomes `[]`).

- [ ] **Step 4: Implement in `assist/perception/detect.py`**

Add `import threading` and, after `DEFAULT_CLASSES`:
```python
# One YOLO pass on the GPU at a time. The live detector and the on-demand
# finder each have their own model; running both on MPS at once is not safe
# to rely on. The finder holds this for its whole pass, which also pauses
# live labeling (the detector skips instead of waiting).
YOLO_LOCK = threading.Lock()


def iter_boxes(results) -> List[tuple]:
    """(x1, y1, x2, y2, class_name, score) for each box of an Ultralytics result."""
    if not results:
        return []
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    names = r0.names
    out = []
    for box in r0.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = (
            names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id])
        )
        out.append((x1, y1, x2, y2, label, score))
    return out
```

Replace `ObjectDetector.detect` with:
```python
    def detect(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        conf_map: np.ndarray,
        block: bool = True,
    ) -> Optional[List[DetectedObject]]:
        """Label one frame. ``block=False`` returns None at once if another
        YOLO pass (the finder) holds YOLO_LOCK."""
        if not YOLO_LOCK.acquire(blocking=block):
            return None
        try:
            self._ensure_model()
            if self._model is None:
                return []
            try:
                boxes = iter_boxes(self._predict(rgb_bgr))
            except Exception as e:
                log_exc("detect", "YOLO predict failed", e)
                return []
        finally:
            YOLO_LOCK.release()

        w = rgb_bgr.shape[1]
        out: List[DetectedObject] = []
        for x1, y1, x2, y2, label, score in boxes:
            z, cmed = median_depth_in_box(depth, conf_map, x1, y1, x2, y2)
            if z is None:
                continue
            out.append(
                DetectedObject(
                    label=label,
                    dist_m=z,
                    bearing=bearing_from_cx(0.5 * (x1 + x2), w),
                    conf_depth=cmed,
                    box=(int(x1), int(y1), int(x2), int(y2)),
                    score=score,
                )
            )

        # Keep more detections; no class-type filtering. LLM ranks importance.
        out.sort(key=lambda o: o.dist_m)
        return out[:15]
```

In `assist/app.py` `_detect_watch`, replace the body of the `try:` that runs the pass:
```python
            try:
                fresh = self.detector.detect(*job, block=False)
                if fresh is None:
                    # The finder holds the GPU: live labeling sits this pass
                    # out and the current labels stay.
                    continue
                self.objects = self.detection_hold.update(fresh)
                self.framelog.detected((time.monotonic() - last_start) * 1000.0)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (previous 112 + new).

- [ ] **Step 6: Checkpoint (no commit — user approval gate)**

---

### Task 2: SceneStore carries depth + confidence

**Files:**
- Modify: `assist/perception/scene_store.py`
- Modify: `assist/app.py` (`_frame_step` publish line)
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `SceneStore.publish(rgb_bgr, scene, objects=None, depth=None, conf=None)`; `SceneSnapshot.depth: Optional[np.ndarray]`, `SceneSnapshot.conf: Optional[np.ndarray]`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_tools.py`)
```python
def test_scene_store_keeps_depth_and_conf():
    s = SceneStore()
    rgb = np.zeros((4, 4, 3), np.uint8)
    depth = np.ones((4, 4), np.float32)
    conf = np.full((4, 4), 2, np.uint8)
    s.publish(rgb, {"zones_m": {}}, objects=[], depth=depth, conf=conf)
    snap = s.snapshot(copy_rgb=False)
    assert snap.depth is depth and snap.conf is conf


def test_scene_store_depth_defaults_to_none(store):
    snap = store.snapshot()
    assert snap.depth is None and snap.conf is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tools.py -q`
Expected: FAIL — `publish() got an unexpected keyword argument 'depth'`.

- [ ] **Step 3: Implement**

`assist/perception/scene_store.py`: add to `SceneSnapshot` (after `updated_at`):
```python
    depth: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None
```
In `SceneStore.__init__` add `self._depth = None` and `self._conf = None`. Change `publish`:
```python
    def publish(
        self,
        rgb_bgr: np.ndarray,
        scene: dict,
        objects: Optional[List[DetectedObject]] = None,
        depth: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            self._rgb = rgb_bgr
            self._scene = dict(scene) if scene else {}
            self._objects = list(objects or [])
            # References, not copies: the loop builds new arrays every frame
            # and never writes into a published one.
            self._depth = depth
            self._conf = conf
            self._updated_at = time.monotonic()
```
In `snapshot`, pass `depth=self._depth, conf=self._conf` to `SceneSnapshot(...)`.

`assist/app.py` `_frame_step`:
```python
        self.store.publish(
            frame.rgb_bgr, self.scene, objects, depth=frame.depth, conf=frame.conf
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 3: ObjectFinder

**Files:**
- Create: `assist/perception/find.py`
- Modify: `assist/perception/__init__.py`
- Test: `tests/test_finder.py`

**Interfaces:**
- Consumes: `YOLO_LOCK`, `iter_boxes`, `bearing_from_cx`, `yolo_device` (Task 1); `median_depth_in_box` (`assist/capture`).
- Produces: `FindHit(label, prompt, score, dist_m: Optional[float], bearing, box)` with property `certain`; `FinderUnavailable(RuntimeError)`; `prompts_for(target: str) -> list[str]`; `select_hit(hits: list[FindHit]) -> Optional[FindHit]`; `ObjectFinder(model_name=None, imgsz=None, conf=0.10)` with `find(rgb_bgr, depth, conf_map, target) -> list[FindHit]` (sorted by score desc) and `warmup()`; constants `STRONG_SCORE = 0.35`, `FIND_MIN_SCORE = 0.10`.

- [ ] **Step 1: Write the failing tests**

`tests/test_finder.py`:
```python
"""On-demand focused finder: vocabulary, selection, depth, locking, fallback."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from assist.perception.detect import YOLO_LOCK
from assist.perception.find import (
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
    model = FakeYolo(boxes=[(10, 10, 50, 90, 2, 0.6)])  # cls 2 = "glass door"
    hits = _finder(model).find(*_frame(depth_m=2.0), "door")
    assert len(hits) == 1
    h = hits[0]
    assert h.label == "door" and h.prompt == "glass door"
    assert h.dist_m == pytest.approx(2.0)
    assert h.bearing == "left" and h.certain


def test_find_keeps_hit_without_valid_depth():
    model = FakeYolo(boxes=[(100, 10, 180, 90, 0, 0.2)])
    hits = _finder(model).find(*_frame(depth_m=0.0), "door")
    assert hits[0].dist_m is None and not hits[0].certain


def test_find_sorted_by_score():
    model = FakeYolo(boxes=[(0, 0, 20, 20, 0, 0.2), (30, 0, 60, 20, 0, 0.7)])
    hits = _finder(model).find(*_frame(), "door")
    assert [h.score for h in hits] == [0.7, 0.2]


def test_set_classes_only_when_target_changes():
    model = FakeYolo()
    f = _finder(model)
    f.find(*_frame(), "door")
    f.find(*_frame(), "door")
    f.find(*_frame(), "person")
    assert model.set_classes_calls == 2


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
    model = FakeYolo(boxes=[(0, 0, 20, 20, 0, 0.9)], delay=0.05)
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
    model = FakeYolo(boxes=[(0, 0, 20, 20, 0, 0.9)], fail_on={"mps"})
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_finder.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'assist.perception.find'`.

- [ ] **Step 3: Implement `assist/perception/find.py`**
```python
"""On-demand focused detection for find_object.

The live detector runs a small model over 17 classes; doors and people it
scores low are simply missing from its list. The finder runs only when asked,
with a bigger model, higher resolution, a low score floor and a vocabulary of
just the target and its synonyms (YOLO-World scores each prompt, so a doorway
or glass door can match where "door" alone would not).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..capture import median_depth_in_box
from ..debuglog import log, log_exc
from .detect import YOLO_LOCK, bearing_from_cx, iter_boxes, yolo_device

DEFAULT_FIND_MODEL = "yolov8l-worldv2.pt"
DEFAULT_FIND_IMGSZ = 1280
FIND_MIN_SCORE = 0.10
# Hits at or above this are spoken as certain; below as "possibly".
STRONG_SCORE = 0.35

FIND_SYNONYMS = {
    "door": ("door", "doorway", "glass door", "entrance"),
    "person": ("person", "pedestrian", "child"),
    "stairs": ("stairs", "staircase", "steps"),
}


class FinderUnavailable(RuntimeError):
    """The finder model could not be loaded; callers fall back to stored labels."""


@dataclass
class FindHit:
    label: str
    prompt: str
    score: float
    dist_m: Optional[float]
    bearing: str
    box: tuple

    @property
    def certain(self) -> bool:
        return self.score >= STRONG_SCORE


def prompts_for(target: str) -> List[str]:
    t = " ".join((target or "").lower().split())
    if not t:
        return []
    return list(FIND_SYNONYMS.get(t, (t,)))


def select_hit(hits: List[FindHit]) -> Optional[FindHit]:
    """Nearest strong hit with a distance, else the best strong, else the best weak."""
    strong = [h for h in hits if h.certain]
    if strong:
        measured = [h for h in strong if h.dist_m is not None]
        if measured:
            return min(measured, key=lambda h: h.dist_m)
        return max(strong, key=lambda h: h.score)
    return max(hits, key=lambda h: h.score, default=None)


class ObjectFinder:
    """Own YOLO-World instance; safe to construct without weights or GPU."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        imgsz: Optional[int] = None,
        conf: float = FIND_MIN_SCORE,
    ):
        self.model_name = model_name or os.getenv("FIND_MODEL") or DEFAULT_FIND_MODEL
        self.imgsz = imgsz or int(os.getenv("FIND_IMGSZ") or DEFAULT_FIND_IMGSZ)
        self.conf = conf
        self.device = yolo_device()
        self._model = None
        self._failed = False
        self._prompts: Optional[List[str]] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self._failed:
            raise FinderUnavailable(self.model_name)
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            log("find", "finder model loaded", model=self.model_name, device=self.device, imgsz=self.imgsz)
        except Exception as exc:
            # One attempt only: retrying a failed download on every find
            # would stall each answer.
            self._failed = True
            log_exc("find", "finder model unavailable", exc)
            raise FinderUnavailable(self.model_name) from exc

    def _predict(self, rgb_bgr: np.ndarray):
        try:
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device
            )
        except Exception as exc:
            if self.device == "cpu":
                raise
            log_exc("find", f"finder on {self.device} failed, falling back to cpu", exc)
            self.device = "cpu"
            return self._model.predict(
                rgb_bgr, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device
            )

    def find(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        conf_map: np.ndarray,
        target: str,
    ) -> List[FindHit]:
        prompts = prompts_for(target)
        if not prompts:
            return []
        label = " ".join(target.lower().split())
        # Held for the whole pass: live labeling skips while we run.
        with YOLO_LOCK:
            self._ensure_model()
            if prompts != self._prompts:
                self._model.set_classes(prompts)
                self._prompts = prompts
            boxes = iter_boxes(self._predict(rgb_bgr))

        w = rgb_bgr.shape[1]
        hits = []
        for x1, y1, x2, y2, prompt, score in boxes:
            z, _ = median_depth_in_box(depth, conf_map, x1, y1, x2, y2)
            hits.append(
                FindHit(
                    label=label,
                    prompt=prompt,
                    score=score,
                    dist_m=z,
                    bearing=bearing_from_cx(0.5 * (x1 + x2), w),
                    box=(int(x1), int(y1), int(x2), int(y2)),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    def warmup(self, shape: tuple = (1920, 1440, 3)) -> None:
        """Load + one pass so the first real find skips load and MPS compile."""
        dummy = np.zeros(shape, dtype=np.uint8)
        depth = np.ones(shape[:2], dtype=np.float32)
        conf = np.full(shape[:2], 2, dtype=np.uint8)
        self.find(dummy, depth, conf, "door")
```

`assist/perception/__init__.py`: add `from .find import FindHit, FinderUnavailable, ObjectFinder` and the three names to `__all__`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_finder.py -q` then `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 4: find_object uses the finder (fresh path, wording, fallback)

**Files:**
- Modify: `assist/tools/__init__.py`
- Modify: `assist/agent/__init__.py`
- Test: `tests/test_find_object_fresh.py`

**Interfaces:**
- Consumes: `SceneSnapshot.depth/conf` (Task 2); `ObjectFinder.find`, `select_hit`, `FindHit.certain` (Task 3).
- Produces: `ToolRegistry(store, llm=None, ocr=None, finder=None)`; `find_object` result keys `ok, found, label, prompt, certain, score, bearing, dist_m, source, age_ms` (fresh) or the existing keys plus `source="stored"`, `certain=True` (stored).

- [ ] **Step 1: Write the failing tests**

`tests/test_find_object_fresh.py`:
```python
"""find_object runs the finder on the newest frame and words weak hits with doubt."""

from __future__ import annotations

import numpy as np

from assist.perception.find import FindHit, FinderUnavailable
from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry


class FakeFinder:
    def __init__(self, hits=(), error=None):
        self.hits = list(hits)
        self.error = error
        self.calls = []

    def find(self, rgb, depth, conf, target):
        self.calls.append(target)
        if self.error:
            raise self.error
        return list(self.hits)


STORED_SCENE = {
    "zones_m": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "door", "dist_m": 3.0, "bearing": "right", "conf_depth": 2, "score": 0.5}],
    "hint": "",
}


def _store(with_depth=True):
    s = SceneStore()
    rgb = np.zeros((8, 8, 3), np.uint8)
    depth = np.ones((8, 8), np.float32) if with_depth else None
    conf = np.full((8, 8), 2, np.uint8) if with_depth else None
    s.publish(rgb, STORED_SCENE, objects=[], depth=depth, conf=conf)
    return s


def _reg(finder, with_depth=True):
    return ToolRegistry(_store(with_depth), ocr=lambda rgb: [], finder=finder)


def _hit(score, dist, bearing="left", prompt="doorway"):
    return FindHit("door", prompt, score, dist, bearing, (0, 0, 1, 1))


def test_fresh_certain_hit():
    reg = _reg(FakeFinder([_hit(0.8, 2.1)]))
    out = reg.execute("find_object", {"label": "door"})
    assert out["source"] == "fresh" and out["found"] and out["certain"]
    assert out["dist_m"] == 2.1 and out["prompt"] == "doorway"
    assert reg.spoken_from_tool("find_object", out) == (
        "In front of you a door: on the left, about 2.1 meters."
    )


def test_fresh_weak_hit_says_possibly():
    reg = _reg(FakeFinder([_hit(0.2, 2.1)]))
    out = reg.execute("find_object", {"label": "door"})
    assert out["certain"] is False
    assert reg.spoken_from_tool("find_object", out).startswith("Possibly a door")


def test_fresh_hit_without_distance():
    reg = _reg(FakeFinder([_hit(0.8, None)]))
    out = reg.execute("find_object", {"label": "door"})
    assert out["dist_m"] is None
    assert "distance unknown" in reg.spoken_from_tool("find_object", out)


def test_fresh_not_found():
    reg = _reg(FakeFinder([]))
    out = reg.execute("find_object", {"label": "door"})
    assert out["found"] is False and out["source"] == "fresh"
    assert reg.spoken_from_tool("find_object", out) == "I don't see a door."


def test_alias_is_canonicalised_before_finder():
    finder = FakeFinder([])
    _reg(finder).execute("find_object", {"label": "дверь"})
    _reg(finder).execute("find_object", {"label": "people"})
    assert finder.calls == ["door", "person"]


def test_finder_error_falls_back_to_stored_labels():
    reg = _reg(FakeFinder(error=FinderUnavailable("x")))
    out = reg.execute("find_object", {"label": "door"})
    assert out["source"] == "stored" and out["found"]
    assert out["dist_m"] == 3.0


def test_no_depth_uses_stored_labels():
    finder = FakeFinder([_hit(0.9, 1.0)])
    out = _reg(finder, with_depth=False).execute("find_object", {"label": "door"})
    assert out["source"] == "stored" and finder.calls == []


def test_plural_target_wording():
    reg = _reg(FakeFinder([]))
    out = reg.execute("find_object", {"label": "stairs"})
    assert reg.spoken_from_tool("find_object", out) == "I don't see stairs."


def test_empty_label_is_an_error():
    out = _reg(FakeFinder([])).execute("find_object", {"label": "  "})
    assert out == {"ok": False, "error": "label required"}


def test_agent_prompt_mentions_possibly_and_unknown_distance():
    from assist.agent import SYSTEM_INSTRUCTION

    assert "possibly" in SYSTEM_INSTRUCTION
    assert "distance is unknown" in SYSTEM_INSTRUCTION
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_find_object_fresh.py -q`
Expected: FAIL — `TypeError: ToolRegistry.__init__() got an unexpected keyword argument 'finder'`.

- [ ] **Step 3: Implement in `assist/tools/__init__.py`**

Imports: add `from ..perception.find import select_hit`.

Helper near `_bearing_en`:
```python
_NO_ARTICLE = {"stairs", "steps", "people"}


def _with_article(word: str) -> str:
    w = (word or "").strip()
    if not w:
        return "it"
    if w in _NO_ARTICLE:
        return w
    return ("an " if w[0] in "aeiou" else "a ") + w
```

`ToolRegistry.__init__` gains `finder: Optional[Any] = None` and stores `self._finder = finder` (comment: "On-demand ObjectFinder; None → find uses stored labels.").

`find_object` declaration description:
```python
                "description": (
                    "Run a fresh, focused detection on the current camera frame "
                    "for one object (door, person, car, or any other object word). "
                    "Returns side, meters (dist_m null if unknown) and certain=false "
                    "for a weak hit."
                ),
```

Replace `_find_object` with:
```python
    def _find_object(self, args: dict) -> dict:
        label = (args.get("label") or "").strip()
        if not label:
            return {"ok": False, "error": "label required"}
        snap = self.store.snapshot(copy_rgb=False)
        if not snap.has_frame:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        if self._finder is not None and snap.depth is not None:
            try:
                return self._find_fresh(snap, label)
            except Exception as exc:
                log("find", f"finder failed: {type(exc).__name__}: {exc}")
        return self._find_stored(snap, label)

    def _find_fresh(self, snap, label: str) -> dict:
        target = _match_labels(label)[0]
        t0 = time.monotonic()
        hits = self._finder.find(snap.rgb_bgr, snap.depth, snap.conf, target)
        hit = select_hit(hits)
        log(
            "find",
            "pass",
            target=target,
            ms=round((time.monotonic() - t0) * 1000.0),
            hits=len(hits),
            best=(
                f"{hit.prompt} {hit.score:.2f} "
                f"{'-' if hit.dist_m is None else f'{hit.dist_m:.1f}m'}"
                if hit
                else "-"
            ),
        )
        out = {"ok": True, "label": target, "source": "fresh", "age_ms": round(snap.age_ms, 1)}
        if hit is None:
            return {**out, "found": False}
        return {
            **out,
            "found": True,
            "prompt": hit.prompt,
            "certain": hit.certain,
            "score": round(hit.score, 2),
            "bearing": hit.bearing,
            "dist_m": None if hit.dist_m is None else round(hit.dist_m, 2),
        }
```
Rename the old body (from `data = compact_scene_for_llm(snap.scene)` on) to `_find_stored(self, snap, label: str) -> dict`, adding `"source": "stored"` to both returns and `"certain": True` to the found return.

Replace the `find_object` branch of `spoken_from_tool`:
```python
        if name == "find_object":
            target = result.get("label") or ""
            if not result.get("found"):
                return f"I don't see {_with_article(target)}."
            lab = result.get("label_en") or _with_article(target)
            where = _bearing_en(result.get("bearing", ""))
            dist = result.get("dist_m")
            dist_txt = "distance unknown" if dist is None else f"about {dist} meters"
            if result.get("certain", True):
                return f"In front of you {lab}: {where}, {dist_txt}."
            return f"Possibly {lab}: {where}, {dist_txt}."
```

`assist/agent/__init__.py` line 20 becomes:
```
- “where is the door / person / car / bus” → find_object; if not found, say so and suggest turning — do not guess. If certain is false say “possibly”; if dist_m is null say the distance is unknown.
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass, including the untouched `tests/test_tools.py` find tests (stored path).

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 5: Wire the finder into AssistApp + config docs

**Files:**
- Modify: `assist/app.py`
- Modify: `.env.example`, `README.md`
- Test: `tests/test_detect_pacing.py`

**Interfaces:**
- Consumes: `ObjectFinder` (Task 3), `ToolRegistry(..., finder=)` (Task 4).
- Produces: `AssistApp.finder: Optional[ObjectFinder]`; finder warmed in `_detect_watch` before live labeling starts.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_detect_pacing.py`; also add `app.finder = None` inside `worker_app`)
```python
def test_finder_warmed_before_live_labeling():
    order = []
    app = worker_app(interval=0.05)
    app.detector.warmup = lambda: order.append("detector")
    app.finder = SimpleNamespace(warmup=lambda: order.append("finder"))
    original = app.detector.detect

    def detect(*a, **k):
        order.append("pass")
        return original(*a, **k)

    app.detector.detect = detect
    app._start_detect_watch()
    try:
        app._submit_detect("A", None, None)
        assert app.detector.done.wait(1.0)
    finally:
        app._stop_detect_watch()
    assert order[:3] == ["detector", "finder", "pass"]


def test_finder_warmup_failure_does_not_stop_labeling():
    app = worker_app(interval=0.05)

    def boom():
        raise RuntimeError("no weights")

    app.finder = SimpleNamespace(warmup=boom)
    app._start_detect_watch()
    try:
        app._submit_detect("A", None, None)
        assert app.detector.done.wait(1.0)
    finally:
        app._stop_detect_watch()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_detect_pacing.py -q`
Expected: FAIL — order lacks `"finder"`.

- [ ] **Step 3: Implement**

`assist/app.py`:
- import `ObjectFinder` in the `from .perception import (...)` block.
- `__init__`, right after `self.detector = ...`:
```python
        # On-demand focused detector for find_object; shares the GPU with the
        # live detector through YOLO_LOCK.
        self.finder = ObjectFinder() if enable_yolo else None
```
- `self.tools = ToolRegistry(self.store, llm=self.llm, finder=self.finder)`
- `_detect_watch`, after the detector warmup `try/except`:
```python
        if self.finder is not None:
            t0 = time.monotonic()
            try:
                self.finder.warmup()
                log("app", "finder warm", ms=round((time.monotonic() - t0) * 1000))
            except Exception as exc:
                log_exc("app", "finder warmup failed", exc)
```

`.env.example`, after `DETECT_INTERVAL_SEC=0.5`:
```
# find_object (p/c/f keys, voice "where is …") runs a fresh focused pass with
# this model at this resolution. yolov8l-worldv2.pt downloads ~90 MB once.
FIND_MODEL=yolov8l-worldv2.pt
FIND_IMGSZ=1280
```

`README.md` env table, after the `DETECT_INTERVAL_SEC` row:
```
| `FIND_MODEL` | Model for the on-demand find pass (default `yolov8l-worldv2.pt`, ~90 MB download once) |
| `FIND_IMGSZ` | Resolution of the find pass (default `1280`) |
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint (no commit)**

---

### Task 6: Real-model verification

**Files:**
- Create (scratch, not in repo): `<scratchpad>/bench_finder.py`

- [ ] **Step 1: Benchmark the real finder on MPS**

```python
import time
import numpy as np
from assist.perception.find import ObjectFinder

f = ObjectFinder()
t0 = time.monotonic(); f.warmup(); print("warmup s", round(time.monotonic() - t0, 1), "device", f.device)
rgb = (np.random.rand(1920, 1440, 3) * 255).astype(np.uint8)
depth = np.full((1920, 1440), 2.0, np.float32); conf = np.full((1920, 1440), 2, np.uint8)
for target in ("door", "door", "person", "door", "person"):
    t = time.monotonic(); hits = f.find(rgb, depth, conf, target)
    print(target, round((time.monotonic() - t) * 1000), "ms", len(hits), "hits")
```
Run from the repo root: `.venv/bin/python <scratchpad>/bench_finder.py`
Expected: weights download once into the repo root (`*.pt` is gitignored), warm-up a few seconds, then per-find ms on `mps`. Record the numbers for the user.

- [ ] **Step 2: Full suite + import smoke**

Run: `.venv/bin/python -m pytest -q` and `.venv/bin/python -c "import assist.app"`
Expected: all pass, import clean.

- [ ] **Step 3: Hand to the user for a live check**

Door and person that the stored labels missed; look for `find | pass … best=…` and `app | finder warm` in the log. Commit only after the user approves.
