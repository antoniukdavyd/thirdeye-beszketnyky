"""YOLO labeling cadence: time-paced, newest frame, hold sized to the interval."""

import threading
import time
from types import SimpleNamespace

import pytest

from assist.app import AssistApp, detect_interval_sec, hold_passes
from assist.perception.detect import DetectionHold


def test_detect_interval_default(monkeypatch):
    monkeypatch.delenv("DETECT_INTERVAL_SEC", raising=False)
    assert detect_interval_sec() == 0.5


def test_detect_interval_env_override(monkeypatch):
    monkeypatch.setenv("DETECT_INTERVAL_SEC", "0.2")
    assert detect_interval_sec() == 0.2


@pytest.mark.parametrize("raw", ["abc", ""])
def test_detect_interval_bad_value_falls_back(monkeypatch, raw):
    monkeypatch.setenv("DETECT_INTERVAL_SEC", raw)
    assert detect_interval_sec() == 0.5


def test_detect_interval_has_floor(monkeypatch):
    monkeypatch.setenv("DETECT_INTERVAL_SEC", "0")
    assert detect_interval_sec() > 0


def test_hold_passes_tracks_interval():
    # Hold ~0.5 s of empty passes regardless of how often YOLO runs.
    assert hold_passes(0.5) == 1
    assert hold_passes(0.15) == 3
    assert hold_passes(0.12) == 4
    assert hold_passes(2.0) == 1


class FakeDetector:
    def __init__(self):
        self.seen = []
        self.done = threading.Event()

    def warmup(self):
        pass

    def detect(self, rgb, depth, conf):
        self.seen.append((rgb, time.monotonic()))
        self.done.set()
        return []


def worker_app(interval: float) -> AssistApp:
    app = AssistApp.__new__(AssistApp)
    app.detector = FakeDetector()
    app.detection_hold = DetectionHold(ttl_frames=1)
    app.objects = []
    app.watchdog = SimpleNamespace(beat=lambda *a, **k: None, forget=lambda *a: None)
    app.framelog = SimpleNamespace(detected=lambda ms: None)
    app._detect_interval = interval
    app._detect_stop = threading.Event()
    app._detect_wake = threading.Event()
    app._detect_lock = threading.Lock()
    app._detect_input = None
    app._detect_thread = None
    return app


def test_worker_uses_newest_frame_after_waiting():
    app = worker_app(interval=0.3)
    app._start_detect_watch()
    try:
        app._submit_detect("A", None, None)
        assert app.detector.done.wait(1.0)
        app.detector.done.clear()
        # Both land while the next pass is not yet due; only C may be labeled.
        app._submit_detect("B", None, None)
        app._submit_detect("C", None, None)
        assert app.detector.done.wait(1.0)
    finally:
        app._stop_detect_watch()
    assert [f for f, _ in app.detector.seen] == ["A", "C"]


def test_worker_paces_passes_by_interval():
    app = worker_app(interval=0.2)
    app._start_detect_watch()
    try:
        end = time.monotonic() + 0.75
        i = 0
        while time.monotonic() < end:
            app._submit_detect(i, None, None)
            i += 1
            time.sleep(1 / 60)
    finally:
        app._stop_detect_watch()
    starts = [t for _, t in app.detector.seen]
    assert 3 <= len(starts) <= 5
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert min(gaps) >= 0.19
