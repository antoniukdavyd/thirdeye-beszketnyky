"""Crash forensics and demo resilience: watchdog, hooks, frame-error skipping."""

import threading

import pytest

import assist.app as appmod
from assist import crashlog
from assist.perception.detect import ObjectDetector


def test_watchdog_logs_stall_once_with_stacks_and_recovery(capsys):
    wd = crashlog.Watchdog(stall_sec=1.0)
    wd.beat("main")
    t0 = wd._beats["main"]

    wd.check(now=t0 + 0.5)
    assert wd.stall_count == 0

    wd.check(now=t0 + 1.2)
    wd.check(now=t0 + 2.0)  # still stalled: no second report
    out = capsys.readouterr().out
    assert wd.stall_count == 1
    assert out.count("main STALLED") == 1
    assert "--- thread 'MainThread' ---" in out  # stacks dumped

    wd.beat("main")
    wd.check()
    assert "main resumed" in capsys.readouterr().out
    assert "stalled" not in wd.health()


def test_watchdog_per_name_limit_and_forget(capsys):
    wd = crashlog.Watchdog(stall_sec=1.0)
    wd.beat("detect", limit_sec=3.0)
    t0 = wd._beats["detect"]
    wd.check(now=t0 + 2.0)
    assert wd.stall_count == 0
    wd.forget("detect")
    wd.check(now=t0 + 10.0)
    assert wd.stall_count == 0


def test_thread_exception_is_logged_with_thread_name(capsys, monkeypatch):
    monkeypatch.setattr(threading, "excepthook", crashlog._thread_excepthook)

    def boom():
        raise RuntimeError("kaboom")

    t = threading.Thread(target=boom, name="record3d-rx")
    t.start()
    t.join()
    out = capsys.readouterr().out
    assert "UNCAUGHT RuntimeError in thread 'record3d-rx': kaboom" in out
    assert "Traceback" in out


def _bare_app():
    app = appmod.AssistApp.__new__(appmod.AssistApp)
    app._frame_errors_in_row = 0
    app._frame_error_counts = {}
    app.frame_i = 7
    return app


def test_bad_frame_is_skipped_not_fatal(monkeypatch, capsys):
    monkeypatch.setattr(appmod.cv2, "waitKey", lambda _ms: 255)
    app = _bare_app()
    for _ in range(5):
        try:
            raise ValueError("buffer is smaller than requested size")
        except ValueError as exc:
            assert app._frame_failed(exc) == 255
    out = capsys.readouterr().out
    # Full traceback only for the first few repeats.
    assert out.count("frame error: ValueError") == 3
    assert out.count("Traceback") == 3


def test_persistent_frame_errors_eventually_raise(monkeypatch):
    monkeypatch.setattr(appmod.cv2, "waitKey", lambda _ms: 255)
    app = _bare_app()
    app._frame_errors_in_row = appmod.MAX_FRAME_ERRORS_IN_ROW - 1
    with pytest.raises(ValueError):
        try:
            raise ValueError("still broken")
        except ValueError as exc:
            app._frame_failed(exc)


def test_yolo_gpu_failure_falls_back_to_cpu():
    calls = []

    class FakeModel:
        def predict(self, _img, **kw):
            calls.append(kw["device"])
            if kw["device"] != "cpu":
                raise RuntimeError("MPS backend out of memory")
            return []

    det = ObjectDetector()
    det._model = FakeModel()
    det.device = "mps"
    assert det._predict(object()) == []
    assert calls == ["mps", "cpu"]
    assert det.device == "cpu"
