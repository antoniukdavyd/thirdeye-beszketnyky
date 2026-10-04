"""read_text tool: exact strings + bearing from OCR, graceful when unavailable."""

from __future__ import annotations

import numpy as np

from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry


def _store_with_frame():
    s = SceneStore()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    s.publish(rgb, {"zones_m": {"left": 1.0, "center": 1.0, "right": 1.0}}, objects=[])
    return s


def _fake_ocr(lines):
    """Return an OCR callable yielding the given (text, conf, bearing) lines."""

    def _ocr(rgb_bgr):
        return [
            {"text": t, "conf": c, "bearing": b} for (t, c, b) in lines
        ]

    return _ocr


def test_read_text_returns_exact_strings_and_bearing():
    ocr = _fake_ocr([("Bus 42", 0.98, "left"), ("Downtown", 0.95, "center")])
    reg = ToolRegistry(_store_with_frame(), ocr=ocr)
    out = reg.execute("read_text", {})
    assert out["ok"] and out["found"]
    texts = [ln["text"] for ln in out["lines"]]
    assert texts == ["Bus 42", "Downtown"]
    assert out["lines"][0]["bearing"] == "left"


def test_read_text_spoken_phrase():
    ocr = _fake_ocr([("Bus 42", 0.98, "left")])
    reg = ToolRegistry(_store_with_frame(), ocr=ocr)
    out = reg.execute("read_text", {})
    spoken = reg.spoken_from_tool("read_text", out)
    assert "Bus 42" in spoken


def test_read_text_nothing_found():
    reg = ToolRegistry(_store_with_frame(), ocr=_fake_ocr([]))
    out = reg.execute("read_text", {})
    assert out["ok"] and not out["found"]
    spoken = reg.spoken_from_tool("read_text", out)
    assert "readable text" in spoken.lower() or "don't" in spoken.lower()


def test_read_text_unavailable_degrades():
    def _broken(_rgb):
        raise RuntimeError("ocr unavailable")

    reg = ToolRegistry(_store_with_frame(), ocr=_broken)
    out = reg.execute("read_text", {})
    assert out["ok"] is False


def test_read_text_filters_low_confidence():
    ocr = _fake_ocr([("solid", 0.9, "center"), ("noise", 0.1, "right")])
    reg = ToolRegistry(_store_with_frame(), ocr=ocr)
    out = reg.execute("read_text", {})
    texts = [ln["text"] for ln in out["lines"]]
    assert "solid" in texts
    assert "noise" not in texts


def test_read_text_in_function_schema():
    reg = ToolRegistry(SceneStore())
    names = {d["name"] for d in reg.declarations()}
    assert "read_text" in names
