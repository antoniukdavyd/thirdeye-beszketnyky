"""Tool routing for keyboard/voice describe requests."""

from __future__ import annotations

import time
from typing import List

import numpy as np

from assist.agent.modes import Intent
from assist.app import AssistApp


class SlowLLM:
    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls: List[str] = []
        self.available = True
        self.model = "mock"

    def describe(self, rgb_bgr, scene, user_question="", mode=Intent.SCENE):
        self.calls.append(user_question)
        time.sleep(self.delay)
        return "In front of you a test scene."


def test_distance_uses_tool_not_vision(monkeypatch):
    spoken: List[str] = []
    monkeypatch.setattr(
        "assist.app.speak", lambda *a, **k: spoken.append(a[0] if a else "")
    )

    app = AssistApp(enable_yolo=False, enable_voice=False)
    llm = SlowLLM()
    app.llm = llm
    app.tools.llm = llm
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    scene = {
        "zones_m": {"left": 1.0, "center": 0.8, "right": 2.0},
        "objects": [
            {"label": "person", "dist_m": 1.1, "bearing": "center", "conf_depth": 3, "score": 0.9}
        ],
        "hint": "",
    }
    app.store.publish(rgb, scene, [])
    app.last_rgb = rgb
    app.scene = scene

    app._request_describe("how far", mode=Intent.DISTANCE)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not spoken:
        time.sleep(0.05)

    assert spoken
    assert "meter" in spoken[-1].lower() or "m" in spoken[-1]
    assert llm.calls == []  # no vision for distance


def test_scene_calls_vision_llm(monkeypatch):
    spoken: List[str] = []
    monkeypatch.setattr(
        "assist.app.speak", lambda *a, **k: spoken.append(a[0] if a else "")
    )

    app = AssistApp(enable_yolo=False, enable_voice=False)
    llm = SlowLLM(delay=0.02)
    app.llm = llm
    app.tools.llm = llm
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    scene = {"zones_m": {}, "objects": [], "hint": ""}
    app.store.publish(rgb, scene, [])
    app.last_rgb = rgb
    app.scene = scene

    app._request_describe("first question", mode=Intent.SCENE)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not llm.calls:
        time.sleep(0.05)

    assert llm.calls
    assert "first" in llm.calls[0] or llm.calls[0]
