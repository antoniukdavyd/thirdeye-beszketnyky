"""Offline FIND fallback must match by target label, not loose substring."""

from __future__ import annotations

from assist.agent.modes import Intent
from assist.llm.openrouter import OpenRouterClient


SCENE = {
    "zones_m": {"left": 2.0, "center": 1.5, "right": 2.2},
    "objects": [
        {"label": "person", "dist_m": 1.2, "bearing": "left"},
        {"label": "door", "dist_m": 3.0, "bearing": "center"},
        {"label": "car", "dist_m": 4.5, "bearing": "right"},
        {"label": "table", "dist_m": 1.0, "bearing": "center-left"},
    ],
}


def _client() -> OpenRouterClient:
    return OpenRouterClient(api_key="")  # force offline


def test_find_door():
    phrase = _client().describe(
        rgb_bgr=__import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        scene=SCENE,
        user_question="Where is the door?",
        mode=Intent.FIND,
    )
    assert "door" in phrase.lower()
    assert "don't see" not in phrase.lower() and "do not see" not in phrase.lower()


def test_find_person():
    phrase = _client().describe(
        rgb_bgr=__import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        scene=SCENE,
        user_question="Where are the people?",
        mode=Intent.FIND,
    )
    assert "person" in phrase.lower()


def test_find_missing_object():
    phrase = _client().describe(
        rgb_bgr=__import__("numpy").zeros((8, 8, 3), dtype="uint8"),
        scene=SCENE,
        user_question="Where is the chair?",
        mode=Intent.FIND,
    )
    assert "don't see" in phrase.lower() or "do not see" in phrase.lower()
