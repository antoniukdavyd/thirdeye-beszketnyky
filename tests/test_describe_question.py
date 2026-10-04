"""describe_scene must forward the user's actual question to the vision LLM."""

from __future__ import annotations

import numpy as np

from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry


class _FakeLLM:
    """Captures the question passed into describe()."""

    def __init__(self) -> None:
        self.last_question = None

    def describe(self, rgb_bgr, scene, user_question="", mode=None):
        self.last_question = user_question
        return f"answer to: {user_question}"


def _store_with_frame():
    s = SceneStore()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    s.publish(rgb, {"zones_m": {"left": 1.0, "center": 1.0, "right": 1.0}}, objects=[])
    return s


def test_describe_scene_forwards_question():
    llm = _FakeLLM()
    reg = ToolRegistry(_store_with_frame(), llm=llm)
    out = reg.execute("describe_scene", {"question": "what is on the table?"})
    assert out["ok"]
    assert llm.last_question == "what is on the table?"
    assert "what is on the table?" in out["text"]


def test_describe_scene_focus_alias_still_works():
    llm = _FakeLLM()
    reg = ToolRegistry(_store_with_frame(), llm=llm)
    reg.execute("describe_scene", {"focus": "describe the doorway"})
    assert llm.last_question == "describe the doorway"


def test_describe_scene_default_question_when_empty():
    llm = _FakeLLM()
    reg = ToolRegistry(_store_with_frame(), llm=llm)
    reg.execute("describe_scene", {})
    assert llm.last_question  # non-empty default, not a crash


def test_describe_scene_question_in_function_schema():
    reg = ToolRegistry(SceneStore())
    decl = next(d for d in reg.declarations() if d["name"] == "describe_scene")
    assert "question" in decl["parameters"]["properties"]
