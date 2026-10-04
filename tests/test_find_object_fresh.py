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
