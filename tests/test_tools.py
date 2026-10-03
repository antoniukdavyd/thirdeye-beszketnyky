"""Unit tests for SceneStore + agent tools."""

from __future__ import annotations

import time

import numpy as np
import pytest

from assist.perception.detect import DetectedObject
from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry


def _scene_with_person():
    return {
        "zones_m": {"left": 1.2, "center": 0.9, "right": 2.0},
        "objects": [
            {
                "label": "person",
                "dist_m": 1.02,
                "bearing": "center",
                "conf_depth": 5,
                "score": 0.9,
            },
            {
                "label": "bag",
                "dist_m": 1.54,
                "bearing": "left",
                "conf_depth": 4,
                "score": 0.7,
            },
        ],
        "hint": "center close",
        "device": "test",
    }


@pytest.fixture
def store():
    s = SceneStore()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    s.publish(rgb, _scene_with_person(), objects=[])
    return s


def test_scene_store_age(store):
    snap = store.snapshot()
    assert snap.has_frame
    assert snap.age_ms >= 0
    time.sleep(0.02)
    assert store.snapshot().age_ms >= 15


def test_sense_snapshot(store):
    reg = ToolRegistry(store)
    out = reg.execute("sense_snapshot", {})
    assert out["ok"]
    assert "zones_m" in out["data"]
    assert out["age_ms"] >= 0


def test_measure_distances_target_person(store):
    reg = ToolRegistry(store)
    out = reg.execute("measure_distances", {"target": "людей"})
    assert out["ok"]
    objs = out["data"]["objects"]
    assert len(objs) == 1
    assert objs[0]["dist_m"] == 1.02
    assert "1.02" in out["spoken"] or "meter" in out["spoken"]


def test_measure_no_invent_without_frame():
    reg = ToolRegistry(SceneStore())
    out = reg.execute("measure_distances", {})
    assert not out["ok"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("sense_snapshot", {}),
        ("measure_distances", {}),
        ("find_object", {"label": "person"}),
        ("describe_scene", {}),
    ],
)
def test_tools_reject_stale_camera_frame(store, name, args):
    store._updated_at = time.monotonic() - 1.501

    out = ToolRegistry(store).execute(name, args)

    assert out == {"ok": False, "error": "No current camera frame"}


def test_find_object_door_missing(store):
    reg = ToolRegistry(store)
    out = reg.execute("find_object", {"label": "дверь"})
    assert out["ok"]
    assert out["found"] is False


def test_find_object_person(store):
    reg = ToolRegistry(store)
    out = reg.execute("find_object", {"label": "person"})
    assert out["ok"] and out["found"]
    assert out["dist_m"] == 1.02
    spoken = reg.spoken_from_tool("find_object", out)
    assert "meter" in spoken


def test_measure_fast(store):
    reg = ToolRegistry(store)
    t0 = time.perf_counter()
    for _ in range(20):
        reg.execute("measure_distances", {"target": "person"})
    ms = (time.perf_counter() - t0) * 1000 / 20
    assert ms < 50.0
