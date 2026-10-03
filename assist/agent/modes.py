"""Tool / keyboard modes for scene vs distance vs find."""

from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    SCENE = "scene"
    DISTANCE = "distance"
    FIND = "find"
