"""Agent tools: read SceneStore; optional vision for describe_scene."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from ..agent.modes import Intent
from ..debuglog import log
from ..llm.openrouter import OpenRouterClient, _clip_spoken
from ..perception.scene import compact_scene_for_llm
from ..perception.scene_store import SceneStore

SCENE_STALE_MS = 1500
OCR_MIN_CONF = 0.3

# label aliases: user/agent string → YOLO substrings
_LABEL_ALIASES = {
    "person": ("person",),
    "people": ("person",),
    "человек": ("person",),
    "люди": ("person",),
    "людей": ("person",),
    "door": ("door",),
    "дверь": ("door",),
    "car": ("car",),
    "машина": ("car",),
    "bus": ("bus", "truck"),
    "автобус": ("bus", "truck"),
    "table": ("table",),
    "стол": ("table",),
    "chair": ("chair",),
    "стул": ("chair",),
    "traffic light": ("traffic light", "traffic"),
    "светофор": ("traffic light", "traffic"),
    "stairs": ("stairs",),
    "лестница": ("stairs",),
    "bag": ("bag",),
    "сумка": ("bag",),
    "laptop": ("laptop",),
    "ноутбук": ("laptop",),
    "bottle": ("bottle",),
    "бутылка": ("bottle",),
}


def _match_labels(target: str) -> tuple[str, ...]:
    t = (target or "").strip().lower()
    if not t:
        return ()
    if t in _LABEL_ALIASES:
        return _LABEL_ALIASES[t]
    for key, labs in _LABEL_ALIASES.items():
        if key in t or t in key:
            return labs
    return (t,)


def _bearing_en(bearing: str) -> str:
    m = {
        "left": "on the left",
        "center-left": "center-left",
        "center": "in the center",
        "center-right": "center-right",
        "right": "on the right",
    }
    return m.get((bearing or "").lower(), bearing or "")


class ToolRegistry:
    """Local tools for the realtime agent (and keyboard shortcuts)."""

    def __init__(
        self,
        store: SceneStore,
        llm: Optional[OpenRouterClient] = None,
        ocr: Optional[Callable[[Any], list]] = None,
    ) -> None:
        self.store = store
        self.llm = llm or OpenRouterClient()
        # Injectable OCR callable (rgb_bgr -> [{text, conf, bearing, bbox}]).
        # Defaults to Apple Vision; tests pass a fake.
        if ocr is None:
            from ..perception.ocr import recognize_text

            ocr = recognize_text
        self._ocr = ocr

    def declarations(self) -> list[dict]:
        """Gemini / OpenAI-style function declarations."""
        return [
            {
                "name": "sense_snapshot",
                "description": (
                    "Read current LiDAR/camera SENSOR_JSON: zone clearances "
                    "L/C/R in meters and nearby detected objects. "
                    "Use for clearances and nearby hazards like cars. "
                    "Call when you need fresh sensor context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "measure_distances",
                "description": (
                    "Return distances in meters from SENSOR_JSON only. "
                    "Optional target filters objects (person, door, car, …). "
                    "Never invent meters — use this tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Object type to measure, e.g. person, door",
                        },
                    },
                },
            },
            {
                "name": "find_object",
                "description": (
                    "Find whether an object is in view; return side and meters from JSON."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Object to find, e.g. door, person, car",
                        },
                    },
                    "required": ["label"],
                },
            },
            {
                "name": "read_text",
                "description": (
                    "Read text in view (signs, bus/route numbers, door labels, "
                    "notices) with on-device OCR. Returns EXACT strings and their "
                    "side. Use for “read this / what does the sign say / which bus / "
                    "route number”. Report the exact text; never guess or correct it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "describe_scene",
                "description": (
                    "Answer a visual question about the scene from camera + sensors. "
                    "Use for “what’s around / what’s ahead / what is this / describe”. "
                    "Pass the user's actual words in `question`. Not for exact meters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's actual question, verbatim",
                        },
                        "focus": {
                            "type": "string",
                            "description": "Alias for question (optional)",
                        },
                    },
                },
            },
        ]

    def execute(self, name: str, args: Optional[dict] = None) -> dict:
        args = args or {}
        t0 = time.monotonic()
        handlers: dict[str, Callable[[dict], dict]] = {
            "sense_snapshot": self._sense_snapshot,
            "measure_distances": self._measure_distances,
            "find_object": self._find_object,
            "describe_scene": self._describe_scene,
            "read_text": self._read_text,
        }
        fn = handlers.get(name)
        if fn is None:
            out = {"ok": False, "error": f"unknown tool {name}"}
        else:
            try:
                out = fn(args)
            except Exception as e:
                out = {"ok": False, "error": str(e)}
        dt = (time.monotonic() - t0) * 1000.0
        log("tool", f"{name} done", ms=round(dt, 1), ok=int(out.get("ok", False)))
        return out

    def spoken_from_tool(self, name: str, result: dict) -> str:
        """Turn tool result into a short English phrase for keyboard / offline TTS."""
        if not result.get("ok"):
            return result.get("error") or "No camera data."
        if name == "describe_scene":
            return result.get("text") or "No description."
        if name == "read_text":
            lines = result.get("lines") or []
            if not lines:
                return "I don't see any readable text."
            parts = []
            for ln in lines[:4]:
                side = _bearing_en(ln.get("bearing", ""))
                parts.append(f"{ln['text']} ({side})" if side else ln["text"])
            return "Text: " + "; ".join(parts) + "."
        if name == "find_object":
            if not result.get("found"):
                return f"I don't see {result.get('label', 'the object')}."
            lab = result.get("label_en") or result.get("label")
            return (
                f"In front of you {lab}: {_bearing_en(result.get('bearing', ''))}, "
                f"about {result.get('dist_m')} meters."
            )
        if name in ("measure_distances", "sense_snapshot"):
            spoken = result.get("spoken")
            if spoken:
                return spoken
            return _clip_spoken(
                "In front of you "
                + json.dumps(result.get("data") or {}, ensure_ascii=False)[:120]
            )
        return str(result)

    def _sense_snapshot(self, args: dict) -> dict:
        snap = self.store.snapshot(copy_rgb=False)
        if not snap.has_frame:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        data = compact_scene_for_llm(snap.scene)
        data["objects"] = (data.get("objects") or [])[:8]
        spoken = self._spoken_distances(data, target=None)
        return {
            "ok": True,
            "age_ms": round(snap.age_ms, 1),
            "data": data,
            "spoken": spoken,
        }

    def _measure_distances(self, args: dict) -> dict:
        snap = self.store.snapshot(copy_rgb=False)
        if not snap.has_frame:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        data = compact_scene_for_llm(snap.scene)
        target = (args.get("target") or "").strip() or None
        if target:
            labs = _match_labels(target)
            objs = [
                o
                for o in (data.get("objects") or [])
                if any(l in str(o.get("label", "")).lower() for l in labs)
            ]
            data = {
                "zones_m": data.get("zones_m") or {},
                "objects": objs[:6],
                "hint": data.get("hint", ""),
                "target": target,
            }
        else:
            data["objects"] = (data.get("objects") or [])[:6]
        spoken = self._spoken_distances(data, target=target)
        return {
            "ok": True,
            "age_ms": round(snap.age_ms, 1),
            "data": data,
            "spoken": spoken,
        }

    def _find_object(self, args: dict) -> dict:
        label = (args.get("label") or "").strip()
        if not label:
            return {"ok": False, "error": "label required"}
        snap = self.store.snapshot(copy_rgb=False)
        if not snap.has_frame:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        data = compact_scene_for_llm(snap.scene)
        labs = _match_labels(label)
        hits = [
            o
            for o in (data.get("objects") or [])
            if any(l in str(o.get("label", "")).lower() for l in labs)
        ]
        if not hits:
            return {
                "ok": True,
                "found": False,
                "label": label,
                "age_ms": round(snap.age_ms, 1),
            }
        hit = min(hits, key=lambda o: float(o.get("dist_m") or 99))
        label_en = {
            "person": "a person",
            "door": "a door",
            "car": "a car",
            "table": "a table",
            "chair": "a chair",
        }.get(str(hit.get("label", "")).lower(), hit.get("label"))
        return {
            "ok": True,
            "found": True,
            "label": hit.get("label"),
            "label_en": label_en,
            "bearing": hit.get("bearing"),
            "dist_m": hit.get("dist_m"),
            "age_ms": round(snap.age_ms, 1),
        }

    def _describe_scene(self, args: dict) -> dict:
        snap = self.store.snapshot(copy_rgb=True)
        if not snap.has_frame or snap.rgb_bgr is None:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        # Prefer the user's real question; `focus` kept as a backward alias.
        q = (args.get("question") or args.get("focus") or "").strip()
        if not q:
            q = "Describe the scene in front of the user."
        text = self.llm.describe(
            snap.rgb_bgr,
            snap.scene,
            user_question=q,
            mode=Intent.SCENE,
        )
        return {
            "ok": True,
            "text": _clip_spoken(text),
            "age_ms": round(snap.age_ms, 1),
        }

    def _read_text(self, args: dict) -> dict:
        snap = self.store.snapshot(copy_rgb=True)
        if not snap.has_frame or snap.rgb_bgr is None:
            return {"ok": False, "error": "No frame"}
        if snap.age_ms > SCENE_STALE_MS:
            return {"ok": False, "error": "No current camera frame"}
        try:
            raw = self._ocr(snap.rgb_bgr)
        except Exception as e:
            log("tool", f"read_text ocr failed: {e}")
            return {"ok": False, "error": "ocr unavailable"}
        lines = [
            {
                "text": str(ln.get("text", "")).strip(),
                "conf": round(float(ln.get("conf", 0.0)), 3),
                "bearing": ln.get("bearing", ""),
            }
            for ln in (raw or [])
            if str(ln.get("text", "")).strip()
            and float(ln.get("conf", 0.0)) >= OCR_MIN_CONF
        ]
        return {
            "ok": True,
            "found": bool(lines),
            "lines": lines,
            "age_ms": round(snap.age_ms, 1),
        }

    def _spoken_distances(self, data: dict, target: Optional[str]) -> str:
        objs = data.get("objects") or []
        zones = data.get("zones_m") or {}
        if objs:
            o = objs[0]
            lab = o.get("label")
            if "person" in str(lab).lower():
                lab = "a person"
            return (
                f"In front of you {lab} {_bearing_en(str(o.get('bearing', '')))}, "
                f"about {o.get('dist_m')} meters."
            )
        parts = []
        for name, side in (
            ("left", "left"),
            ("center", "center"),
            ("right", "right"),
        ):
            v = zones.get(side)
            if v is not None:
                parts.append(f"{name} {v} m")
        if parts:
            return "In front of you zones: " + ", ".join(parts) + "."
        if target:
            return f"I don't see {target} in the sensor zone."
        return "No distance data."


def parse_tool_arguments(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return json.loads(raw)
    return {}


def deepgram_think_functions(registry: ToolRegistry) -> list[dict]:
    """Function defs for Deepgram Voice Agent.

    Omit ``client_side`` and ``endpoint``: Deepgram rejects ``client_side`` in
    Settings (UNPARSABLE_CLIENT_MESSAGE) and treats missing ``endpoint`` as
    client-side execution.
    """
    out = []
    for d in registry.declarations():
        out.append(
            {
                "name": d["name"],
                "description": d["description"],
                "parameters": d.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return out


def gemini_tool_declarations(registry: ToolRegistry) -> list[Any]:
    """Convert to google.genai types.Tool if SDK available, else raw dicts."""
    decls = registry.declarations()
    try:
        from google.genai import types

        fns = []
        for d in decls:
            fns.append(
                types.FunctionDeclaration(
                    name=d["name"],
                    description=d["description"],
                    parameters=d.get("parameters") or {"type": "object", "properties": {}},
                )
            )
        return [types.Tool(function_declarations=fns)]
    except Exception:
        return decls
