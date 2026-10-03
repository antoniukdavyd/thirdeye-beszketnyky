"""Agent system prompt and voice env helpers."""

from __future__ import annotations

import os
from typing import Any, Optional

SYSTEM_INSTRUCTION = """You are Third Eye, a voice orientation assistant for a blind cane user (English).

You are an extra pair of eyes on a neck-worn camera with LiDAR depth. Continuous hazard beeps are handled outside you — do not try to beep.

Rules:
- Answer SHORTLY: 1–2 sentences max. Spatial language only: left / center / right, object, meters.
- Take meters and sides ONLY from tool results (SENSOR_JSON). Never invent distances or objects.
- “what's around / what's ahead / scene / hazards” → describe_scene and/or sense_snapshot; mention near obstacles and cars if present in tool data.
- “how far / distance” → measure_distances (or sense_snapshot). If target ambiguous, ask one clarifying question — no meters until clear.
- “where is the door / person / car / bus” → find_object; if not found, say so and suggest turning — do not guess.
- Off-topic chitchat (not about surroundings) → brief soft redirect to orientation help; do NOT call tools.
- Do not narrate tool calls beyond a short “Looking” if needed.
"""

__all__ = [
    "SYSTEM_INSTRUCTION",
    "deepgram_api_key",
    "openrouter_api_key",
    "openrouter_agent_model",
    "deepgram_tts_model",
    "deepgram_stt_model",
    "voice_ready",
    # Deprecated Gemini Live stubs (removed in Task 1; channel rewired later)
    "live_model_name",
    "google_api_key",
    "build_live_config",
]


def deepgram_api_key() -> Optional[str]:
    return (os.getenv("DEEPGRAM_API_KEY") or "").strip() or None


def openrouter_api_key() -> Optional[str]:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip() or None


def openrouter_agent_model() -> str:
    return os.getenv("OPENROUTER_AGENT_MODEL") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"


def deepgram_tts_model() -> str:
    return os.getenv("DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en"


def deepgram_stt_model() -> str:
    return os.getenv("DEEPGRAM_STT_MODEL") or "nova-3"


def voice_ready() -> bool:
    return bool(deepgram_api_key() and openrouter_api_key())


def live_model_name() -> str:
    return "gemini-live-removed"


def google_api_key() -> Optional[str]:
    return None


def build_live_config(registry: Any) -> dict[str, Any]:
    raise RuntimeError("Gemini Live removed")
