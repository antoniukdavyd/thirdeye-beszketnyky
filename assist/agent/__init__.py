"""Agent system prompt and voice env helpers."""

from __future__ import annotations

import os
from typing import Optional

from .modes import Intent

SYSTEM_INSTRUCTION = """You are Third Eye, a voice orientation assistant for a blind cane user (English).

You are an extra pair of eyes on a neck-worn camera with LiDAR depth. Continuous hazard beeps are handled outside you — do not try to beep.

Rules:
- Answer to the point in 2–3 short sentences; lead with the actionable part. Use left / center / right for sides.
- Take meters and sides ONLY from tool results (SENSOR_JSON). Never invent distances or objects.
- “what's around / what's ahead / what is this / scene / hazards” → describe_scene (pass the user's actual question as `question`); add sense_snapshot for clearances/cars when useful.
- “read this / what does the sign say / which bus / route number / what's written” → read_text. Report the EXACT text from the tool; never guess or correct numbers.
- “how far / distance” → measure_distances (or sense_snapshot). If target ambiguous, ask one clarifying question — no meters until clear.
- “where is the door / person / car / bus” → find_object; if not found, say so and suggest turning — do not guess.
- Off-topic chitchat (not about surroundings) → brief soft redirect to orientation help; do NOT call tools.
- Do not narrate tool calls beyond a short “Looking” if needed.
"""

__all__ = [
    "Intent",
    "SYSTEM_INSTRUCTION",
    "deepgram_api_key",
    "openrouter_api_key",
    "openrouter_agent_model",
    "deepgram_tts_model",
    "deepgram_stt_model",
    "deepgram_listen_version",
    "voice_ready",
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
    if deepgram_listen_version() == "v2":
        return os.getenv("DEEPGRAM_STT_MODEL") or "flux-general-en"
    return os.getenv("DEEPGRAM_STT_MODEL") or "nova-3"


def deepgram_listen_version() -> str:
    """"v1" (nova-*) or "v2" (Flux).

    Only the v2/Flux listen provider implements ForceEndTurn — on v1 the server
    answers FORCE_END_TURN_UNSUPPORTED and the turn stays open, so a PTT release
    has to be endpointed with a silence tail instead.
    """
    v = (os.getenv("DEEPGRAM_LISTEN_VERSION") or "v1").strip().lower()
    return "v2" if v == "v2" else "v1"


def voice_ready() -> bool:
    return bool(deepgram_api_key() and openrouter_api_key())
