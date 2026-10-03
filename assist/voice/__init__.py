"""Local TTS helpers for keyboard shortcuts and boot prompts."""

from .audio import SAMPLE_RATE, is_speaking, speak, stop_speaking

__all__ = [
    "SAMPLE_RATE",
    "is_speaking",
    "speak",
    "stop_speaking",
]
