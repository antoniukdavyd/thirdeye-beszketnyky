"""Hardware key-state helpers for true hold-to-talk PTT (macOS)."""

from __future__ import annotations

from ctypes import CDLL, c_bool, c_uint32
from functools import lru_cache

# macOS virtual keycode for Space
_SPACE_KEYCODE = 49
_HID_SYSTEM_STATE = 1  # kCGEventSourceStateHIDSystemState


@lru_cache(maxsize=1)
def _cg_event_source_key_state():
    try:
        lib = CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/"
            "ApplicationServices"
        )
        fn = lib.CGEventSourceKeyState
        fn.argtypes = [c_uint32, c_uint32]
        fn.restype = c_bool
        return fn
    except OSError:
        return None


def is_space_down() -> bool:
    """Return True while the physical Space key is held (macOS HID state)."""
    fn = _cg_event_source_key_state()
    if fn is None:
        return False
    return bool(fn(_HID_SYSTEM_STATE, _SPACE_KEYCODE))


def space_hold_available() -> bool:
    return _cg_event_source_key_state() is not None
