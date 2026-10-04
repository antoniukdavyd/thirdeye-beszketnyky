"""Local TTS for keyboard shortcuts and boot prompts (edge-tts / macOS say).

Realtime voice uses Deepgram audio; this module is not that speech pipeline.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import threading
from typing import Callable, Optional

from ..debuglog import log, log_exc

SAMPLE_RATE = 16000

_tts_lock = threading.Lock()
_tts_speaking = False
_tts_proc: Optional[subprocess.Popen] = None
_tts_tmp: Optional[str] = None


def is_speaking() -> bool:
    return _tts_speaking


def stop_speaking() -> None:
    """Interrupt current TTS playback only (tracked process)."""
    global _tts_speaking, _tts_proc, _tts_tmp
    proc = None
    with _tts_lock:
        proc = _tts_proc
        _tts_proc = None
        _tts_speaking = False
    if proc is not None:
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _tts_provider() -> str:
    return (os.getenv("TTS_PROVIDER") or "edge").lower().strip()


def _tts_voice_edge() -> str:
    return os.getenv("TTS_VOICE") or "en-US-JennyNeural"


def _speak_edge(text: str, on_start: Optional[Callable[[], None]]) -> bool:
    """Synthesize with edge-tts and play via afplay. Returns False on failure."""
    global _tts_speaking, _tts_proc, _tts_tmp
    try:
        import edge_tts
    except ImportError:
        print("edge-tts not installed — falling back to say")
        log("tts", "edge-tts missing → fallback say")
        return False

    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    _tts_tmp = path
    try:

        async def _save():
            communicate = edge_tts.Communicate(text, _tts_voice_edge())
            await communicate.save(path)

        try:
            asyncio.run(_save())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save())
            finally:
                loop.close()

        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size < 100:
            print(f"edge-tts produced empty audio ({size} bytes)")
            log("tts", "edge empty audio", bytes=size)
            return False

        print(f"[TTS/edge] {_tts_voice_edge()} ({size} bytes): {text[:60]!r}")
        log(
            "tts",
            "edge play start",
            voice=_tts_voice_edge(),
            bytes=size,
            text=text[:80],
        )
        proc = subprocess.Popen(
            ["afplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        with _tts_lock:
            _tts_proc = proc
            _tts_speaking = True
        if on_start:
            try:
                on_start()
            except Exception as e:
                print(f"TTS on_start error: {e}")
                log_exc("tts", "on_start", e)
        rc = proc.wait()
        err = ""
        try:
            if proc.stderr:
                err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore")
        except Exception:
            pass
        if rc != 0:
            print(f"afplay failed rc={rc} {err[:200]}")
            log("tts", "afplay failed", rc=rc, err=err[:120])
            return False
        print("[TTS/edge] done")
        log("tts", "edge play done")
        return True
    except Exception as e:
        print(f"edge-tts error: {e}")
        log_exc("tts", "edge error", e)
        return False
    finally:
        with _tts_lock:
            if _tts_proc is not None:
                _tts_proc = None
            _tts_speaking = False
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
        _tts_tmp = None


def _speak_say(
    text: str,
    voice: str,
    on_start: Optional[Callable[[], None]],
) -> bool:
    global _tts_speaking, _tts_proc
    try:
        proc = subprocess.Popen(
            ["say", "-v", voice, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print(f"[TTS] {text}")
        return False
    print(f"[TTS/say] {voice}: {text[:60]!r}")
    log("tts", "say play start", voice=voice, text=text[:80])
    with _tts_lock:
        _tts_proc = proc
        _tts_speaking = True
    if on_start:
        try:
            on_start()
        except Exception as e:
            print(f"TTS on_start error: {e}")
            log_exc("tts", "on_start", e)
    try:
        rc = proc.wait()
        if rc != 0:
            err = ""
            try:
                if proc.stderr:
                    err = (proc.stderr.read() or b"").decode(
                        "utf-8", errors="ignore"
                    )
            except Exception:
                pass
            print(f"say failed rc={rc} {err[:200]}")
            log("tts", "say failed", rc=rc, err=err[:120])
            return False
        log("tts", "say play done")
        return True
    finally:
        with _tts_lock:
            if _tts_proc is proc:
                _tts_proc = None
            _tts_speaking = False


def speak(
    text: str,
    voice: Optional[str] = None,
    blocking: bool = False,
    on_start: Optional[Callable[[], None]] = None,
    on_end: Optional[Callable[[], None]] = None,
    local: bool = False,
) -> None:
    """TTS with barge-in via stop_speaking(). Prefer edge-tts neural English.

    ``local=True`` skips cloud edge-tts and uses the offline macOS ``say``
    voice directly — for short system phrases (boot, error) so they are instant
    and never hang ~25s on a slow/blocked network before falling back.
    """
    text = (text or "").strip()
    if not text:
        return

    def _run():
        stop_speaking()
        provider = _tts_provider()
        log(
            "tts",
            "speak requested",
            provider=provider,
            blocking=int(blocking),
            chars=len(text),
            local=int(local),
        )
        ok = False
        if provider in ("edge", "auto") and not local:
            ok = _speak_edge(text, on_start)
        if not ok:
            for v in (
                voice,
                os.getenv("TTS_SAY_VOICE"),
                "Samantha",
                "Ava",
            ):
                if not v:
                    continue
                if _speak_say(text, v, on_start):
                    ok = True
                    break
        if not ok:
            print(f"[TTS/print] {text}")
            log("tts", "fallback print only (no audio backend)", text=text[:80])
            if on_start:
                try:
                    on_start()
                except Exception:
                    pass
        if on_end:
            try:
                on_end()
            except Exception as e:
                log_exc("tts", "on_end", e)

    if blocking:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()
