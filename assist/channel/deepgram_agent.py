"""Deepgram Voice Agent session with push-to-talk microphone gating."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from enum import Enum
from typing import Callable, Optional, Set

from ..agent import (
    SYSTEM_INSTRUCTION,
    deepgram_api_key,
    deepgram_listen_version,
    deepgram_stt_model,
    deepgram_tts_model,
    openrouter_agent_model,
    openrouter_api_key,
    voice_ready,
)
from ..debuglog import log, log_exc
from ..tools import ToolRegistry, deepgram_think_functions, parse_tool_arguments
from ..voice.audio import speak, stop_speaking

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
LISTEN_TIMEOUT_SECONDS = 12.0
# A physical Space hold is bounded separately: the 12 s toggle-mode cap used to
# cut the mic out from under a user who was still talking.
HOLD_MAX_SECONDS = 45.0
# After Space release, keep sending silence so Deepgram can end the user turn.
PTT_RELEASE_SILENCE_SEC = 0.45
PTT_SILENCE_CHUNK = bytes(int(INPUT_RATE * 0.04) * 2)  # 40 ms mono s16le
# Send a KeepAlive message if no audio has flowed for this long, so the
# Deepgram Voice Agent socket survives idle gaps between PTT turns.
KEEPALIVE_INTERVAL_SEC = 5.0

MIC_CHUNK_BYTES = int(INPUT_RATE * 0.04) * 2
# ~6 s of mic backlog. Deep enough to hold the pre-roll a user speaks while the
# websocket is still coming up, and to ride out a network hiccup without
# silently throwing away the middle of an utterance.
MIC_QUEUE_MAX = 150
# Coalesce a backlog into one send instead of trickling it out one 40 ms frame
# per loop pass, which is how a hiccup used to snowball into seconds of lag.
MIC_BATCH_MAX = 25

# Hold back assistant playback until this much audio has landed. The websocket
# delivers TTS in bursts; draining it the instant the first frame arrives means
# the output callback hits an empty buffer and emits zero-fill, which the user
# hears as choppy, clicking speech.
PLAY_PREBUFFER_BYTES = int(OUTPUT_RATE * 0.20) * 2
# Pure leak guard — far longer than any clipped reply.
PLAY_MAX_BYTES = int(OUTPUT_RATE * 30.0) * 2

# A dropped socket mid-walk must not end the session silently: reconnect
# quietly and only announce once recovery is genuinely out of attempts.
RECONNECT_MAX_ATTEMPTS = 5
RECONNECT_BACKOFF_BASE = 0.25
RECONNECT_BACKOFF_CAP = 2.0

# Safety net for the readiness gate: if SettingsApplied never shows up (server
# change, dropped frame), start sending anyway rather than leaving a blind
# user's microphone permanently muted.
READY_GRACE_SEC = 3.0

# Upstream bug, reproduced against the live API (see docs): after a client-side
# FunctionCallResponse, Deepgram reports ttt_text_latency but never emits the
# assistant ConversationText, never calls TTS, and never sends audio — the turn
# just dies silently. Tool results themselves arrive fine, so rather than leave
# the user with no answer, speak the tool's own phrase locally if the agent has
# said nothing this long after we answered its function call.
TOOL_REPLY_TIMEOUT_SEC = 5.0

_FATAL_AUTH_MARKERS = ("invalid credentials", "401", "403", "unauthorized")


class RealtimeState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LISTENING = "listening"
    SPEAKING = "speaking"
    THINKING = "thinking"


class MicGate:
    """Thread-safe push-to-talk gate."""

    def __init__(self) -> None:
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> None:
        with self._lock:
            self._armed = True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False


def dispatch_function_calls(tools: ToolRegistry, calls: list) -> list[dict]:
    responses = []
    for call in calls:
        if getattr(call, "client_side", True) is False:
            continue
        name = call.name
        t0 = time.monotonic()
        args = parse_tool_arguments(getattr(call, "arguments", {}) or {})
        result = tools.execute(name, args)
        dt = (time.monotonic() - t0) * 1000.0
        log("tool_call", name, ms=round(dt, 1))
        responses.append(
            {
                "id": call.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return responses


class DeepgramVoiceSession:
    """Own a Deepgram Agent websocket, microphone, and PCM playback stream."""

    def __init__(
        self,
        tools: ToolRegistry,
        on_state: Optional[Callable[[str], None]] = None,
        enable: bool = True,
    ) -> None:
        self.tools = tools
        self.on_state = on_state
        self.enable = enable
        self.state = RealtimeState.IDLE
        self.mic_gate = MicGate()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        # Set once Deepgram has accepted Settings. Media sent before that is
        # discarded by the server, which is how the first PTT turn used to be
        # lost while the socket was still coming up.
        self._ready = threading.Event()
        self._session_active = False
        self._listen_deadline: Optional[float] = None
        self._silence_until: Optional[float] = None
        # True while Space is physically held. Server-side turn events must not
        # cut the mic out from under a user who is still talking.
        self._hold = False
        self._end_turn_pending = False
        self._play_buf = bytearray()
        self._play_lock = threading.Lock()
        self._audio_done = False
        self._playing = False
        self._last_phrase = ""
        self._failure_announced = False
        self._mic_drops = 0
        self._tasks: Set[asyncio.Task] = set()
        # ForceEndTurn only exists on the v2 (Flux) listen provider; on nova-*
        # the server replies FORCE_END_TURN_UNSUPPORTED and leaves the turn open.
        self._force_end_turn_ok = deepgram_listen_version() == "v2"
        self._awaiting_reply_since: Optional[float] = None
        self._fallback_phrase = ""

    @property
    def available(self) -> bool:
        if not self.enable or not voice_ready():
            return False
        try:
            import deepgram  # noqa: F401
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def active(self) -> bool:
        return self._session_active

    @property
    def ready(self) -> bool:
        """True once the socket is up and Settings were accepted."""
        return self._ready.is_set()

    def set_last_phrase(self, text: str) -> None:
        self._last_phrase = text or ""

    def _set_state(self, state: RealtimeState) -> None:
        # Only log real transitions — assistant audio arrives in many small
        # chunks and would otherwise spam "state → speaking" hundreds of times.
        changed = state != self.state
        self.state = state
        if changed:
            log("realtime", f"state → {state.value}")
        if self.on_state:
            try:
                self.on_state(state.value)
            except Exception:
                pass

    def start_session(self) -> None:
        """Open the Agent websocket, leaving the PTT microphone gate closed."""
        if self.active:
            return
        if not self.available:
            speak("Realtime unavailable. Check Deepgram and OpenRouter API keys.")
            return
        self._stop.clear()
        self._ready.clear()
        self._failure_announced = False
        self.mic_gate.disarm()
        self._session_active = True
        self._set_state(RealtimeState.CONNECTING)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log("realtime", "Deepgram session starting")

    def prewarm(self) -> None:
        """Open the socket at boot so the first PTT press has a live session.

        Connecting takes a websocket handshake plus a Settings round-trip. Doing
        that on the first Space press meant the user's first sentence was spoken
        into a socket that did not exist yet — the "it only works on the second
        or third try" symptom.
        """
        if self.active or not self.available:
            return
        if (os.getenv("VOICE_PREWARM") or "1").strip().lower() in ("0", "false", "no"):
            log("realtime", "prewarm disabled by VOICE_PREWARM")
            return
        log("realtime", "prewarming Deepgram session")
        self.start_session()

    def end_session(self, announce: bool = True) -> None:
        self._stop.set()
        self._ready.clear()
        self.mic_gate.disarm()
        self._listen_deadline = None
        self._hold = False
        self._session_active = False
        stop_speaking()
        with self._play_lock:
            self._play_buf.clear()
            self._audio_done = False
            self._playing = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._set_state(RealtimeState.IDLE)
        if announce:
            speak("Off")

    def arm_listen(self, hold: bool = False) -> None:
        if not self.active:
            return
        self._hold = hold
        self._end_turn_pending = False
        self.mic_gate.arm()
        self._listen_deadline = time.monotonic() + self._listen_budget()
        # Don't claim "listening" before the socket can carry audio; the mic
        # keeps buffering either way and the sender flushes it on ready.
        self._set_state(
            RealtimeState.LISTENING if self.ready else RealtimeState.CONNECTING
        )

    def _listen_budget(self) -> float:
        return HOLD_MAX_SECONDS if self._hold else LISTEN_TIMEOUT_SECONDS

    def disarm_listen(self) -> None:
        self.mic_gate.disarm()
        self._listen_deadline = None
        self._hold = False
        # Tail silence helps Deepgram endpoint the utterance after PTT release,
        # and an explicit ForceEndTurn tells it not to wait for more speech.
        self._silence_until = time.monotonic() + PTT_RELEASE_SILENCE_SEC
        self._end_turn_pending = self._force_end_turn_ok
        if self.active and self.state == RealtimeState.LISTENING:
            self._set_state(RealtimeState.IDLE)

    def _server_end_of_turn(self) -> None:
        """Deepgram thinks the user turn ended. Honour it only if Space is up.

        Deepgram endpoints on a pause. Trusting that while the key is still held
        killed the mic mid-sentence, and the release handler then had nothing to
        do — so the rest of the question was never sent and the user had to let
        go and start over.
        """
        if self._hold:
            return
        self.mic_gate.disarm()
        self._listen_deadline = None

    def request_ptt_barge_in(self, hold: bool = False) -> None:
        stop_speaking()
        with self._play_lock:
            if self.active:
                self._hold = hold
                self._end_turn_pending = False
                self.mic_gate.arm()
                self._listen_deadline = time.monotonic() + self._listen_budget()
            self._play_buf.clear()
            self._audio_done = False
            self._playing = False
            if self.active:
                # Barge-in only happens over assistant audio, so the socket is
                # live by definition — no need to gate on readiness here.
                self._set_state(RealtimeState.LISTENING)
        log("realtime", "PTT barge-in")

    def _queue_assistant_audio(self, pcm: bytes) -> bool:
        """Queue assistant PCM only when no user listening turn is armed."""
        if not pcm:
            return False
        with self._play_lock:
            if self.mic_gate.armed:
                self._play_buf.clear()
                self._audio_done = False
                self._playing = False
                return False
            self.mic_gate.disarm()
            self._audio_done = False
            if len(self._play_buf) < PLAY_MAX_BYTES:
                self._play_buf.extend(pcm)
            self._set_state(RealtimeState.SPEAKING)
        return True

    def _sender_action(
        self, pcm: Optional[bytes], now: float, last_activity: float
    ) -> tuple[str, Optional[bytes]]:
        """Decide what the sender does this tick. Pure → unit-testable.

        Returns (action, payload):
          - ("media", pcm):   armed, forward captured mic audio
          - ("end_turn", None): Space was just released — tell Deepgram the user
                                 turn is over instead of waiting for its own
                                 endpointing to notice
          - ("silence", chunk): just after PTT release, short tail to help
                                 Deepgram endpoint the user turn
          - ("keepalive", None): idle between turns — send a real KeepAlive
                                 message. Deepgram's inactivity timer ignores
                                 silent audio (it wants user speech), so silence
                                 does NOT keep the socket alive; KeepAlive does.
          - ("idle", None):    nothing to send this tick
        """
        if self.mic_gate.armed:
            return ("media", pcm) if pcm else ("idle", None)
        if self._end_turn_pending:
            return ("end_turn", None)
        if self._silence_until is not None and now < self._silence_until:
            return ("silence", PTT_SILENCE_CHUNK)
        if now - last_activity >= KEEPALIVE_INTERVAL_SEC:
            return ("keepalive", None)
        return ("idle", None)

    def tick(self) -> None:
        return

    def _should_reconnect(self, exc: BaseException, attempt: int) -> bool:
        """Transient socket loss is recoverable; auth and shutdown are not."""
        if self._stop.is_set():
            return False
        if isinstance(exc, asyncio.CancelledError):
            return False
        if attempt > RECONNECT_MAX_ATTEMPTS:
            return False
        detail = f"{type(exc).__name__} {exc}".lower()
        if any(marker in detail for marker in _FATAL_AUTH_MARKERS):
            return False
        return True

    def _reconnect_backoff(self, attempt: int) -> float:
        step = RECONNECT_BACKOFF_BASE * (2 ** max(0, attempt - 1))
        return min(RECONNECT_BACKOFF_CAP, step)

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:
            log_exc("realtime", "Deepgram session failed", exc)
            # Timeout / connection errors are a network problem, not a broken
            # app — say so distinctly instead of the generic "speech is down".
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, OSError)):
                self._fail_session("Network problem, check your connection.")
            else:
                self._fail_session("Speech is down, try again.")
        finally:
            self._ready.clear()
            self._session_active = False
            self.mic_gate.disarm()
            self._set_state(RealtimeState.IDLE)

    def _fail_session(self, phrase: str) -> None:
        """Close the failed voice turn and announce it at most once."""
        self._stop.set()
        self._ready.clear()
        self._session_active = False
        self.mic_gate.disarm()
        self._listen_deadline = None
        self._hold = False
        self._set_state(RealtimeState.IDLE)
        if not self._failure_announced:
            self._failure_announced = True
            # Local TTS: the failure may itself be a network outage, so don't
            # try (and hang on) cloud TTS to announce it.
            speak(phrase, local=True)

    def _settings(self):
        from deepgram.agent.v1.types import (
            AgentV1Settings,
            AgentV1SettingsAgent,
            AgentV1SettingsAgentListen,
            AgentV1SettingsAgentListenProvider_V1,
            AgentV1SettingsAgentListenProvider_V2,
            AgentV1SettingsAudio,
            AgentV1SettingsAudioInput,
            AgentV1SettingsAudioOutput,
        )
        from deepgram.types import (
            SpeakSettingsV1,
            SpeakSettingsV1Provider_Deepgram,
            ThinkSettingsV1,
            ThinkSettingsV1Endpoint,
            ThinkSettingsV1FunctionsItem,
            ThinkSettingsV1Provider_OpenAi,
        )

        headers = {"Authorization": f"Bearer {openrouter_api_key()}"}
        referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        functions = [
            ThinkSettingsV1FunctionsItem(**definition)
            for definition in deepgram_think_functions(self.tools)
        ]
        think = ThinkSettingsV1(
            provider=ThinkSettingsV1Provider_OpenAi(model=openrouter_agent_model()),
            endpoint=ThinkSettingsV1Endpoint(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
            ),
            prompt=SYSTEM_INSTRUCTION,
            functions=functions,
        )
        if deepgram_listen_version() == "v2":
            listen_provider = AgentV1SettingsAgentListenProvider_V2(
                model=deepgram_stt_model()
            )
        else:
            listen_provider = AgentV1SettingsAgentListenProvider_V1(
                model=deepgram_stt_model()
            )
        return AgentV1Settings(
            audio=AgentV1SettingsAudio(
                input=AgentV1SettingsAudioInput(
                    encoding="linear16", sample_rate=INPUT_RATE
                ),
                output=AgentV1SettingsAudioOutput(
                    encoding="linear16",
                    sample_rate=OUTPUT_RATE,
                    container="none",
                ),
            ),
            agent=AgentV1SettingsAgent(
                listen=AgentV1SettingsAgentListen(provider=listen_provider),
                think=think,
                speak=SpeakSettingsV1(
                    provider=SpeakSettingsV1Provider_Deepgram(
                        model=deepgram_tts_model()
                    )
                ),
            ),
        )

    def _spawn(self, coro) -> None:
        """Run work off the websocket read loop, keeping a strong task ref."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _main(self) -> None:
        from deepgram import AsyncDeepgramClient
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)

        def queue_audio(pcm: bytes) -> None:
            try:
                audio_queue.put_nowait(pcm)
                return
            except asyncio.QueueFull:
                pass
            # Full means we are behind. Drop the *oldest* frame: losing the
            # start of a backlog beats refusing the live audio and gouging a
            # hole in the middle of what the user is saying right now.
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(pcm)
            except Exception:
                pass
            self._mic_drops += 1
            if self._mic_drops % 25 == 1:
                log("realtime", "mic backlog dropping frames", drops=self._mic_drops)

        def mic_callback(indata, frames, time_info, status):  # noqa: ARG001
            if self._stop.is_set() or not self.mic_gate.armed:
                return
            try:
                self._loop.call_soon_threadsafe(queue_audio, bytes(indata))
            except RuntimeError:
                pass

        def output_callback(outdata, frames, time_info, status):  # noqa: ARG001
            needed = frames * 2
            drained = False
            chunk = b""
            # Never block a CoreAudio realtime callback on a Python lock: if the
            # websocket thread holds it, emit one silent block instead of
            # stalling the device (which starves the whole audio graph).
            if self._play_lock.acquire(blocking=False):
                try:
                    if self.mic_gate.armed:
                        self._play_buf.clear()
                        self._audio_done = False
                        self._playing = False
                    else:
                        if not self._playing and (
                            len(self._play_buf) >= PLAY_PREBUFFER_BYTES
                            or self._audio_done
                        ):
                            self._playing = True
                        if self._playing:
                            chunk = bytes(self._play_buf[:needed])
                            del self._play_buf[:needed]
                    if self._audio_done and not self._play_buf:
                        self._audio_done = False
                        self._playing = False
                        drained = True
                finally:
                    self._play_lock.release()
            outdata[:] = chunk + (b"\x00" * (needed - len(chunk)))
            if drained:
                try:
                    self._loop.call_soon_threadsafe(self._playback_drained)
                except RuntimeError:
                    pass

        input_stream = sd.RawInputStream(
            samplerate=INPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(INPUT_RATE * 0.04),
            callback=mic_callback,
        )
        output_stream = sd.RawOutputStream(
            samplerate=OUTPUT_RATE,
            channels=1,
            dtype="int16",
            blocksize=int(OUTPUT_RATE * 0.04),
            callback=output_callback,
        )
        client = AsyncDeepgramClient(api_key=deepgram_api_key())
        # Open the devices once and keep them across reconnects: a PortAudio
        # open/close per attempt is slow and perturbs the shared output device.
        input_stream.start()
        output_stream.start()
        try:
            attempt = 0
            while not self._stop.is_set():
                attempt += 1
                try:
                    await self._session_once(client, audio_queue)
                    if self._stop.is_set():
                        return
                    # Clean close with no stop request: the server hung up.
                    exc: BaseException = ConnectionError("agent socket closed")
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 — classified below
                    exc = e
                if not self._should_reconnect(exc, attempt):
                    raise exc
                delay = self._reconnect_backoff(attempt)
                log_exc("realtime", f"reconnecting in {delay:.2f}s", exc)
                self._ready.clear()
                self._set_state(RealtimeState.CONNECTING)
                await asyncio.sleep(delay)
        finally:
            self._stop.set()
            self._ready.clear()
            for stream in (input_stream, output_stream):
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    async def _session_once(self, client, audio_queue: "asyncio.Queue[bytes]") -> None:
        """One websocket lifetime: connect, pump audio, return when it ends."""
        send_task = None
        receive_task = None
        try:
            async with client.agent.v1.connect() as agent:
                await agent.send_settings(self._settings())
                log("realtime", "Deepgram websocket connected")

                send_task = asyncio.create_task(self._sender(agent, audio_queue))
                receive_task = asyncio.create_task(self._receiver(agent))
                done, _ = await asyncio.wait(
                    {send_task, receive_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # Surface a socket error instead of treating it as a clean exit.
                for task in done:
                    task.result()
        finally:
            pending = [
                task
                for task in (send_task, receive_task, *tuple(self._tasks))
                if task is not None
            ]
            for task in pending:
                if not task.done():
                    task.cancel()
            self._tasks.clear()
            # Collect them so a cancelled pump cannot surface later as an
            # "exception was never retrieved" warning mid-walk.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _sender(self, agent, audio_queue: "asyncio.Queue[bytes]") -> None:
        # Media sent before SettingsApplied is discarded server-side. Wait, but
        # let the mic keep filling the queue so the pre-roll survives and gets
        # flushed the moment the agent is ready.
        grace = time.monotonic() + READY_GRACE_SEC
        while not self._stop.is_set() and not self._ready.is_set():
            if time.monotonic() >= grace:
                log("realtime", "no SettingsApplied within grace — sending anyway")
                self._on_ready()
                break
            await asyncio.sleep(0.02)
        if self._stop.is_set():
            return
        last_activity = time.monotonic()
        while not self._stop.is_set():
            if (
                self.mic_gate.armed
                and self._listen_deadline is not None
                and time.monotonic() >= self._listen_deadline
            ):
                log("realtime", "listen window expired")
                self.disarm_listen()
            try:
                pcm = await asyncio.wait_for(audio_queue.get(), timeout=0.04)
            except asyncio.TimeoutError:
                pcm = None
            if pcm is not None:
                pcm = self._drain_mic_backlog(audio_queue, pcm)
            now = time.monotonic()
            if self._reply_watchdog_due(now):
                self._speak_tool_fallback()
            action, payload = self._sender_action(pcm, now, last_activity)
            if action == "media":
                await agent.send_media(payload)
                last_activity = now
            elif action == "end_turn":
                self._end_turn_pending = False
                try:
                    await agent.send_force_end_turn()
                except Exception as exc:
                    log_exc("realtime", "force_end_turn failed", exc)
                last_activity = now
            elif action == "silence":
                await agent.send_media(payload)
                last_activity = now
            elif action == "keepalive":
                # Deepgram-sanctioned idle keepalive (silence does not reset
                # its inactivity timer).
                await agent.send_keep_alive()
                last_activity = now

    @staticmethod
    def _drain_mic_backlog(audio_queue: "asyncio.Queue[bytes]", pcm: bytes) -> bytes:
        parts = [pcm]
        for _ in range(MIC_BATCH_MAX):
            try:
                parts.append(audio_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return parts[0] if len(parts) == 1 else b"".join(parts)

    async def _receiver(self, agent) -> None:
        async for message in agent:
            if self._stop.is_set():
                return
            await self._handle_message(agent, message)

    async def _handle_message(self, agent, message) -> None:
        """Classify one agent message. Must never await slow work.

        The websocket client stops reading the socket once 32 messages are
        queued, and pongs are parsed by that same reader — so blocking here
        long enough to fill the queue makes the client time out its own
        keepalive ping and tear down a healthy connection. That is why tool
        calls are dispatched as separate tasks.
        """
        from deepgram.agent.v1.types import (
            AgentV1AgentAudioDone,
            AgentV1AgentStartedSpeaking,
            AgentV1AgentThinking,
            AgentV1ConversationText,
            AgentV1Error,
            AgentV1FunctionCallRequest,
            AgentV1SettingsApplied,
            AgentV1Warning,
            AgentV1Welcome,
        )

        if isinstance(message, bytes):
            self._clear_reply_watchdog()
            self._queue_assistant_audio(message)
            return
        if isinstance(message, AgentV1SettingsApplied):
            self._on_ready()
            return
        if isinstance(message, AgentV1Welcome):
            log("realtime", "agent welcome")
            return
        if isinstance(message, AgentV1Warning):
            code = getattr(message, "code", "") or ""
            log(
                "realtime",
                "AgentV1Warning",
                code=code,
                desc=(getattr(message, "description", "") or "")[:200],
            )
            if "FORCE_END_TURN_UNSUPPORTED" in code.upper():
                # Fall back to the silence tail for the rest of the session.
                self._force_end_turn_ok = False
                self._end_turn_pending = False
            return
        if isinstance(message, AgentV1Error):
            code = getattr(message, "code", "") or ""
            desc = getattr(message, "description", "") or ""
            log("realtime", "AgentV1Error", code=code, desc=desc[:240])
            detail = f"{code} {desc}".lower()
            thinking_failed = any(
                term in detail
                for term in (
                    "openrouter",
                    "llm",
                    "completion",
                    "think",
                    "unparsable",
                    "settings",
                )
            )
            self._fail_session(
                "I can't think right now."
                if thinking_failed
                else "Speech is down, try again."
            )
            return
        if isinstance(message, AgentV1ConversationText) and message.role != "user":
            self._clear_reply_watchdog()
            return
        if isinstance(message, AgentV1ConversationText) and message.role == "user":
            self._server_end_of_turn()
            if not self._hold:
                self._set_state(RealtimeState.THINKING)
            return
        if isinstance(message, AgentV1AgentThinking):
            self._server_end_of_turn()
            if not self._hold:
                self._set_state(RealtimeState.THINKING)
            return
        if isinstance(message, AgentV1AgentStartedSpeaking):
            return
        if isinstance(message, AgentV1AgentAudioDone):
            with self._play_lock:
                drained = not self._play_buf
                self._audio_done = not drained
            if drained:
                self._playback_drained()
            return
        if isinstance(message, AgentV1FunctionCallRequest):
            self._spawn(self._run_function_calls(agent, list(message.functions)))

    def _on_ready(self) -> None:
        if self._ready.is_set():
            return
        self._ready.set()
        self._failure_announced = False
        log("realtime", "settings applied — agent ready")
        if self.mic_gate.armed:
            # The connect/settings round-trip must not eat the listen window.
            self._listen_deadline = time.monotonic() + self._listen_budget()
            self._set_state(RealtimeState.LISTENING)
        elif self.state == RealtimeState.CONNECTING:
            self._set_state(RealtimeState.IDLE)

    async def _run_function_calls(self, agent, functions: list) -> None:
        try:
            if self.tools is None:
                responses = [
                    {
                        "id": function.id,
                        "name": function.name,
                        "content": json.dumps(
                            {"ok": False, "error": "tools unavailable"},
                            ensure_ascii=False,
                        ),
                    }
                    for function in functions
                ]
            else:
                responses = await asyncio.to_thread(
                    dispatch_function_calls, self.tools, functions
                )
            for resp in responses:
                from deepgram.agent.v1.types import AgentV1SendFunctionCallResponse

                await agent.send_function_call_response(
                    AgentV1SendFunctionCallResponse(**resp)
                )
            self._arm_reply_watchdog(responses)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_exc("realtime", "function call dispatch failed", exc)

    def _arm_reply_watchdog(self, responses: list[dict]) -> None:
        """Remember what to say if the agent never speaks after a tool call."""
        phrase = ""
        if self.tools is not None:
            for resp in responses:
                try:
                    result = json.loads(resp.get("content") or "{}")
                    spoken = self.tools.spoken_from_tool(resp.get("name", ""), result)
                except Exception:
                    continue
                if spoken:
                    phrase = spoken
        self._fallback_phrase = phrase
        self._awaiting_reply_since = time.monotonic()

    def _clear_reply_watchdog(self) -> None:
        self._awaiting_reply_since = None
        self._fallback_phrase = ""

    def _reply_watchdog_due(self, now: float) -> bool:
        """Pure → unit-testable. True when the agent owes us a reply and is late."""
        if self._awaiting_reply_since is None or not self._fallback_phrase:
            return False
        if self.mic_gate.armed or self.state == RealtimeState.SPEAKING:
            return False
        return now - self._awaiting_reply_since >= TOOL_REPLY_TIMEOUT_SEC

    def _speak_tool_fallback(self) -> None:
        phrase = self._fallback_phrase
        self._clear_reply_watchdog()
        if not phrase:
            return
        log("realtime", "agent silent after tool call — speaking tool result", 
            text=phrase[:80])
        self.set_last_phrase(phrase)
        speak(phrase)

    def _playback_drained(self) -> None:
        if self.active and self.state == RealtimeState.SPEAKING:
            self._set_state(RealtimeState.IDLE)


def create_voice_channel(
    tools: ToolRegistry,
    on_state: Optional[Callable[[str], None]] = None,
    enable: bool = True,
) -> Optional[DeepgramVoiceSession]:
    if not enable or not voice_ready():
        log("realtime", "Deepgram/OpenRouter key missing — voice disabled")
        return None
    session = DeepgramVoiceSession(tools=tools, on_state=on_state, enable=True)
    if not session.available:
        log("realtime", "deepgram-sdk / sounddevice unavailable")
        return None
    return session
