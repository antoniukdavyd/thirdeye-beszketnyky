"""Deepgram Voice Agent session with push-to-talk microphone gating."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from enum import Enum
from typing import Callable, Optional

from ..agent import (
    SYSTEM_INSTRUCTION,
    deepgram_api_key,
    deepgram_stt_model,
    deepgram_tts_model,
    openrouter_agent_model,
    openrouter_api_key,
    voice_ready,
)
from ..debuglog import log, log_exc
from ..tools import ToolRegistry, deepgram_think_functions
from ..voice.audio import speak, stop_speaking

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
LISTEN_TIMEOUT_SECONDS = 12.0


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
        self._session_active = False
        self._listen_deadline: Optional[float] = None
        self._play_buf = bytearray()
        self._play_lock = threading.Lock()
        self._audio_done = False
        self._last_phrase = ""

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

    def set_last_phrase(self, text: str) -> None:
        self._last_phrase = text or ""

    def _set_state(self, state: RealtimeState) -> None:
        self.state = state
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
        self.mic_gate.disarm()
        self._session_active = True
        self._set_state(RealtimeState.CONNECTING)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log("realtime", "Deepgram session starting")

    def end_session(self, announce: bool = True) -> None:
        self._stop.set()
        self.mic_gate.disarm()
        self._listen_deadline = None
        self._session_active = False
        stop_speaking()
        with self._play_lock:
            self._play_buf.clear()
            self._audio_done = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._set_state(RealtimeState.IDLE)
        if announce:
            speak("Off")

    def arm_listen(self) -> None:
        if not self.active:
            return
        self.mic_gate.arm()
        self._listen_deadline = time.monotonic() + LISTEN_TIMEOUT_SECONDS
        self._set_state(RealtimeState.LISTENING)

    def disarm_listen(self) -> None:
        self.mic_gate.disarm()
        self._listen_deadline = None
        if self.active and self.state == RealtimeState.LISTENING:
            self._set_state(RealtimeState.IDLE)

    def request_ptt_barge_in(self) -> None:
        stop_speaking()
        with self._play_lock:
            if self.active:
                self.mic_gate.arm()
                self._listen_deadline = (
                    time.monotonic() + LISTEN_TIMEOUT_SECONDS
                )
            self._play_buf.clear()
            self._audio_done = False
            if self.active:
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
                return False
            self.mic_gate.disarm()
            self._audio_done = False
            self._play_buf.extend(pcm)
            self._set_state(RealtimeState.SPEAKING)
        return True

    def tick(self) -> None:
        return

    def _run_loop(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:
            log_exc("realtime", "Deepgram session failed", exc)
        finally:
            self._session_active = False
            self.mic_gate.disarm()
            self._set_state(RealtimeState.IDLE)

    def _settings(self):
        from deepgram.agent.v1.types import (
            AgentV1Settings,
            AgentV1SettingsAgent,
            AgentV1SettingsAgentListen,
            AgentV1SettingsAgentListenProvider_V1,
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
                listen=AgentV1SettingsAgentListen(
                    provider=AgentV1SettingsAgentListenProvider_V1(
                        model=deepgram_stt_model()
                    )
                ),
                think=think,
                speak=SpeakSettingsV1(
                    provider=SpeakSettingsV1Provider_Deepgram(
                        model=deepgram_tts_model()
                    )
                ),
            ),
        )

    async def _main(self) -> None:
        from deepgram import AsyncDeepgramClient
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

        def queue_audio(pcm: bytes) -> None:
            try:
                audio_queue.put_nowait(pcm)
            except asyncio.QueueFull:
                pass

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
            with self._play_lock:
                if self.mic_gate.armed:
                    self._play_buf.clear()
                    self._audio_done = False
                    chunk = b""
                else:
                    chunk = bytes(self._play_buf[:needed])
                    del self._play_buf[:needed]
                if self._audio_done and not self._play_buf:
                    self._audio_done = False
                    drained = True
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
        send_task = None
        receive_task = None
        try:
            async with client.agent.v1.connect() as agent:
                await agent.send_settings(self._settings())
                input_stream.start()
                output_stream.start()
                self._set_state(
                    RealtimeState.LISTENING
                    if self.mic_gate.armed
                    else RealtimeState.IDLE
                )
                log("realtime", "Deepgram websocket connected")

                async def sender() -> None:
                    while not self._stop.is_set():
                        if (
                            self.mic_gate.armed
                            and self._listen_deadline is not None
                            and time.monotonic() >= self._listen_deadline
                        ):
                            self.disarm_listen()
                        try:
                            pcm = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                        except asyncio.TimeoutError:
                            continue
                        if self.mic_gate.armed:
                            await agent.send_media(pcm)

                async def receiver() -> None:
                    async for message in agent:
                        if self._stop.is_set():
                            break
                        await self._handle_message(agent, message)

                send_task = asyncio.create_task(sender())
                receive_task = asyncio.create_task(receiver())
                while not self._stop.is_set():
                    await asyncio.sleep(0.1)
                    if send_task.done() or receive_task.done():
                        break
        finally:
            self._stop.set()
            for task in (send_task, receive_task):
                if task is not None and not task.done():
                    task.cancel()
            for stream in (input_stream, output_stream):
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    async def _handle_message(self, agent, message) -> None:
        from deepgram.agent.v1.types import (
            AgentV1AgentAudioDone,
            AgentV1AgentThinking,
            AgentV1ConversationText,
            AgentV1FunctionCallRequest,
            AgentV1SendFunctionCallResponse,
        )

        if isinstance(message, bytes):
            self._queue_assistant_audio(message)
            return
        if isinstance(message, AgentV1ConversationText) and message.role == "user":
            self.disarm_listen()
            self._set_state(RealtimeState.THINKING)
            return
        if isinstance(message, AgentV1AgentThinking):
            self.disarm_listen()
            self._set_state(RealtimeState.THINKING)
            return
        if isinstance(message, AgentV1AgentAudioDone):
            with self._play_lock:
                drained = not self._play_buf
                self._audio_done = not drained
            if drained:
                self._playback_drained()
            return
        if isinstance(message, AgentV1FunctionCallRequest):
            content = json.dumps({"ok": False, "error": "tools not wired"})
            for function in message.functions:
                await agent.send_function_call_response(
                    AgentV1SendFunctionCallResponse(
                        id=function.id,
                        name=function.name,
                        content=content,
                    )
                )

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
