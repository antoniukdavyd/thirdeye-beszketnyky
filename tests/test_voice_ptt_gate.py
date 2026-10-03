import asyncio

from deepgram.agent.v1.types import AgentV1Error

from assist.channel.deepgram_agent import (
    DeepgramVoiceSession,
    MicGate,
    RealtimeState,
)


def test_mic_gate_defaults_closed():
    gate = MicGate()
    assert gate.armed is False
    gate.arm()
    assert gate.armed is True
    gate.disarm()
    assert gate.armed is False


def test_assistant_audio_is_dropped_while_ptt_is_armed():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session.mic_gate.arm()

    assert session._queue_assistant_audio(b"assistant pcm") is False
    assert session.mic_gate.armed is True
    assert session._play_buf == b""
    assert session.state == RealtimeState.IDLE


def test_first_accepted_assistant_audio_disarms_mic_and_starts_speaking():
    session = DeepgramVoiceSession(tools=None, enable=False)

    assert session._queue_assistant_audio(b"assistant pcm") is True
    assert session.mic_gate.armed is False
    assert session._play_buf == b"assistant pcm"
    assert session.state == RealtimeState.SPEAKING


def test_ptt_barge_in_clears_playback_and_returns_to_listening(monkeypatch):
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session._play_buf.extend(b"assistant pcm")
    session._audio_done = True
    session._set_state(RealtimeState.SPEAKING)
    monkeypatch.setattr(
        "assist.channel.deepgram_agent.stop_speaking", lambda: None
    )

    session.request_ptt_barge_in()

    assert session._play_buf == b""
    assert session._audio_done is False
    assert session.mic_gate.armed is True
    assert session.state == RealtimeState.LISTENING


def test_run_loop_speaks_when_session_fails(monkeypatch):
    spoken = []
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.mic_gate.arm()
    monkeypatch.setattr(
        "assist.channel.deepgram_agent.asyncio.run",
        lambda coro: (coro.close(), (_ for _ in ()).throw(OSError("offline")))[1],
    )
    monkeypatch.setattr("assist.channel.deepgram_agent.speak", spoken.append)

    session._run_loop()

    assert spoken == ["Speech is down, try again."]
    assert session.active is False
    assert session.mic_gate.armed is False
    assert session.state == RealtimeState.IDLE


def test_agent_error_speaks_and_ends_session(monkeypatch):
    spoken = []
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.mic_gate.arm()
    monkeypatch.setattr("assist.channel.deepgram_agent.speak", spoken.append)
    error = AgentV1Error(
        code="OPENROUTER_ERROR",
        description="OpenRouter completion failed",
    )

    asyncio.run(session._handle_message(None, error))

    assert spoken == ["I can't think right now."]
    assert session._stop.is_set()
    assert session.active is False
    assert session.mic_gate.armed is False
    assert session.state == RealtimeState.IDLE
