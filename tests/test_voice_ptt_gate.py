import asyncio

from deepgram.agent.v1.types import AgentV1Error

from assist.channel.deepgram_agent import (
    PTT_SILENCE_CHUNK,
    DeepgramVoiceSession,
    MicGate,
    RealtimeState,
)


def test_sender_forwards_mic_only_while_armed():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session.mic_gate.arm()
    assert session._sender_action(b"mic pcm", now=100.0, last_activity=100.0) == ("media", b"mic pcm")
    assert session._sender_action(None, now=100.0, last_activity=100.0) == ("idle", None)


def test_sender_sends_silence_tail_for_endpointing_after_release():
    # Right after PTT release, a short silence tail helps Deepgram endpoint the
    # user turn. _silence_until is set by disarm_listen().
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._silence_until = 100.5
    action, payload = session._sender_action(None, now=100.2, last_activity=100.0)
    assert action == "silence"
    assert payload == PTT_SILENCE_CHUNK


def test_sender_sends_keepalive_when_idle_between_turns():
    # The real bug: during a long idle gap, silence does NOT reset Deepgram's
    # inactivity timer (it wants user speech). A KeepAlive message does.
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._silence_until = None
    # <5s since last activity → nothing
    assert session._sender_action(None, now=103.0, last_activity=100.0) == ("idle", None)
    # >=5s since last activity → keepalive
    assert session._sender_action(None, now=105.0, last_activity=100.0) == ("keepalive", None)


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


def test_run_loop_announces_network_problem_on_connection_error(monkeypatch):
    spoken = []
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.mic_gate.arm()
    monkeypatch.setattr(
        "assist.channel.deepgram_agent.asyncio.run",
        lambda coro: (coro.close(), (_ for _ in ()).throw(OSError("offline")))[1],
    )
    monkeypatch.setattr(
        "assist.channel.deepgram_agent.speak",
        lambda text, **kw: spoken.append(text),
    )

    session._run_loop()

    # OSError = network problem, distinct phrase (not "speech is down").
    assert spoken == ["Network problem, check your connection."]
    assert session.active is False
    assert session.mic_gate.armed is False
    assert session.state == RealtimeState.IDLE


def test_agent_error_speaks_and_ends_session(monkeypatch):
    spoken = []
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.mic_gate.arm()
    monkeypatch.setattr(
        "assist.channel.deepgram_agent.speak",
        lambda text, **kw: spoken.append(text),
    )
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
