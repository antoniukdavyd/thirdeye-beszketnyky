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
