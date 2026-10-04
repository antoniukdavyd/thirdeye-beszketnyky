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


def test_should_reconnect_on_network_error_within_attempt_cap():
    session = DeepgramVoiceSession(tools=None, enable=False)
    assert session._should_reconnect(TimeoutError(), attempt=1) is True
    assert session._should_reconnect(OSError("reset"), attempt=3) is True


def test_should_not_reconnect_after_attempt_cap():
    from assist.channel.deepgram_agent import RECONNECT_MAX_ATTEMPTS

    session = DeepgramVoiceSession(tools=None, enable=False)
    assert session._should_reconnect(TimeoutError(), attempt=RECONNECT_MAX_ATTEMPTS + 1) is False


def test_should_not_reconnect_when_stopping():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._stop.set()
    assert session._should_reconnect(TimeoutError(), attempt=1) is False


def test_should_not_reconnect_on_cancel():
    import asyncio as _asyncio

    session = DeepgramVoiceSession(tools=None, enable=False)
    assert session._should_reconnect(_asyncio.CancelledError(), attempt=1) is False


def test_reconnect_backoff_grows_and_caps():
    session = DeepgramVoiceSession(tools=None, enable=False)
    b1 = session._reconnect_backoff(1)
    b2 = session._reconnect_backoff(2)
    assert 0 < b1 <= b2
    assert session._reconnect_backoff(99) <= 2.0  # capped


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


def test_hold_survives_server_endpointing_mid_utterance():
    # Deepgram endpoints on a pause. While Space is physically down that must
    # not close the mic, or the rest of the sentence is never sent.
    from deepgram.agent.v1.types import AgentV1AgentThinking

    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session._ready.set()
    session.arm_listen(hold=True)

    asyncio.run(session._handle_message(None, AgentV1AgentThinking(content="thinking")))

    assert session.mic_gate.armed is True
    assert session.state == RealtimeState.LISTENING


def test_toggle_turn_still_ends_on_server_endpointing():
    from deepgram.agent.v1.types import AgentV1AgentThinking

    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session._ready.set()
    session.arm_listen(hold=False)

    asyncio.run(session._handle_message(None, AgentV1AgentThinking(content="thinking")))

    assert session.mic_gate.armed is False
    assert session.state == RealtimeState.THINKING


def test_release_asks_for_force_end_turn_when_provider_supports_it():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session._ready.set()
    session._force_end_turn_ok = True  # v2 / Flux listen provider
    session.arm_listen(hold=True)

    session.disarm_listen()

    assert session._hold is False
    # end_turn wins over the silence tail so Deepgram stops waiting for speech.
    assert session._sender_action(None, now=0.0, last_activity=0.0) == ("end_turn", None)


def test_release_uses_silence_tail_on_v1_listen_provider():
    # nova-* answers FORCE_END_TURN_UNSUPPORTED, so the turn must be endpointed
    # with trailing silence instead.
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session._ready.set()
    session._force_end_turn_ok = False
    session.arm_listen(hold=True)

    session.disarm_listen()

    action, payload = session._sender_action(None, now=0.0, last_activity=0.0)
    assert action == "silence"
    assert payload == PTT_SILENCE_CHUNK


def test_force_end_turn_warning_disables_further_attempts():
    from deepgram.agent.v1.types import AgentV1Warning

    session = DeepgramVoiceSession(tools=None, enable=False)
    session._force_end_turn_ok = True
    session._end_turn_pending = True
    warning = AgentV1Warning(
        code="FORCE_END_TURN_UNSUPPORTED",
        description="ForceEndTurn requires a Deepgram v2 (Flux) listen provider.",
    )

    asyncio.run(session._handle_message(None, warning))

    assert session._force_end_turn_ok is False
    assert session._end_turn_pending is False


def test_hold_gets_a_longer_listen_window_than_toggle():
    from assist.channel.deepgram_agent import (
        HOLD_MAX_SECONDS,
        LISTEN_TIMEOUT_SECONDS,
    )

    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.arm_listen(hold=False)
    assert session._listen_budget() == LISTEN_TIMEOUT_SECONDS
    session.arm_listen(hold=True)
    assert session._listen_budget() == HOLD_MAX_SECONDS
    assert HOLD_MAX_SECONDS > LISTEN_TIMEOUT_SECONDS


def test_arm_before_ready_reports_connecting_not_listening():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True

    session.arm_listen(hold=True)

    assert session.mic_gate.armed is True  # mic buffers the pre-roll
    assert session.state == RealtimeState.CONNECTING


def test_settings_applied_restarts_listen_window_and_goes_listening():
    session = DeepgramVoiceSession(tools=None, enable=False)
    session._session_active = True
    session.arm_listen(hold=True)
    stale = session._listen_deadline

    session._on_ready()

    assert session.ready is True
    assert session.state == RealtimeState.LISTENING
    # The connect round-trip must not eat into the user's listen window.
    assert session._listen_deadline >= stale


def test_playback_waits_for_prebuffer_then_drains():
    from assist.channel.deepgram_agent import PLAY_PREBUFFER_BYTES

    session = DeepgramVoiceSession(tools=None, enable=False)
    session._queue_assistant_audio(b"\x00" * 64)
    assert session._playing is False  # too little audio to start cleanly
    session._queue_assistant_audio(b"\x00" * PLAY_PREBUFFER_BYTES)
    assert len(session._play_buf) >= PLAY_PREBUFFER_BYTES


def test_assistant_audio_is_capped_to_avoid_unbounded_growth():
    from assist.channel.deepgram_agent import PLAY_MAX_BYTES

    session = DeepgramVoiceSession(tools=None, enable=False)
    for _ in range(4):
        session._queue_assistant_audio(b"\x00" * (PLAY_MAX_BYTES // 2))
    assert len(session._play_buf) <= PLAY_MAX_BYTES + (PLAY_MAX_BYTES // 2)


def test_mic_backlog_is_coalesced_into_one_send():
    q = asyncio.Queue()
    for i in range(5):
        q.put_nowait(bytes([i]))
    out = DeepgramVoiceSession._drain_mic_backlog(q, b"\xff")
    assert out == b"\xff\x00\x01\x02\x03\x04"
    assert q.empty()


def test_should_not_reconnect_on_bad_credentials():
    from deepgram.core.api_error import ApiError

    session = DeepgramVoiceSession(tools=None, enable=False)
    exc = ApiError(status_code=401, body="Websocket initialized with invalid credentials.")
    assert session._should_reconnect(exc, attempt=1) is False


def test_settings_uses_v1_listen_provider_by_default(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_LISTEN_VERSION", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = DeepgramVoiceSession(tools=_StubTools(), enable=False)

    provider = session._settings().agent.listen.provider
    assert provider.version == "v1"
    assert provider.model == "nova-3"
    assert session._force_end_turn_ok is False


def test_settings_switches_to_flux_when_opted_in(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_LISTEN_VERSION", "v2")
    monkeypatch.delenv("DEEPGRAM_STT_MODEL", raising=False)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    session = DeepgramVoiceSession(tools=_StubTools(), enable=False)

    provider = session._settings().agent.listen.provider
    assert provider.version == "v2"
    assert provider.model == "flux-general-en"
    assert session._force_end_turn_ok is True


class _StubTools:
    def declarations(self):
        return [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}]


class _SpokenTools:
    def spoken_from_tool(self, name, result):
        return result.get("spoken") or ""


def test_tool_reply_watchdog_fires_when_agent_stays_silent():
    from assist.channel.deepgram_agent import TOOL_REPLY_TIMEOUT_SEC

    session = DeepgramVoiceSession(tools=_SpokenTools(), enable=False)
    session._arm_reply_watchdog(
        [{"name": "measure_distances", "content": '{"spoken": "Wall, 0.9 meters."}'}]
    )

    assert session._fallback_phrase == "Wall, 0.9 meters."
    assert session._reply_watchdog_due(session._awaiting_reply_since) is False
    assert (
        session._reply_watchdog_due(
            session._awaiting_reply_since + TOOL_REPLY_TIMEOUT_SEC
        )
        is True
    )


def test_tool_reply_watchdog_stands_down_once_agent_speaks():
    from assist.channel.deepgram_agent import TOOL_REPLY_TIMEOUT_SEC

    session = DeepgramVoiceSession(tools=_SpokenTools(), enable=False)
    session._arm_reply_watchdog(
        [{"name": "measure_distances", "content": '{"spoken": "Wall, 0.9 meters."}'}]
    )
    armed_at = session._awaiting_reply_since

    asyncio.run(session._handle_message(None, b"assistant pcm"))

    assert session._awaiting_reply_since is None
    assert session._reply_watchdog_due(armed_at + TOOL_REPLY_TIMEOUT_SEC) is False


def test_tool_reply_watchdog_does_not_talk_over_a_new_user_turn():
    from assist.channel.deepgram_agent import TOOL_REPLY_TIMEOUT_SEC

    session = DeepgramVoiceSession(tools=_SpokenTools(), enable=False)
    session._arm_reply_watchdog(
        [{"name": "measure_distances", "content": '{"spoken": "Wall, 0.9 meters."}'}]
    )
    session.mic_gate.arm()

    assert (
        session._reply_watchdog_due(
            session._awaiting_reply_since + TOOL_REPLY_TIMEOUT_SEC
        )
        is False
    )
