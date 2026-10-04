from types import SimpleNamespace

from assist.app import AssistApp


class FakeVoice:
    def __init__(self, *, active: bool, state: str, available: bool = True):
        self.active = active
        self.available = available
        self.state = SimpleNamespace(value=state)
        self.mic_gate = SimpleNamespace(armed=(state == "listening"))
        self.calls = []
        self.holds = []

    def start_session(self):
        self.calls.append("start_session")
        self.active = True

    def arm_listen(self, hold: bool = False):
        self.calls.append("arm_listen")
        self.holds.append(hold)
        self.state = SimpleNamespace(value="listening")
        self.mic_gate.armed = True

    def disarm_listen(self):
        self.calls.append("disarm_listen")
        self.state = SimpleNamespace(value="idle")
        self.mic_gate.armed = False

    def request_ptt_barge_in(self, hold: bool = False):
        self.calls.append("request_ptt_barge_in")
        self.holds.append(hold)
        self.state = SimpleNamespace(value="listening")
        self.mic_gate.armed = True


def app_with_voice(voice, *, hold_mode: bool = False):
    app = AssistApp.__new__(AssistApp)
    app.voice = voice
    app._space_held = False
    app._space_hold_mode = hold_mode
    return app


def test_first_v_starts_session_and_arms_listening():
    voice = FakeVoice(active=False, state="idle")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["start_session", "arm_listen"]


def test_v_while_speaking_requests_barge_in():
    voice = FakeVoice(active=True, state="speaking")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["request_ptt_barge_in"]


def test_v_while_listening_disarms_listening():
    voice = FakeVoice(active=True, state="listening")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["disarm_listen"]


def test_v_while_idle_arms_listening():
    voice = FakeVoice(active=True, state="idle")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["arm_listen"]


def test_hold_press_arms_and_release_disarms(monkeypatch):
    voice = FakeVoice(active=True, state="idle")
    app = app_with_voice(voice, hold_mode=True)
    seq = iter([True, True, False])
    monkeypatch.setattr("assist.app.is_space_down", lambda: next(seq))

    app._poll_space_hold()  # press
    assert voice.calls == ["arm_listen"]
    assert app._space_held is True

    app._poll_space_hold()  # still held
    assert voice.calls == ["arm_listen"]

    app._poll_space_hold()  # release
    assert voice.calls == ["arm_listen", "disarm_listen"]
    assert app._space_held is False


def test_waitkey_space_ignored_in_hold_mode():
    voice = FakeVoice(active=True, state="idle")
    app = app_with_voice(voice, hold_mode=True)

    assert app._handle_key(ord(" ")) is True
    assert voice.calls == []


def test_hold_press_marks_turn_as_held(monkeypatch):
    # The session needs to know Space is physically down so Deepgram's own
    # endpointing cannot close the mic mid-sentence.
    voice = FakeVoice(active=True, state="idle")
    app = app_with_voice(voice, hold_mode=True)
    monkeypatch.setattr("assist.app.is_space_down", lambda: True)

    app._poll_space_hold()

    assert voice.calls == ["arm_listen"]
    assert voice.holds == [True]


def test_toggle_press_is_not_held():
    voice = FakeVoice(active=True, state="idle")

    app_with_voice(voice)._toggle_voice()

    assert voice.holds == [False]


def test_release_disarms_even_when_agent_reports_thinking():
    # Deepgram can emit AgentThinking while Space is still down. Gating the
    # release on state == "listening" left the mic armed with nothing to close
    # it, so the next press was swallowed.
    voice = FakeVoice(active=True, state="thinking")
    voice.mic_gate.armed = True
    app = app_with_voice(voice, hold_mode=True)
    app._space_held = True

    app._ptt_release()

    assert voice.calls == ["disarm_listen"]


def test_release_is_noop_when_nothing_armed():
    voice = FakeVoice(active=True, state="idle")
    voice.mic_gate.armed = False

    app_with_voice(voice, hold_mode=True)._ptt_release()

    assert voice.calls == []
