from types import SimpleNamespace

from assist.app import AssistApp


class FakeVoice:
    def __init__(self, *, active: bool, state: str, available: bool = True):
        self.active = active
        self.available = available
        self.state = SimpleNamespace(value=state)
        self.calls = []

    def start_session(self):
        self.calls.append("start_session")
        self.active = True

    def arm_listen(self):
        self.calls.append("arm_listen")

    def disarm_listen(self):
        self.calls.append("disarm_listen")

    def request_ptt_barge_in(self):
        self.calls.append("request_ptt_barge_in")


def app_with_voice(voice):
    app = AssistApp.__new__(AssistApp)
    app.voice = voice
    return app


def test_first_space_starts_session_and_arms_listening():
    voice = FakeVoice(active=False, state="idle")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["start_session", "arm_listen"]


def test_space_while_speaking_requests_barge_in():
    voice = FakeVoice(active=True, state="speaking")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["request_ptt_barge_in"]


def test_space_while_listening_disarms_listening():
    voice = FakeVoice(active=True, state="listening")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["disarm_listen"]


def test_space_while_idle_arms_listening():
    voice = FakeVoice(active=True, state="idle")

    app_with_voice(voice)._toggle_voice()

    assert voice.calls == ["arm_listen"]
