from assist.channel.deepgram_agent import MicGate


def test_mic_gate_defaults_closed():
    gate = MicGate()
    assert gate.armed is False
    gate.arm()
    assert gate.armed is True
    gate.disarm()
    assert gate.armed is False
