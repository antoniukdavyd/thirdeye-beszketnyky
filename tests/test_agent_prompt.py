# tests/test_agent_prompt.py
from assist.agent import SYSTEM_INSTRUCTION, voice_ready


def test_system_instruction_forbids_invented_meters():
    text = SYSTEM_INSTRUCTION.lower()
    assert "never invent" in text or "only from tool" in text
    assert "off-topic" in text or "not about" in text
    assert "1–2" in SYSTEM_INSTRUCTION or "1-2" in SYSTEM_INSTRUCTION or "short" in text


def test_voice_ready_false_without_keys(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert voice_ready() is False
