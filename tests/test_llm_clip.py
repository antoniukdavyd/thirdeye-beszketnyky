"""LLM reply clipping for short TTS."""

from assist.llm.openrouter import _clip_spoken


def test_clip_keeps_short_multi_sentence_answer():
    # The answerer should keep the full 2–3 sentence actionable reply, not
    # truncate to the first sentence.
    t = _clip_spoken("In front of you a hall. People on the left. A table on the right.")
    assert t == "In front of you a hall. People on the left. A table on the right."


def test_clip_trims_to_whole_sentences_under_cap():
    text = "First sentence here. Second sentence here. Third one is extra padding."
    out = _clip_spoken(text, max_chars=40)
    assert out == "First sentence here."
    assert len(out) <= 40


def test_clip_long_no_period():
    long = "word " * 50
    out = _clip_spoken(long, max_chars=40)
    assert len(out) <= 41
