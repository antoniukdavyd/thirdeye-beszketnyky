"""phash gate: skip re-calling the vision LLM on a near-identical frame+question."""

from __future__ import annotations

import numpy as np

from assist.llm.openrouter import OpenRouterClient


class _CountingClient(OpenRouterClient):
    """Force the online path and count real describe calls."""

    def __init__(self):
        super().__init__(api_key="test-key")
        self.calls = 0

    def _describe_online(self, rgb_bgr, scene, user_question, mode):
        self.calls += 1
        return f"answer {self.calls}"


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_cache_hit_on_identical_frame_and_question():
    c = _CountingClient()
    f = _frame(1)
    a = c.describe(f, {}, user_question="what is this?")
    b = c.describe(f.copy(), {}, user_question="what is this?")
    assert a == b
    assert c.calls == 1  # second call served from cache


def test_cache_miss_on_different_question():
    c = _CountingClient()
    f = _frame(1)
    c.describe(f, {}, user_question="what is this?")
    c.describe(f, {}, user_question="read the sign")
    assert c.calls == 2


def test_cache_miss_on_different_frame():
    c = _CountingClient()
    c.describe(_frame(1), {}, user_question="what is this?")
    c.describe(_frame(999), {}, user_question="what is this?")
    assert c.calls == 2
