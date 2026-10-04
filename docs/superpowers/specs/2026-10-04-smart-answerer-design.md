# Third Eye — Smart Answerer (B+ hybrid) Design

**Date:** 2026-10-04
**Scope:** Mac + iPhone Record3D (RGB + LiDAR), English, cane users. Voice "answerer" quality + speed.
**Status:** Approved design → implementation.
**Supersedes concerns in:** `2026-10-03-voice-control-design.md` (voice turn-taking already shipped via Deepgram Voice Agent + PTT).

---

## 1. Problem (confirmed with user)

The voice assistant gives **shallow descriptions, slowly, and feels dumb**. Activation (PTT) is fine. Goal: a working demo where the assistant answers "what's ahead / what is this / read this sign / which bus" **accurately and fast**.

Root causes found in code:

1. **Blind double-hop.** Deepgram think model (`gpt-4o-mini`, text) is blind; to answer visual questions it calls `describe_scene`, which fires a *second* OpenRouter vision call (`tools/__init__.py:289`). Two LLM hops ⇒ 2–5 s.
2. **Double token clamp.** `LLM_MAX_TOKENS=55` (`.env.example`) + prompt "1–2 sentences max" (`agent/__init__.py:16`) + clip in `_prompt_and_tokens` (`openrouter.py:126`). ~55 tokens ≈ one sentence ⇒ "light description."
3. **`describe_scene` ignores the real question.** It hardcodes `"Describe the scene in front of the user."` (`tools/__init__.py:288`); the user's actual words never reach vision.
4. **No OCR.** "which bus / what does the sign say / route number" cannot work at all.
5. **Vision model is `gpt-4o-mini`** at 512px/q60 — weak scene vision + OCR.

## 2. Chosen architecture — B+ hybrid

Keep the Deepgram Voice Agent (PTT, STT, TTS, barge-in) untouched. Think model (`gpt-4o-mini`) stays a **router**. Fix the **tools** and the **vision call**.

Research (LiveKit/Pipecat) confirms the production default is a **cascade** (STT → vision-LLM → TTS), not speech-to-speech — so B+ is architecturally sound, not a shortcut. A fully "sighted think" (single multimodal hop) is the ideal but requires replacing the Deepgram-agent loop; out of scope for the 10-hour demo, noted as future work.

### Tool map

| User intent | Tool | Engine | LLM? |
|---|---|---|---|
| meters / how far / is it clear ahead | `sense_snapshot`, `measure_distances` | LiDAR JSON + depth zones | no |
| what's ahead / what is this / describe | `describe_scene` (rewritten) | **one GPT-4o vision call**, sees frame + LiDAR + **real question** | yes |
| read this / which bus / what's on the sign | `read_text` (**new**) | **ocrmac** (Apple Vision, on-device ~150 ms) → optional GPT-4o reasoning over the exact strings | OCR no-LLM; LLM only to reason |
| find door/person/car | `find_object` | YOLO + depth | no |

### Key fixes

- **F1 — real question to vision.** `describe_scene` gains `question` (keep `focus` as alias). Router passes the user's words; the returned text is the final answer, not a stub for the think model to paraphrase.
- **F2 — OCR tool `read_text`.** ocrmac returns `[(text, conf, bbox), …]`. Return exact strings + side (left/center/right from bbox x). VLMs hallucinate route numbers; OCR must own exact text. Optional second step: pass strings to GPT-4o only for reasoning ("which bus goes downtown").
- **F3 — unclamp tokens + reword prompt.** `LLM_MAX_TOKENS` → 150; vision cap in `_prompt_and_tokens` raised to match; system prompt → "Answer to the point, 2–3 short sentences; lead with the actionable part."
- **F4 — vision model = gpt-4o.** Via `OPENROUTER_MODEL`; image to ≤768px, q~80.
- **F5 — speed:** (a) **filler** "Looking…" — done via the system prompt (the agent speaks it over its own Deepgram TTS stream). A local edge-tts/afplay filler was rejected: it would play over the same output device and fight Deepgram audio. (b) **phash gate** — if the frame is ~unchanged since the last describe AND the question matches, reuse the cached answer instead of re-calling GPT-4o. Guarded on `imagehash`; absent → cache disabled, no crash.
- **F6 — depth zones upgrade (if time):** obstacle-aware L/C/R on top of `depth_zones.py` (ground-plane / height filter so the floor directly ahead doesn't read as "blocked"). Lower priority than F1–F4; the answerer wins come from F1–F5.

## 3. Components & interfaces

- `assist/perception/ocr.py` (**new**) — `recognize_text(rgb_bgr) -> list[TextLine]` with `TextLine(text, conf, bearing, bbox)`. Wraps `ocrmac`; degrades to `{ok: False, "error": "ocr unavailable"}` if import/platform fails (keeps tests + non-Mac CI green).
- `assist/tools/__init__.py` — add `read_text` declaration + `_read_text` handler + `spoken_from_tool` branch; extend `describe_scene` params with `question`.
- `assist/llm/openrouter.py` — plumb `user_question` already supported; raise token caps; default side ≤768px. Add tiny phash cache keyed on frame hash (optional helper, no new hard dep if `imagehash` absent → cache disabled).
- `assist/channel/deepgram_agent.py` — on `tool_call` for `describe_scene`/`read_text`, emit filler phrase once (reuse existing TTS/`speak`), guarded so it doesn't fight Deepgram audio.
- `assist/agent/__init__.py` — reword `SYSTEM_INSTRUCTION`; add routing hint for `read_text`.
- `.env` / `.env.example` — reconcile (dead `STT_PROVIDER`/`DEEPGRAM_MODEL`/`WHISPER_MODEL`), set `OPENROUTER_MODEL=openai/gpt-4o`, `LLM_MAX_TOKENS=150`.

## 4. Dependencies

- `ocrmac` (MIT) — Apple Vision OCR, on-device, no key. Add to requirements (Mac-only; import guarded).
- `imagehash` (BSD-2) — optional, for phash gate. Guarded; absence disables cache only.
- `gpt-4o` via existing OpenRouter key — no new infra.
- **No** Ultralytics change (note: AGPL — fine for demo, flag for productization).

## 5. Error / degrade behavior

| Condition | Behavior |
|---|---|
| ocrmac/import unavailable | `read_text` returns `{ok:False}`; router says "I can't read text right now." |
| No/stale frame (`age>1500ms`) | existing "No current camera frame" |
| GPT-4o error/timeout | existing offline SENSOR_JSON fallback in `openrouter.py` |
| phash lib missing | cache disabled, always call vision (no crash) |
| OCR finds nothing | "I don't see any readable text." |

## 6. Testing (TDD)

New/updated unit tests, no network (mock LLM, fake OCR):
- `test_read_text_tool.py` — `read_text` returns exact strings + bearing from a fake OCR; `spoken_from_tool` phrasing; graceful `{ok:False}` when OCR unavailable.
- `test_describe_question.py` — `describe_scene(question=…)` forwards the user question into `llm.describe(user_question=…)` (assert via a fake LLM), not the hardcoded string.
- `test_deepgram_tools.py` (extend) — `read_text` present in `deepgram_think_functions`.
- Token caps: assert `_prompt_and_tokens` honors raised `LLM_MAX_TOKENS`.
- Keep all existing tests green.

## 7. Demo success criteria

- "What's ahead?" → actionable 2–3 sentence answer with side + rough distance, p50 ≤ ~2.5 s perceived (filler covers the gap).
- "Read that / which bus?" → exact route number/text spoken, not hallucinated.
- "Find the door" → side + meters.
- No regression in PTT, passive beeps, or existing keyboard shortcuts.

## 8. Out of scope (demo)

- Sighted-think single multimodal hop (replace Deepgram-agent loop) — future.
- Full RGB-D free-space (RANSAC ground plane) beyond the simple height filter — stretch.
- Streaming clause-level TTS (Deepgram Voice Agent already chunks internally).
- Non-English, wake word, Bluetooth PTT.

## 9. Phasing (implementation order)

1. **F3+F4** config/prompt/token unclamp + gpt-4o (fastest quality win, ~30 min).
2. **F1** `describe_scene` real question (tests first).
3. **F2** `read_text` + ocrmac (tests first).
4. **F5a** filler audio; **F5b** phash gate.
5. **F6** depth-zone obstacle filter (if time).
