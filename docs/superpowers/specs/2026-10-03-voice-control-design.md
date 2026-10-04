# Voice Control Architecture — nekit Blind-Assist MVP

**Date:** 2026-10-03  
**Scope:** Mac + iPhone Record3D LiDAR; Russian primary; cane users  
**Status:** Design only (no production code in this pass)

---

## 1. Problem diagnosis (from current code)

The voice path is: `VadUtteranceCapture` → batch STT → regex intents → vision LLM → blocking macOS `say`. Several structural bugs make it feel unreliable outdoors and unsafe for navigation.

### 1.1 Echo / self-hearing (critical)

| Issue | Evidence |
|-------|----------|
| Mic stays open while TTS plays | `VadUtteranceCapture._loop` always runs; on speech it calls `stop_speaking()` then buffers audio (`session.py:154–163`) |
| No acoustic echo cancellation | Local `sounddevice` InputStream only (`session.py:99–105`); no WebRTC AEC / Speex / mute-while-TTS |
| Post-TTS listen gate too short | `notify_reply_done` sets `_listen_after = now + 0.4s` (`session.py:288–289`) — room reverb + `say` tail often >400 ms |
| Session-start gate fixed, not tied to TTS end | `_listen_after = now + 2.2` then async `speak(...)` (`session.py:257–260`) — if greeting is longer/shorter, gate drifts |
| Dropped audio during gate is silent | `_on_utterance` returns without feedback if `time < _listen_after` (`session.py:298–299`) |

**Symptom:** user hears answer → assistant re-hears its own voice → wrong STT / false barge-in / “Не расслышала”.

### 1.2 Fake barge-in + state lies

| Issue | Evidence |
|-------|----------|
| Any VAD hit kills TTS immediately | `if is_speaking(): stop_speaking()` before min-speech duration (`session.py:154–156`) — street noise / cane taps interrupt answers |
| `SPEAKING` set before audio exists | `app.py:110–111` calls `mark_speaking()` **before** `llm.describe()`; UI/state say speaking while still thinking |
| Barge-in has no sustained-speech gate | Industry pattern: require ~250–300 ms confirmed speech before interrupt (RealtimeSTT / Pipecat mute strategies) |

### 1.3 Latency stack is batch-end → wait → speak

| Stage | Current behavior | Typical cost |
|-------|------------------|--------------|
| Endpointing | Fixed silence `VOICE_SILENCE_END_SEC=0.85` (`session.py:29`) | ≥850 ms after last word |
| STT | Full WAV POST to Deepgram REST `/v1/listen` (`audio.py:123–144`) or local Whisper `base` CPU (`audio.py:160–191`) | 300 ms–several s |
| Intent | Sync regex (`intents.py`) | ~0 |
| LLM | Vision OpenRouter, full JPEG (`app.py:112–117`) | 1–5 s |
| TTS | `speak(..., blocking=True)` waits for entire `say` (`app.py:120`) | full utterance length |

**No streaming STT, no EagerEndOfTurn, no TTS first-audio-while-LLM-continues.** User-perceived “end of speech → first reply audio” often 3–8 s.

### 1.4 VAD quality

| Issue | Evidence |
|-------|----------|
| Silero used as one-shot on last 512 samples | `session.py:127–132` — no rolling state / `get_speech_timestamps` for the utterance |
| Energy fallback threshold 0.015 RMS | `session.py:135` — outdoor / traffic / beeps false-positive heavily |
| Min utter 0.45 s / max 12 s | `session.py:30–31` — long FIND questions get truncated poorly; short noise still passes if energy high |

### 1.5 Intent router (safety / UX)

| Issue | Evidence |
|-------|----------|
| Almost everything → SCENE | Default free-form returns `Intent.SCENE` (`intents.py:74–79`) — misheard noise triggers expensive vision LLM |
| FIND needs locate cue or `?` | `intents.py:57–59` — STT often omits `?`; “дверь” alone becomes SCENE |
| No confirmation for FIND/DISTANCE | Wrong meters or wrong object is safety-critical for cane users |
| UNKNOWN feedback identical to empty STT | Both say «Не расслышала» (`session.py:309–319`) — user cannot tell empty vs unrecognized |

### 1.6 Session / activation semantics

| Issue | Evidence |
|-------|----------|
| Activation = OpenCV key `v` | `app.py:148–152` — not accessible without finding keyboard; cane user alone cannot start reliably |
| Always-on VAD after `v` | No wake word; outdoor always-listen amplifies false triggers |
| FOLLOWUP auto-ends at 20 s | `FOLLOWUP_SEC` + `tick()` (`session.py:28, 275–281`) — surprise session death mid-walk |
| Busy drops utterances | `_busy` early return (`session.py:296–297`) — no “занята” / queue of one |
| Latest-wins LLM silent | `app.py:96–100` replaces pending job with no voice cue |

### 1.7 TTS / orchestration

| Issue | Evidence |
|-------|----------|
| Blocking `say` | `audio.py:51–84`, `app.py:120` — cannot stream sentence 1 while LLM finishes sentence 2 |
| Interrupt via `killall say` | `audio.py:40–46` — blunt; races with new `speak` |
| Beeps continue during dialogue | By design (README) — good for safety, but raises energy-VAD false triggers |

---

## 2. Target product spec (UX contract)

### 2.1 Principles (blind cane users, RU)

1. **Predictable activation** — user always knows when the mic is armed.
2. **Short answers first** — orientation in ≤2 sentences; meters only when asked or confirmed.
3. **Never invent distance** — DISTANCE/FIND meters only from SENSOR_JSON.
4. **Interruptible but not twitchy** — barge-in requires sustained human speech, not noise.
5. **Honest failure** — distinct prompts for empty STT, low confidence, busy, offline.
6. **Hands-free preferred** — keyboard `v` is debug/fallback, not primary.

### 2.2 Activation model (recommendation)

**Primary (MVP):** **Push-to-talk (PTT) session** — hold/toggle a dedicated hardware or software trigger (Space / foot pedal / AirPods double-tap / Bluetooth button). Mic arms only while PTT active or for a short follow-up window after reply.

**Secondary (v2):** Always-on **wake word** («Некит» / «Ассистент») with high-precision detector + immediate earcon, then one command turn.

**Rejected for outdoor MVP:** true always-listen without wake word (street noise + beeps + TTS echo).

| Mode | When | Mic | Notes |
|------|------|-----|-------|
| IDLE | default | off | Passive beeps only |
| ARMED / LISTENING | after PTT or wake | on | Earcon «слушаю» |
| THINKING | after EOT | soft-mute TTS echo | Earcon optional tick |
| SPEAKING | TTS playing | **input muted** (+ cooldown) | Barge-in only if sustained speech after unmute policy |
| FOLLOWUP | after TTS | on for N seconds | Configurable 8–15 s; announce before auto-idle |
| IDLE | timeout / «стоп» | off | Earcon «выключено» |

### 2.3 Turn-taking / barge-in / AEC

**Hard rule for laptop speakers (no WebRTC transport):**

1. While TTS plays: **mute mic to STT path** (do not feed VAD→utterance).
2. After TTS ends: **cooldown 600–900 ms** before accepting utterances (covers `say` tail + room).
3. Optional **half-duplex barge-in**: if user holds PTT during SPEAKING → stop TTS immediately and listen.
4. If open-mic barge-in desired later: require **≥280 ms Silero speech** + energy above adaptive noise floor before `stop_speaking()`.

**Prefer headphones for development;** production cane kit should assume speaker + mic proximity → mute strategy is the reliable fix (Pipecat LocalAudioTransport has no AEC; browser WebRTC does).

### 2.4 Intent taxonomy + confirmation

| Intent | Examples (RU) | Action | Confirm? |
|--------|---------------|--------|----------|
| STOP | стоп, хватит, отмена | End session | No |
| SCENE | что вокруг, опиши | Vision narrative, no required meters | No |
| DISTANCE | как далеко…, сколько метров | Meters from JSON only | **Yes if target ambiguous** («человек или дверь?») |
| FIND | где дверь / найди машину | Side + meters if in JSON | **Yes if STT confidence low or object not in scene** |
| REPEAT | повтори | Re-speak last phrase | No |
| HELP | что умеешь | Short capability list | No |
| UNKNOWN | — | Clarify, do **not** call vision LLM | — |

**Confirmation UX:** one short question; timeout 4 s → cancel safely («отменено»). Never invent meters while confirming.

### 2.5 Latency budgets (p50 outdoors, Wi‑Fi)

| Milestone | Budget |
|-----------|--------|
| Speech end → final transcript | ≤ 450 ms (streaming STT / Flux EOT) |
| Transcript → intent | ≤ 50 ms |
| Intent → first LLM token (non-vision path) | ≤ 400 ms |
| Vision path first token | ≤ 1500 ms |
| **Speech end → first TTS audio** | **≤ 1.2 s** SCENE template/offline; **≤ 2.5 s** vision |
| Barge-in stop (PTT) | ≤ 100 ms to silence |

Instrument these with timestamps in logs.

### 2.6 Failure modes

| Failure | User hears | System behavior |
|---------|------------|-----------------|
| Empty / garbage STT | «Не расслышала, повторите» | Stay FOLLOWUP; no LLM |
| Low-confidence FIND | «Ищете дверь?» | Wait confirm |
| No frame / no LiDAR | «Нет кадра» | Stay FOLLOWUP |
| LLM busy | «Секунду» once | Latest-wins queue; coalesce |
| STT cloud down | «Перехожу на офлайн» | Fallback GigaAM/Whisper |
| Noise burst during ARMED | ignore if < min speech | Adaptive noise floor |

### 2.7 Target state machine

```
IDLE
  --(PTT/wake)--> ARMED
ARMED
  --(speech start)--> CAPTURING
  --(PTT release / timeout idle)--> IDLE
CAPTURING
  --(EOT / Flux EndOfTurn)--> THINKING
  --(PTT cancel)--> ARMED
THINKING
  --(routed intent)--> ACTING
  --(empty/unknown)--> SPEAKING(clarify) --> FOLLOWUP
ACTING
  --(LLM+TTS start)--> SPEAKING   # mark SPEAKING only when audio starts
SPEAKING
  --(TTS done + cooldown)--> FOLLOWUP
  --(PTT barge-in)--> CAPTURING
FOLLOWUP
  --(speech)--> CAPTURING
  --(timeout / стоп)--> IDLE
```

**Contract:** `SPEAKING` means audio is audible. Never set SPEAKING during LLM wait.

---

## 3. External research shortlist

### 3.1 Orchestration / patterns to steal

| Project | Steal | Skip for nekit MVP |
|---------|-------|--------------------|
| **Deepgram Flux** | Turn events `StartOfTurn` / `EagerEndOfTurn` / `EndOfTurn`; RU via `flux-general-multi` + `language_hint=ru`; ~260 ms p50 EOT | Full rewrite of vision pipeline into Flux-only |
| **RealtimeSTT** | Dual VAD (WebRTC + Silero), pre-roll, realtime vs final ASR, callbacks | Heavy dependency if we only need patterns |
| **Pipecat** | InterruptionFrame cancel chain; **mute-while-TTS + cooldown** for speaker echo | Full framework (overkill for single Mac loop) |
| **livekit-agents** | Production media + AEC via WebRTC | Needs browser/room; not waist-worn Mac loop |
| **WhisperLiveKit / whisper streaming** | Local streaming ASR patterns | RU quality / CPU RTF weaker than GigaAM |
| **OpenAI Realtime** | End-to-end S2S | Cost, RU STT latency reports mixed; couples vision poorly |
| **Rhasspy / Wyoming / Mycroft** | Offline wake-word + intent slots | Older stack; keep wake-word idea only |
| **ElevenLabs Conversational AI** | Streaming TTS quality | Extra vendor; RU + cost; not needed MVP |

### 3.2 STT comparison (Russian, Mac Python)

| Option | Streaming | RU quality | Latency | Offline | Fit |
|--------|-----------|------------|---------|---------|-----|
| **Deepgram Flux multi + `language_hint=ru`** | Yes (WS `/v2/listen`) | Good (Nova-3 class) | Sub-300 ms transcript; EOT ~100–500 ms | No (self-host optional) | **Primary cloud** |
| **Deepgram Nova-3 streaming** | Yes | Strong RU | Sub-300 ms | No | Alt if Flux multi unavailable |
| **Yandex SpeechKit STT v3** | Yes gRPC | Best-in-class RU domain | Real-time partials | No | **Strong RU primary alt** (account/region) |
| **OpenAI gpt-4o-transcribe / Realtime** | Yes | Decent multi | Community: often 1.5–2 s vs Deepgram | No | Fallback / experiment only |
| **AssemblyAI** | Yes | Weaker RU focus | Low for EN | No | Skip for RU MVP |
| **GigaAM v3 e2e-RNNT** | Batch/utterance | Excellent RU (far better than Whisper on Golos) | RTF ≪1 on Apple Silicon / CPU | **Yes** | **Primary offline fallback** |
| **faster-whisper large-v3-turbo** | Batch | OK RU | Slow on CPU (RTF~1); OK GPU | Yes | Secondary offline |
| **Current Whisper `base`** | Batch | Poor RU | Slow + inaccurate | Yes | **Replace** |

### 3.3 TTS comparison

| Option | Streaming | RU quality | Interrupt | Offline | Fit |
|--------|-----------|------------|-----------|---------|-----|
| **macOS `say` (Milena)** | No | Acceptable | killall | Yes | Keep as **offline fallback** |
| **Yandex SpeechKit TTS v3 stream** | Yes | Excellent RU | Easy (stop stream) | No | **Primary cloud TTS** if SpeechKit chosen |
| **OpenAI TTS** | Chunked | Good RU | Yes | No | Good cloud TTS if staying OpenAI/Deepgram |
| **ElevenLabs** | Yes | Variable RU | Yes | No | Optional polish v2 |
| **Piper** | Local | OK with RU voice | Yes | Yes | Offline alt to `say` |

### 3.4 VAD / AEC

- **Silero VAD** local, ~1–3 ms/frame — keep, but use proper streaming API + start/stop hysteresis (not single 512-slice).
- **WebRTC VAD** as fast pre-gate (RealtimeSTT pattern).
- **AEC:** on LocalAudio Mac path, **software AEC is weak**; use **mic mute during TTS + post-cooldown**. True AEC needs WebRTC client or hardware DSP.

---

## 4. Recommended stack for THIS project

### Primary (online)

| Layer | Choice | Why |
|-------|--------|-----|
| Orchestration | Keep AssistApp loop; **rewrite voice session** (do not adopt full Pipecat yet) | Minimal churn with Record3D/YOLO |
| VAD / turn | Silero streaming + **Deepgram Flux** EOT (`flux-general-multi`, `language_hint=ru`) | Fixes endpointing + barge-in semantics |
| STT | Deepgram Flux streaming (fallback Nova-3 WS) | Already have API key pattern; RU supported |
| Intent | Regex + **confidence / confirm** for FIND/DISTANCE; unknown ≠ scene | Safety |
| TTS | OpenAI TTS streaming **or** keep `say` until streaming wired; prefer short phrases | First-audio latency |
| Echo | **Mute mic→STT while SPEAKING + 700 ms cooldown** | Fixes self-echo on laptop speakers |
| Activation | **PTT toggle** (Space + optional Bluetooth) for MVP | Accessible without wake-word yet |

### Fallback (offline / degraded)

| Layer | Choice |
|-------|--------|
| STT | **GigaAM v3 e2e-RNNT** (Python or CoreML/MLX on Apple Silicon) |
| VAD | Silero only + longer silence |
| TTS | macOS `say` / Piper |
| LLM | Existing offline SENSOR_JSON templates |

### Explicit non-goals (MVP)

- Full OpenAI Realtime speech-to-speech (vision + meters control needs text intents).
- Always-on free listen outdoors.
- ElevenLabs as hard dependency.

**One-line stack:** Silero+mute-AEC → Deepgram Flux (`ru`) streaming STT → gated intents → OpenRouter vision → streaming TTS (OpenAI/`say` fallback); offline: GigaAM + `say`.

---

## 5. Phased implementation plan

### Phase 0 — Stop the bleeding (1 session, no new vendors)

1. Mute utterance acceptance while `is_speaking()`; only allow PTT barge-in OR sustained speech ≥280 ms.
2. Cooldown after TTS: **700–900 ms** (replace 0.4 s); tie session-start listen gate to TTS completion callback, not fixed 2.2 s.
3. Call `mark_speaking()` only when TTS audio starts, not before LLM.
4. Empty STT / UNKNOWN: do not default noise to SCENE; require clearer phrase.
5. Earcons: short `say` «слушаю» / «готово» vs long instructional greeting.

### Phase 1 — Streaming STT + PTT (MVP voice)

1. Replace REST Deepgram with **Flux WebSocket** (`/v2/listen`, `language_hint=ru`, keyterms: дверь, машина, метров…).
2. Use `EndOfTurn` (optionally `EagerEndOfTurn` to prefetch scene JSON).
3. PTT: Space hold/toggle; keyboard `v` becomes synonym.
4. Instrument latency logs (EOT → STT final → LLM → TTS start).
5. Confirm prompts for FIND/DISTANCE when confidence low.

### Phase 2 — Offline + better TTS

1. Integrate GigaAM as automatic fallback when Deepgram errors / no network.
2. Streaming TTS (OpenAI or SpeechKit); sentence-split LLM replies.
3. Optional wake word (openWakeWord / Porcupine RU) behind flag.
4. Adaptive noise floor for outdoor beeps.

### Phase 3 — Productization

1. Bluetooth PTT / AirPods controls.
2. Optional SpeechKit A/B for RU WER.
3. Metrics dashboard; user study with cane users.

---

## 6. Risks

| Risk | Mitigation |
|------|------------|
| Flux multi RU quality outdoors | Keyterm prompting; A/B vs SpeechKit; keep GigaAM offline |
| Mute-while-TTS blocks true barge-in | PTT override always works |
| Vision LLM dominates latency | Prefetch scene JSON; short prompts; cache last frame |
| Wake word false accepts | Keep disabled outdoors until tuned |
| Dependency sprawl (Pipecat/LiveKit) | Steal patterns; don’t import whole stack until needed |
| Privacy / cloud STT | Offline path mandatory for sensitive contexts |

---

## 7. Success criteria

- Outdoor walk: ≤1 false auto-command per 5 min in FOLLOWUP with beeps on.
- p50 speech-end → first TTS audio ≤2.5 s with cloud STT + vision.
- Zero self-echo loops in 20 consecutive replies on MacBook speakers.
- Blind tester can start/stop session without sighted help (PTT or wake).
- FIND wrong-object rate drops after confirmation rule (qualitative lab).

---

## 8. References

- Deepgram Flux quickstart / language prompting (`flux-general-multi`, `language_hint=ru`)
- Deepgram measuring streaming latency (sub-300 ms; EOT 100–500 ms)
- Pipecat interruptions + mute strategies; LocalAudioTransport AEC limitation (GitHub #188)
- RealtimeSTT: WebRTC+Silero VAD, pre-roll, dual ASR
- GigaAM evaluation (salute-developers) vs Whisper on Russian sets
- Yandex SpeechKit STT/TTS v3 streaming docs
- Current code: `assist/voice/session.py`, `audio.py`, `intents.py`, `assist/app.py`
