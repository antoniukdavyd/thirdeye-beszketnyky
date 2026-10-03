# Voice Agent + Camera/LiDAR Tools (OpenRouter + Deepgram)

**Date:** 2026-10-04  
**Scope:** Mac + iPhone Record3D LiDAR; English primary; cane users; spatial orientation  
**Status:** Implementation done (Slice A on `feat/third-eye-voice-mvp`) — manual outdoor walk acceptance pending; post-TTS ~700 ms cooldown not implemented  
**Supersedes (voice path):** Gemini Live channel in `2026-10-03-agent-realtime-design.md`  
**Related:** passive beeps / SceneStore concepts from prior agent-realtime + voice-control specs

---

## 1. Product intent

The system is not a general chat app and not a visual UI product. The interface **is** voice, sound, and camera/depth:

- A blind cane user orients in space by **talking** to one agent and **hearing** continuous spatial audio (beeps) plus spoken answers.
- The agent is an independent conversational unit that understands intent and uses camera + LiDAR **tools** well.
- Success = the user learns “what’s near me, where is it, how far” safely and honestly.

---

## 2. Hard decisions

| Decision | Choice |
|----------|--------|
| Voice stack | **Deepgram** (STT + TTS Aura) + **OpenRouter** (agent + vision) |
| Leave behind | Gemini Live / Google voice channel |
| Architecture | **Single agent + tools** (not multi-agent) |
| Interaction | Voice-primary; beeps are continuous safety audio, not a menu |
| Activation | **Push-to-talk** (Space / button); mic only while PTT. Short post-reply open-mic window is optional later, not MVP. |
| Language | **English** primary (prompts, speech, TTS) |
| Off-topic | Soft redirect to surroundings; **no tools** |
| Multi-agent | Rejected as overhead for this product |

---

## 3. Ideal behavior (user-facing)

### 3.1 Always-on sound channel

- Waist-up L/C/R depth beeps (obstacles above cane height) run continuously from the perception loop.
- Extra alert for nearby cars.
- Beeps **never** go through the LLM and **do not stop** during dialogue.

### 3.2 Conversation channel

1. User holds PTT → speaks → releases.
2. Deepgram STT produces text.
3. OpenRouter agent interprets intent; calls tools only when needed.
4. Deepgram TTS speaks a **short** spatial answer (1–2 sentences).
5. While TTS plays: mic path muted to STT (no self-hearing). After TTS: ~700 ms cooldown. PTT during speech = barge-in (stop TTS, listen).

### 3.3 Answer style

- Spatial language: **left / center / right**, object, **meters only from sensor JSON**.
- Prefer orientation over courtesy filler.
- Never invent distance or objects not in view.
- If ambiguous target for distance/find: one clarifying question, then meters only when clear.

### 3.4 Off-topic example

User: “I love Coca-Cola.”  
Agent: brief ack + redirect (“Got it. I can tell you what’s around or help find something — what do you need?”). No camera/LiDAR tools.

---

## 4. Architecture

```
User mic (PTT) / ear
    ↕
Deepgram STT  →  OpenRouter agent (system prompt + tool policy)  →  Deepgram TTS
                      ↕ tools on demand
              SceneStore ← Record3D loop (YOLO + zones) always on
              BeepAlert  ← same loop, never via agent
```

**One conversational agent.** Perception (YOLO, zones, beeps, SceneStore) is a background **service**, not a second agent.

Keyboard `d` / `m` / `p` / `c` / `f` remain debug/fallback: same tools, local TTS if needed — not the primary interface.

---

## 5. Tools (camera + LiDAR portfolio)

| Tool | Purpose | Returns |
|------|---------|---------|
| `sense_snapshot` | Compact current scene facts | SENSOR_JSON + age_ms |
| `measure_distances` | Distances for known targets | meters / sides from JSON only |
| `find_object` | Locate person / car / door / etc. | found + bearing/side + dist_m if available |
| `describe_scene` | Short visual narrative | English sentence (OpenRouter vision) |

### Tool policy

| User need | Agent action |
|-----------|--------------|
| What’s around / orient me | `describe_scene` and/or `sense_snapshot`; short L/C/R summary |
| Where is X | `find_object` |
| How far is X | measure/find JSON only |
| Follow-up (“and left?”) | Fresh tools on new frame — do not invent from memory |
| Off-topic / chitchat | No tools; soft redirect |
| No frame / no depth | Honest failure phrases; no fabrication |

**Invariant:** meters only from tool JSON. Narrative may use vision, but numeric distance must not be guessed.

---

## 6. Voice session mechanics

| Phase | Behavior |
|-------|----------|
| IDLE | Mic off for STT; beeps only |
| PTT down | Mic → Deepgram STT |
| PTT up / end of utterance | Agent (+ tools) |
| THINKING | Do not mark as speaking until TTS audio starts |
| SPEAKING | Mic muted to STT; beeps continue |
| After TTS | Cooldown ~0.7 s; return to IDLE (MVP). Optional follow-up listen window = later. |
| Barge-in | PTT while speaking → stop TTS, capture new utterance |

Latency targets (aspirational p50): speech-end → first TTS audio ≤ ~2.5 s when vision tools run; faster when JSON-only.

---

## 7. Failure modes (spoken English)

| Failure | User hears | System |
|---------|------------|--------|
| Empty / garbage STT | “Sorry, I didn’t catch that.” | No tools; wait for next PTT |
| Off-topic | Soft redirect | No tools |
| No camera frame | “No camera frame right now.” | No invented scene |
| No LiDAR / no meters | “I can see it, but I don’t have distance.” | Side ok without meters |
| Object not in view | “No door in view — try turning.” | No guess |
| Ambiguous target | One clarifying question | Meters only after clarity |
| STT / network down | “Speech is down, try again.” | Honest fail (offline STT later) |
| Agent / OpenRouter down | “I can’t think right now.” | Beeps still run |
| Vision unavailable | SENSOR_JSON brief / “vision unavailable” | No fantasy |

**Prefer “I don’t know / I don’t see” over confident wrong guidance.**

---

## 8. Explicit non-goals (this design)

- Gemini Live / Google speech-to-speech channel.
- Multi-agent orchestration for conversation vs perception.
- Always-on free listening outdoors (no PTT).
- Rich GUI / mode-heavy product UX as the primary interface.
- Inventing meters or objects without sensors.
- Full chitchat companion (off-topic = soft redirect only).

---

## 9. Env (target)

```bash
DEEPGRAM_API_KEY=...
OPENROUTER_API_KEY=...
# optional model overrides
OPENROUTER_AGENT_MODEL=...
OPENROUTER_VISION_MODEL=...
DEEPGRAM_TTS_MODEL=...   # Aura voice id
```

Missing Deepgram key → voice path off; keyboard tools may still work. Missing OpenRouter → no agent/vision; beeps still work.

---

## 10. Success criteria

- Blind tester can orient with voice + beeps without sighted help for PTT start/stop.
- Zero self-echo loops on laptop speakers across a short session (mute-while-TTS).
- No invented distances in answers (spot-check logs: meters only after tool JSON).
- Off-topic utterances never trigger camera/vision tools.
- Outdoor walk: beeps remain useful while occasional PTT Q&A stays short and spatial.

---

## 11. Implementation note

Implemented on branch `feat/third-eye-voice-mvp` (Tasks 1–7): Deepgram Voice Agent + OpenRouter think, PTT mic gate, ToolRegistry dispatch, README/env. See plan `docs/superpowers/plans/2026-10-04-third-eye-voice-hazards-mvp.md`.
