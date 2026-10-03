# Third Eye Slice A — Voice Agent + Hazards/Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Gemini Live with a usable Deepgram Voice Agent + OpenRouter (tools) path so a blind walker gets continuous hazard beeps plus short spoken orientation answers (what's ahead, where/how far) via PTT.

**Architecture:** Keep Record3D → YOLO/zones → SceneStore → BeepAlert unchanged. Swap `assist/channel` from Gemini Live to Deepgram Voice Agent WebSocket: mic (PTT-gated) → Deepgram STT/TTS + turn-taking; `think` LLM = OpenRouter BYO; client-side `FunctionCallRequest` → existing `ToolRegistry`. One conversational agent; perception stays a background service.

**Tech Stack:** Python 3.10+, `deepgram-sdk` (Agent v1), OpenRouter Chat Completions (agent + vision), `sounddevice`, existing OpenCV loop / Record3D / Ultralytics.

**Spec:** `docs/superpowers/specs/2026-10-04-voice-agent-openrouter-deepgram-design.md`  
**Product frame:** Third Eye outdoor walk MVP (no OCR / bus ETA / indoor maps in this plan).

## Global Constraints

- English primary for prompts, TTS, spoken phrases.
- Meters only from tool JSON / SENSOR_JSON — never invent distances.
- Beeps never go through the LLM and must keep running during dialogue.
- No Gemini / `GOOGLE_API_KEY` required for voice after this plan.
- PTT: mic audio is sent only while a listen turn is armed (Space); OpenCV cannot reliably do key-up hold — use press-to-arm until EOT / timeout / second Space.
- Off-topic → soft redirect, no tools.
- YAGNI: no Pipecat/LiveKit, no multi-agent, no OCR/transit in this plan.
- Prefer small focused modules under `assist/channel/`; do not grow `assist/app.py` with WebSocket logic.

## File structure (target)

| Path | Responsibility |
|------|----------------|
| `assist/agent/__init__.py` | Third Eye system prompt; Deepgram/OpenRouter env helpers; drop Gemini Live config |
| `assist/tools/__init__.py` | Keep tools; add `openai_tool_declarations()` / Deepgram function dicts; keep `gemini_tool_declarations` only if tests need or delete |
| `assist/channel/__init__.py` | Public exports: `create_voice_channel`, `RealtimeState`, session type alias |
| `assist/channel/deepgram_agent.py` | Deepgram Voice Agent session: connect, PTT mic gate, playback, function calls |
| `assist/app.py` | Wire new channel; Space = PTT arm; env messaging |
| `assist/voice/audio.py` | Unchanged local TTS for keyboard keys |
| `.env.example`, `requirements.txt`, `README.md` | Keys and docs |
| `tests/test_deepgram_tools.py` | Declaration + function-call dispatch unit tests |
| `tests/test_agent_prompt.py` | Prompt / off-topic policy smoke |

---

### Task 1: Env, deps, and agent prompt (no network voice yet)

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `assist/agent/__init__.py`
- Create: `tests/test_agent_prompt.py`

**Interfaces:**
- Produces: `SYSTEM_INSTRUCTION` (Third Eye outdoor rules), `deepgram_api_key() -> Optional[str]`, `openrouter_api_key() -> Optional[str]`, `voice_ready() -> bool` (Deepgram + OpenRouter), `openrouter_agent_model() -> str`, `deepgram_tts_model() -> str`
- Consumes: none

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent_prompt.py -v`  
Expected: FAIL (import/assert mismatch vs old Gemini `voice_ready`)

- [ ] **Step 3: Update `requirements.txt`**

Replace `google-genai>=1.0` with:

```text
deepgram-sdk>=5.0
```

Keep `sounddevice`, `httpx`, etc. Do not remove other deps used by perception.

- [ ] **Step 4: Rewrite `.env.example`**

```bash
# Copy to .env and fill in

# Deepgram Voice Agent (STT + TTS) — required for Space voice
DEEPGRAM_API_KEY=
# Aura / Flux speak model id
DEEPGRAM_TTS_MODEL=aura-2-thalia-en
# Optional listen model (Nova/Flux)
DEEPGRAM_STT_MODEL=nova-3

# OpenRouter — agent think + describe_scene vision
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_AGENT_MODEL=openai/gpt-4o-mini
OPENROUTER_MODEL=openai/gpt-4o-mini
LLM_JPEG_MAX_SIDE=512
LLM_JPEG_QUALITY=60
LLM_MAX_TOKENS=55

# Local TTS for keyboard shortcuts (d/m/p) and boot prompt (English)
TTS_PROVIDER=edge
TTS_VOICE=en-US-JennyNeural
```

- [ ] **Step 5: Rewrite `assist/agent/__init__.py` helpers + prompt**

Replace Gemini helpers with:

```python
SYSTEM_INSTRUCTION = """You are Third Eye, a voice orientation assistant for a blind cane user (English).

You are an extra pair of eyes on a neck-worn camera with LiDAR depth. Continuous hazard beeps are handled outside you — do not try to beep.

Rules:
- Answer SHORTLY: 1–2 sentences max. Spatial language only: left / center / right, object, meters.
- Take meters and sides ONLY from tool results (SENSOR_JSON). Never invent distances or objects.
- “what's around / what's ahead / scene / hazards” → describe_scene and/or sense_snapshot; mention near obstacles and cars if present in tool data.
- “how far / distance” → measure_distances (or sense_snapshot). If target ambiguous, ask one clarifying question — no meters until clear.
- “where is the door / person / car / bus” → find_object; if not found, say so and suggest turning — do not guess.
- Off-topic chitchat (not about surroundings) → brief soft redirect to orientation help; do NOT call tools.
- Do not narrate tool calls beyond a short “Looking” if needed.
"""

def deepgram_api_key() -> Optional[str]:
    return (os.getenv("DEEPGRAM_API_KEY") or "").strip() or None

def openrouter_api_key() -> Optional[str]:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip() or None

def openrouter_agent_model() -> str:
    return os.getenv("OPENROUTER_AGENT_MODEL") or os.getenv("OPENROUTER_MODEL") or "openai/gpt-4o-mini"

def deepgram_tts_model() -> str:
    return os.getenv("DEEPGRAM_TTS_MODEL") or "aura-2-thalia-en"

def deepgram_stt_model() -> str:
    return os.getenv("DEEPGRAM_STT_MODEL") or "nova-3"

def voice_ready() -> bool:
    return bool(deepgram_api_key() and openrouter_api_key())
```

Remove `live_model_name`, `google_api_key`, `build_live_config` (or leave thin deprecated stubs that raise `RuntimeError("Gemini Live removed")` only if something still imports — prefer delete and fix imports).

Update `__all__` accordingly.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_agent_prompt.py -v`  
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example assist/agent/__init__.py tests/test_agent_prompt.py
git commit -m "feat(agent): Third Eye prompt and Deepgram/OpenRouter env helpers"
```

---

### Task 2: Deepgram / OpenAI-style tool declarations from ToolRegistry

**Files:**
- Modify: `assist/tools/__init__.py`
- Create: `tests/test_deepgram_tools.py`
- Modify: `tests/test_tools.py` only if imports break

**Interfaces:**
- Consumes: `ToolRegistry.declarations()` existing OpenAI-ish dicts
- Produces: `deepgram_think_functions(registry: ToolRegistry) -> list[dict]` — each item `{name, description, parameters, client_side: True}` with **no** `endpoint` key; `parse_tool_arguments(raw: str | dict) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deepgram_tools.py
from assist.perception.scene_store import SceneStore
from assist.tools import ToolRegistry, deepgram_think_functions, parse_tool_arguments


def test_deepgram_functions_are_client_side():
    reg = ToolRegistry(SceneStore())
    fns = deepgram_think_functions(reg)
    names = {f["name"] for f in fns}
    assert names >= {"sense_snapshot", "measure_distances", "find_object", "describe_scene"}
    for f in fns:
        assert f.get("client_side") is True
        assert "endpoint" not in f
        assert "parameters" in f and f["parameters"].get("type") == "object"


def test_parse_tool_arguments_json_string():
    assert parse_tool_arguments('{"label":"door"}') == {"label": "door"}
    assert parse_tool_arguments({}) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deepgram_tools.py -v`  
Expected: FAIL (`deepgram_think_functions` not defined)

- [ ] **Step 3: Implement helpers in `assist/tools/__init__.py`**

```python
def parse_tool_arguments(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return json.loads(raw)
    return {}


def deepgram_think_functions(registry: ToolRegistry) -> list[dict]:
    out = []
    for d in registry.declarations():
        out.append(
            {
                "name": d["name"],
                "description": d["description"],
                "parameters": d.get("parameters")
                or {"type": "object", "properties": {}},
                "client_side": True,
            }
        )
    return out
```

Tighten tool descriptions slightly for hazards (optional one-liner on `sense_snapshot`: “Use for clearances and nearby hazards like cars.”).

If `gemini_tool_declarations` is unused after channel swap, delete it in Task 5; for now keep if still imported.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_deepgram_tools.py tests/test_tools.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assist/tools/__init__.py tests/test_deepgram_tools.py
git commit -m "feat(tools): Deepgram client-side function declarations"
```

---

### Task 3: Deepgram Voice Agent session skeleton (connect + PTT mic gate + playback)

**Files:**
- Create: `assist/channel/deepgram_agent.py`
- Modify: `assist/channel/__init__.py`
- Create: `tests/test_voice_ptt_gate.py`

**Interfaces:**
- Produces: class `DeepgramVoiceSession` with:
  - `available: bool`, `active: bool`, `state: RealtimeState`
  - `start_session() -> None` — connect WS thread if not connected
  - `end_session(announce: bool = True) -> None`
  - `arm_listen() -> None` — start forwarding mic PCM to agent (PTT)
  - `disarm_listen() -> None` — stop forwarding mic
  - `request_ptt_barge_in() -> None` — stop TTS playback buffer + re-arm listen
  - `tick() -> None` — no-op or drain
- Consumes: `deepgram_api_key`, `openrouter_*`, `SYSTEM_INSTRUCTION`, `ToolRegistry`, `deepgram_think_functions`, `sounddevice`

**PTT semantics (OpenCV-safe):**
- `start_session()` opens the agent connection (idle, mic not forwarding).
- Space when idle/connected → `arm_listen()` until Deepgram `UserStoppedSpeaking` / agent finishes user turn, or 12s timeout, or Space again → `disarm_listen()`.
- While assistant speaks, Space → `request_ptt_barge_in()`.

- [ ] **Step 1: Write failing unit test for mic gate (no real API)**

```python
# tests/test_voice_ptt_gate.py
from assist.channel.deepgram_agent import MicGate


def test_mic_gate_defaults_closed():
    g = MicGate()
    assert g.armed is False
    g.arm()
    assert g.armed is True
    g.disarm()
    assert g.armed is False
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_voice_ptt_gate.py::test_mic_gate_defaults_closed -v`

- [ ] **Step 3: Implement `MicGate` + session skeleton in `assist/channel/deepgram_agent.py`**

Implement at minimum:

```python
class MicGate:
    def __init__(self) -> None:
        self._armed = False
        self._lock = threading.Lock()

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def arm(self) -> None:
        with self._lock:
            self._armed = True

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
```

`DeepgramVoiceSession.__init__(tools, on_state=None, enable=True)` mirrors old Gemini session fields needed by `app.py`.

`available`: `voice_ready()` and imports of `deepgram` + `sounddevice` succeed.

Thread + asyncio loop pattern can follow the old Gemini session: background thread runs `asyncio.run(self._main())`.

In `_main()`:
1. `DeepgramClient(api_key=...)`
2. `async with client.agent.v1.connect() as agent:` (adjust to installed SDK API — if sync context manager, wrap with `asyncio.to_thread` or use documented async API; pin to SDK docs at implement time)
3. Send Settings with:
   - audio input: linear16, 16000 (or SDK default; match sounddevice InputStream)
   - listen: Deepgram STT model from env
   - think: `provider.type=open_ai`, `model=openrouter_agent_model()`, `endpoint.url=https://openrouter.ai/api/v1/chat/completions`, headers `Authorization: Bearer {OPENROUTER_API_KEY}`, `HTTP-Referer` optional, `prompt=SYSTEM_INSTRUCTION`, `functions=deepgram_think_functions(tools)`
   - speak: Deepgram TTS model from env
4. Mic callback: if `MicGate.armed`, send PCM; else send silence **or** skip send (prefer skip/silence per Deepgram guidance so connection stays alive — use silence frames if required)
5. On binary/audio messages: enqueue for OutputStream playback; set state SPEAKING on first audio; clear on drain
6. On `UserStoppedSpeaking` (or equivalent): `MicGate.disarm()`, set THINKING
7. On `FunctionCallRequest`: handled in Task 4 (stub: respond with `{"ok":false,"error":"tools not wired"}` so agent does not hang)

Keep `RealtimeState` enum in this module or shared in `channel/__init__.py` (same values as before: idle/connecting/listening/speaking/thinking).

- [ ] **Step 4: Export from `assist/channel/__init__.py`**

```python
from .deepgram_agent import DeepgramVoiceSession, RealtimeState, create_voice_channel

__all__ = ["DeepgramVoiceSession", "RealtimeState", "create_voice_channel"]
```

`create_voice_channel(tools, on_state, enable=True)` returns `DeepgramVoiceSession` or `None` if not `available` (same signature as old factory).

Remove Gemini imports.

- [ ] **Step 5: Run unit tests**

Run: `pytest tests/test_voice_ptt_gate.py -v`  
Expected: PASS  
(Do not require live API keys in CI unit tests.)

- [ ] **Step 6: Commit**

```bash
git add assist/channel/deepgram_agent.py assist/channel/__init__.py tests/test_voice_ptt_gate.py
git commit -m "feat(channel): Deepgram Voice Agent session with PTT mic gate"
```

---

### Task 4: Wire FunctionCallRequest → ToolRegistry

**Files:**
- Modify: `assist/channel/deepgram_agent.py`
- Modify: `tests/test_deepgram_tools.py`

**Interfaces:**
- Consumes: `ToolRegistry.execute(name, args) -> dict`, `parse_tool_arguments`
- Produces: handler that sends FunctionCallResponse with `content=json.dumps(result)`

- [ ] **Step 1: Add unit test for dispatch helper**

```python
# tests/test_deepgram_tools.py (add)
import json
from assist.channel.deepgram_agent import dispatch_function_calls


class _FakeReg:
    def execute(self, name, args):
        return {"ok": True, "name": name, "args": args}


def test_dispatch_function_calls_builds_responses():
    calls = [
        type("C", (), {"id": "1", "name": "find_object", "arguments": '{"label":"door"}', "client_side": True})()
    ]
    out = dispatch_function_calls(_FakeReg(), calls)
    assert len(out) == 1
    assert out[0]["id"] == "1"
    assert out[0]["name"] == "find_object"
    body = json.loads(out[0]["content"])
    assert body["ok"] is True
    assert body["args"]["label"] == "door"
```

(If using SDK types, `dispatch_function_calls` may accept duck-typed objects with `.id/.name/.arguments/.client_side`.)

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `dispatch_function_calls` and hook into message loop**

```python
def dispatch_function_calls(tools: ToolRegistry, calls: list) -> list[dict]:
    responses = []
    for call in calls:
        if getattr(call, "client_side", True) is False:
            continue
        args = parse_tool_arguments(getattr(call, "arguments", {}) or {})
        result = tools.execute(call.name, args)
        responses.append(
            {
                "id": call.id,
                "name": call.name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    return responses
```

In the agent message handler, for each response call the SDK `send_function_call_response` (field names per installed SDK). Run tool execution in a worker thread if the receive loop is sync and tools can block (`describe_scene`).

Log `tool_call` name + ms via existing `debuglog`.

- [ ] **Step 4: pytest**

Run: `pytest tests/test_deepgram_tools.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add assist/channel/deepgram_agent.py tests/test_deepgram_tools.py
git commit -m "feat(channel): dispatch Deepgram function calls to ToolRegistry"
```

---

### Task 5: Integrate into AssistApp (replace Gemini UX)

**Files:**
- Modify: `assist/app.py`
- Modify: `assist/voice/__init__.py` (docstring)
- Modify: `assist/voice/audio.py` (docstring only)
- Delete Gemini-only code paths leftover in `assist/channel/` / `assist/agent/`

**Interfaces:**
- Consumes: `create_voice_channel`, `DeepgramVoiceSession.arm_listen`, `disarm_listen`, `start_session`, `request_ptt_barge_in`
- Produces: Space behavior per PTT semantics

- [ ] **Step 1: Change `_toggle_voice` into PTT-aware handler**

Replace logic roughly with:

```python
def _toggle_voice(self) -> None:
    if self.voice is None:
        speak("Voice unavailable. Need Deepgram and OpenRouter keys.")
        return
    if not self.voice.available:
        speak("Voice unavailable.")
        return
    # Ensure connected
    if not self.voice.active:
        self.voice.start_session()
        # first Space also arms listen
        self.voice.arm_listen()
        return
    st = self.voice.state.value
    if st == "speaking":
        self.voice.request_ptt_barge_in()
        return
    if st == "listening":
        self.voice.disarm_listen()
        return
    self.voice.arm_listen()
```

Update boot prints / HUD help line:

```text
Space=PTT voice  d=scene  m=dist  p/c/f=find  q=quit
```

Replace messages mentioning Google / Gemini with Deepgram + OpenRouter.

- [ ] **Step 2: Fix imports**

`from .agent import Intent, voice_ready` (drop `live_model_name`).  
`from .channel import create_voice_channel` (session type optional).

- [ ] **Step 3: Run existing unit tests**

Run: `pytest tests/ -v`  
Expected: all PASS (no live voice required)

- [ ] **Step 4: Manual smoke (developer machine with keys + Record3D)**

1. `cp .env.example .env` and set `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`
2. `pip install -r requirements.txt`
3. `python run_assist.py`
4. Confirm beeps still work without speaking
5. Space → speak “What's around?” → short spatial answer
6. Space → “How far is the person?” → meters only if in view
7. Space → “I love Coca-Cola” → soft redirect, no vision spam in logs
8. Confirm laptop speakers: no self-echo loop (mic disarmed while speaking)

- [ ] **Step 5: Commit**

```bash
git add assist/app.py assist/voice/__init__.py assist/voice/audio.py assist/channel/ assist/agent/
git commit -m "feat(app): wire Third Eye PTT voice via Deepgram+OpenRouter"
```

---

### Task 6: Docs — README reflects Third Eye outdoor slice

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Modes / Setup / Controls**

Document:
- Product: Third Eye outdoor MVP (cane + neck camera / Record3D)
- Voice: Deepgram + OpenRouter agent + tools
- Passive: Space arms listen (PTT); beeps always on
- Keys unchanged for debug
- Env vars from `.env.example`
- Explicit non-goals this release: OCR, bus ETA, indoor maps

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: Third Eye outdoor voice MVP setup"
```

---

### Task 7: Spec coverage self-check + optional hazard prompt polish

**Files:**
- Modify: `assist/agent/__init__.py` only if smoke showed weak hazard language
- Modify: `docs/superpowers/specs/2026-10-04-voice-agent-openrouter-deepgram-design.md` status line → `Implementation in progress / done`

- [ ] **Step 1: Checklist against spec**

| Spec item | Task |
|-----------|------|
| Deepgram STT+TTS + OpenRouter agent | 3–5 |
| Single agent + tools | 2, 4 |
| PTT mic gate + mute while speaking | 3, 5 |
| Meters from tools only | 1, 4 |
| Off-topic soft redirect | 1 (prompt) |
| Beeps independent | unchanged perception |
| No Gemini | 1, 5 |
| English | 1, 6 |

- [ ] **Step 2: If describe_scene ignores cars/near obstacles, tighten `SYSTEM_INSTRUCTION` one paragraph and commit**

```bash
git commit -am "fix(agent): stronger hazard/orientation prompt wording"
```

---

## Manual acceptance (Slice A done when)

1. Walk simulation indoors: L/C/R beeps still fire with obstacles; car alert still fires on YOLO car/bus/truck near.
2. PTT question “what's ahead?” → ≤2 sentence spatial answer using tools.
3. Distance question without inventing meters when LiDAR missing.
4. Off-topic does not call `describe_scene` (check `[nekit ... tool ...]` logs).
5. No `GOOGLE_API_KEY` needed; voice works with Deepgram + OpenRouter only.

---

## Out of scope (later Third Eye plans)

- OCR / read signs aloud  
- Bus stop name + lines + ETA (city APIs)  
- Indoor navigation  
- True hardware PTT / neck wearable without Mac  
- Pipecat / LiveKit migration  
