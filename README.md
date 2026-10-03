# Third Eye — outdoor walk MVP (nekit / magicstick)

**Third Eye** is a blind-assist outdoor slice: cane user + **neck/waist camera** on **iPhone** via [Record3D](https://record3d.app/) USB streaming, processed on **Mac**.

**Primary language: English** (agent prompts, spoken phrases, local TTS).

## Modes

| Mode | What happens |
|------|----------------|
| **Passive (always on)** | Waist-up L/C/R depth beeps (above cane). Extra alert for nearby cars. Runs continuously; independent of voice. |
| **Voice (`Space` / `v`)** | **Push-to-talk (PTT):** press Space to arm the mic, speak, then Space again (or end-of-turn) to disarm. **Deepgram** STT/TTS + **OpenRouter** agent with camera/LiDAR **tools**. |
| **`d`** | Scene via `describe_scene` (OpenRouter vision). |
| **`m`** | Distances from LiDAR JSON (`measure_distances`). |
| **`p` / `c` / `f`** | Find person / car / door (`find_object`). |

Architecture: **user ↔ dialog agent** (OpenRouter); sensors are tools. **Beeps never go through the LLM** and stay on during voice.

## Out of scope (this release)

- OCR (signs, menus, labels)
- Bus / transit ETA
- Indoor maps or turn-by-turn indoors

## Setup

1. iPhone: Record3D → USB Streaming → Record.
2. Mac:

```bash
cd nekit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill keys in .env (see table below)
```

Environment variables (see [`.env.example`](.env.example)):

| Key | Purpose |
|-----|---------|
| `DEEPGRAM_API_KEY` | Voice agent STT + TTS (required for Space/v) |
| `DEEPGRAM_TTS_MODEL` | Deepgram speak model (default Aura) |
| `DEEPGRAM_STT_MODEL` | Deepgram listen model (default Nova) |
| `OPENROUTER_API_KEY` | Agent “think” + `describe_scene` vision |
| `OPENROUTER_AGENT_MODEL` | Chat model for voice agent |
| `OPENROUTER_MODEL` | Vision model for scene describe |
| `TTS_PROVIDER` | Local TTS for `d`/`m`/boot (`edge` or `say`) |
| `TTS_VOICE` | edge-tts voice (default `en-US-JennyNeural`) |

3. Run:

```bash
python run_assist.py
python run_assist.py --no-yolo    # beeps only
python run_assist.py --no-voice   # keys only
```

Without `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY`, voice is off; keyboard tools and passive beeps still work. Without `OPENROUTER_API_KEY`, `describe_scene` uses offline templates from SENSOR_JSON.

## Layout

```
assist/
  app.py           # loop + HUD + SceneStore publish
  agent/           # Third Eye prompt, env helpers
  channel/         # Deepgram Voice Agent session (PTT)
  tools/           # sense / measure / find / describe
  capture/         # Record3D
  perception/      # zones, YOLO, SceneStore
  voice/           # local TTS only (keyboard / boot)
  llm/             # OpenRouter vision + agent
```

## Controls

- `Space` / `v` — **PTT:** arm listen (mic to Deepgram) / disarm / barge-in while agent speaks
- `d` / `m` / `p` / `c` / `f` — scene / distance / find (debug shortcuts; unchanged)
- `q` / `Esc` — quit
- Speak naturally in English after arming PTT; agent calls tools for meters and scene
- Logs: `[nekit ...]` including tool calls and realtime state
