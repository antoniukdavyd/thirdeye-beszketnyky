# Third Eye — outdoor walk MVP (nekit / magicstick)

**Third Eye** is a blind-assist outdoor slice: cane user + **neck/waist camera** on **iPhone** via [Record3D](https://record3d.app/) USB streaming, processed on **Mac**.

**Primary language: English** (agent prompts, spoken phrases, local TTS).

## Modes

| Mode | What happens |
|------|----------------|
| **Passive (always on)** | Waist-up L/C/R depth beeps (above cane). Extra alert for nearby cars. Runs continuously; independent of voice. |
| **Voice (`Space` / `v`)** | **Hold Space to talk** (release = end of turn). `v` = toggle fallback. **Deepgram** STT/TTS + **OpenRouter** agent with camera/LiDAR **tools**. Click the OpenCV window first. |
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

**Prerequisites:** Python **3.11 or 3.12** (not 3.13 — `record3d` has no 3.13 build) and `cmake` (`record3d` builds a C++ extension).

```bash
cd nekit
brew install cmake            # required to build record3d
python3.11 -m venv .venv      # must be 3.11/3.12, not 3.13
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# fill keys in .env (see table below)
```

Always run inside the activated venv (`source .venv/bin/activate`) — plain
`python3 run_assist.py` uses the system Python and will miss dependencies.

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
| `R3D_BACKEND` | `native` (default, newest-frame reader, no lag) or `record3d` (vendor lib) |
| `FRAME_LOG_SEC` | Camera/render loop stats interval in seconds (`0` = off) |
| `YOLO_DEVICE` | `auto` (Apple GPU), `mps` or `cpu` |
| `DETECT_INTERVAL_SEC` | Seconds between YOLO labeling passes (default `0.5` = every 30th frame at 60 fps; beeps still every frame) |
| `NEKIT_LOG_FILE` | Session log in `logs/` (`0` = off) |

3. Run:

```bash
python run_assist.py
python run_assist.py --no-yolo    # beeps only
python run_assist.py --no-voice   # keys only
```

Without `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY`, voice is off; keyboard tools and passive beeps still work. Without `OPENROUTER_API_KEY`, `describe_scene` uses offline templates from SENSOR_JSON.

## When it crashes or freezes

Every run writes `logs/nekit-<date>-<time>.log` (path printed at startup): all terminal
output plus

- `crash | UNCAUGHT ...`: an exception that escaped, with thread name and traceback
- `Fatal Python error: ...`: a native crash (segfault/abort) with every thread's stack
- `watchdog | main STALLED`: render loop (or `detect`) stuck >1.5 s, with all stacks at that moment
- `app | frame error`: a bad frame that was skipped (the app keeps running)
- `frame | Record3D stream lost`: the phone stopped sending; reconnects automatically
- `health | ...`: every 10 s, RAM, CPU, threads, thermal state
- `app | main loop exit reason=...`: why the app stopped

To dump all stacks from a live app without stopping it: `kill -USR1 <pid>`.
Send the whole log file when reporting a problem.

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

- **Hold `Space`** — talk while held; release to send. `v` — toggle listen. Barge-in: hold Space while agent speaks
- `d` / `m` / `p` / `c` / `f` — scene / distance / find (debug shortcuts; unchanged)
- `q` / `Esc` — quit
- Speak naturally in English after arming PTT; agent calls tools for meters and scene
- Logs: `[nekit ...]` including tool calls and realtime state
