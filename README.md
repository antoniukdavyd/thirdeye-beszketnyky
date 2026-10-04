# Third Eye

Third Eye is an assistive vision prototype for blind and visually impaired people. It combines a wearable iPhone camera and depth sensing with audio alerts and a voice assistant to help users understand their surroundings.

## Business value

The project explores how familiar consumer hardware can provide practical orientation support alongside a cane: nearby obstacle awareness, object locations, distance estimates, scene descriptions, and reading visible text. Continuous audio cues provide immediate feedback, while spoken questions let users request context when they need it.

The current MVP runs on an iPhone connected to a Mac, with English voice interaction.

## Architecture and orchestration

The application coordinates two parallel paths:

- **Continuous perception:** camera and depth frames feed local distance analysis and background object detection. Proximity beeps run independently of the voice agent.
- **On-demand assistance:** a voice agent interprets questions and calls tools to inspect the latest scene, measure distances, locate objects, read text, or generate a visual description.

```text
iPhone camera + depth → USB capture → Local perception → Proximity alerts
                                            ↓
                                      Shared scene state
                                            ↕
User speech → Voice agent → Tool execution → Spoken response
```

A shared `SceneStore` connects perception to the agent's tools. Capture uses the newest available frame, and detection runs in a worker thread to keep the display and depth alerts responsive. Distances come from sensor data; OCR runs locally, while conversational reasoning and visual descriptions use cloud models.

The code is organized under `assist/` into capture, perception, agent tools, voice interaction, and model integration. `run_assist.py` starts the application.

## Tech stack

- **Python 3.11/3.12** for application logic, threading, and asynchronous orchestration.
- **OpenCV and NumPy** for image processing and depth analysis.
- **PyTorch and Ultralytics YOLO-World** for local object detection, with Apple GPU acceleration when available.
- **Record3D** for iPhone camera and depth streaming; **Apple Vision** for local OCR.
- **Deepgram and OpenRouter** for voice interaction and language/vision models.

## Run locally

You need a Mac, a depth-capable iPhone (LiDAR recommended for the outdoor setup), Record3D with USB streaming, a USB cable, and a microphone and audio output. Use Python 3.11 or 3.12 with the supplied dependencies.

From the project directory:

```bash
brew install python@3.11 cmake
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `DEEPGRAM_API_KEY` and `OPENROUTER_API_KEY` to enable the voice assistant. Other model, audio, and performance settings are documented in [`.env.example`](.env.example). For a run without API keys, leave both key values empty: passive alerts and local sensor tools remain available, and scene descriptions use an offline fallback.

Connect the iPhone over USB, open Record3D, enable **USB Streaming**, and start recording. Then run inside the activated virtual environment:

```bash
python run_assist.py
```

Optional flags: `--no-voice` disables the voice session; `--no-yolo` disables object detection.

Click the camera window to use the controls:

| Control | Action |
| --- | --- |
| Hold `Space` | Talk; release to finish the turn |
| `v` | Toggle listening |
| `d` / `m` | Describe the scene / report distances |
| `p` / `c` / `f` | Find a person / car / door |
| `q` / `Esc` | Quit |

Session logs are saved in `logs/`; the active log path is printed at startup.
