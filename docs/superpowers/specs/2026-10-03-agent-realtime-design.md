# Agent + Realtime Channel + On-Demand Camera Tools

**Date:** 2026-10-03  
**Scope:** Mac + iPhone Record3D LiDAR; English primary; cane users  
**Status:** Implementation — realtime-only (no legacy Flux path)

---

## Architecture

```
User mic/ear
    ↕
Gemini Live channel (speech-to-speech)
    ↕
Agent (system prompt + tool policy)
    ↕ tools on demand
SceneStore ← Record3D loop (YOLO + zones) always on
BeepAlert  ← same loop, never via agent
```

Keyboard `d`/`m`/`p`/`c`/`f` call the same tools and speak via local edge-tts (not Live).

## Tools

| Name | Returns |
|------|---------|
| `sense_snapshot` | compact SENSOR_JSON + age_ms |
| `measure_distances` | meters/sides from JSON only |
| `find_object` | found + bearing + dist_m |
| `describe_scene` | short English sentence (OpenRouter vision) |

Meters only from tool JSON. Beeps never through the agent.

## Env

```bash
GOOGLE_API_KEY=...
GEMINI_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-09-2025
OPENROUTER_API_KEY=...
TTS_VOICE=en-US-JennyNeural
```

Missing `GOOGLE_API_KEY` → voice off; keys still work.
