"""Blind-assist MVP: Mac Python + iPhone Record3D LiDAR.

Layers:
  capture/     — Record3D RGB-D stream
  perception/  — depth zones, YOLO, scene JSON
  voice/       — STT/TTS, session, intents
  llm/         — OpenRouter vision
  app.py       — main loop orchestrator
"""

__version__ = "0.1.0"
