"""Main assist loop: waist-up beeps + Deepgram voice agent + on-demand tools."""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

from .agent import Intent, voice_ready
from .capture import Record3DCapture
from .channel import DeepgramVoiceSession, create_voice_channel
from .debuglog import log
from .llm import OpenRouterClient
from .perception import (
    BeepAlert,
    DetectedObject,
    DetectionHold,
    ObjectDetector,
    SceneStore,
    ZoneFilter,
    build_scene,
    compute_zones,
)
from .tools import ToolRegistry
from .voice import speak

WINDOW = "nekit assist"
DETECT_EVERY_N = 4
CENTER_ALERT_M = 0.8


def hud_scale(frame_h: int) -> float:
    return max(1.0, float(frame_h) / 480.0)


def hud_metrics(frame_h: int) -> dict:
    s = hud_scale(frame_h)
    bar_h = max(72, int(88 * s))
    return {
        "scale": s,
        "zone_font": 0.9 * s,
        "obj_font": 0.75 * s,
        "help_font": 0.65 * s,
        "status_font": 0.7 * s,
        "bar_h": bar_h,
        "thickness": max(2, int(2 * s)),
    }


def _draw_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float,
    color: tuple,
    thickness: int,
    pad: bool = True,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    if pad:
        cv2.rectangle(
            img,
            (x - 4, y - th - 6),
            (x + tw + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)


class AssistApp:
    def __init__(self, enable_yolo: bool = True, enable_voice: bool = True):
        load_dotenv()
        self.capture = Record3DCapture()
        self.beep = BeepAlert()
        self.zone_filter = ZoneFilter()
        self.detector = ObjectDetector() if enable_yolo else None
        self.detection_hold = DetectionHold(ttl_frames=3)
        self.llm = OpenRouterClient()
        self.store = SceneStore()
        self.tools = ToolRegistry(self.store, llm=self.llm)

        self.objects: List[DetectedObject] = []
        self.scene: dict = {}
        self.last_rgb: Optional[np.ndarray] = None
        self.last_phrase = ""
        self.frame_i = 0
        self._status = "idle"

        self.voice: Optional[DeepgramVoiceSession] = create_voice_channel(
            self.tools,
            on_state=self._on_voice_state,
            enable=enable_voice,
        )

    def connect(self, dev_idx: int = 0) -> None:
        self.capture.connect(dev_idx=dev_idx)

    def _on_voice_state(self, state: str) -> None:
        self._status = state

    def _run_tool_speak(self, name: str, args: dict) -> None:
        """Keyboard shortcuts: run tool and speak via local TTS."""

        def _work():
            try:
                result = self.tools.execute(name, args)
                phrase = self.tools.spoken_from_tool(name, result)
                self.last_phrase = phrase
                if self.voice is not None:
                    self.voice.set_last_phrase(phrase)
                print(f"[tool/{name}] {phrase}")
                log("tool", f"speak {name}", text=phrase[:120])
                speak(phrase, blocking=True)
            finally:
                self._status = (
                    "idle"
                    if self.voice is None or not self.voice.active
                    else self.voice.state.value
                )

        threading.Thread(target=_work, daemon=True).start()

    def _request_describe(
        self,
        question: str,
        mode: Intent = Intent.SCENE,
    ) -> None:
        if mode == Intent.DISTANCE:
            self._run_tool_speak("measure_distances", {})
            return
        if mode == Intent.FIND:
            q = (question or "").lower()
            label = "door"
            if "люд" in q or "человек" in q or "person" in q or "people" in q:
                label = "person"
            elif "машин" in q or "car" in q:
                label = "car"
            elif "стол" in q or "table" in q:
                label = "table"
            self._run_tool_speak("find_object", {"label": label})
            return

        snap = self.store.snapshot(copy_rgb=True)
        if not snap.has_frame or snap.rgb_bgr is None:
            speak("No frame", blocking=True)
            return

        self._run_tool_speak("describe_scene", {"focus": question})

    def _toggle_voice(self) -> None:
        if self.voice is None:
            log("app", "voice unavailable — need Deepgram + OpenRouter")
            speak("Voice unavailable. Need Deepgram and OpenRouter keys.")
            return
        log(
            "app",
            "key Space/v",
            voice_state=self.voice.state.value,
            active=int(self.voice.active),
        )
        if not self.voice.available:
            speak("Voice unavailable.")
            return
        if not self.voice.active:
            self.voice.start_session()
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

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            log("app", "quit key")
            return False

        if key in (ord("v"), ord("V"), ord(" ")):
            self._toggle_voice()
            return True

        if key in (ord("d"), ord("D")):
            log("app", "key d → describe_scene")
            self._request_describe(
                "Describe the scene in front of the user.",
                mode=Intent.SCENE,
            )
            return True

        if key in (ord("m"), ord("M")):
            log("app", "key m → measure_distances")
            self._request_describe(
                "How far are the nearest important objects?",
                mode=Intent.DISTANCE,
            )
            return True

        if key in (ord("f"), ord("F")):
            log("app", "key f → find door")
            self._request_describe("Where is the door?", mode=Intent.FIND)
            return True

        if key in (ord("p"), ord("P")):
            log("app", "key p → find person")
            self._request_describe("Where are the people?", mode=Intent.FIND)
            return True

        if key in (ord("c"), ord("C")):
            log("app", "key c → find car")
            self._request_describe("Where is the car?", mode=Intent.FIND)
            return True

        return True

    def _draw(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        zones,
        objects: List[DetectedObject],
    ) -> np.ndarray:
        d_vis = np.clip(depth, 0, 5.0) / 5.0
        d_color = cv2.applyColorMap((d_vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis = cv2.addWeighted(rgb, 0.72, d_color, 0.28, 0)
        h, w = vis.shape[:2]
        m = hud_metrics(h)
        y0, y1 = int(h * 0.12), int(h * 0.55)
        cv2.rectangle(vis, (0, y0), (w - 1, y1), (255, 200, 0), 1)
        w3 = w // 3
        for x in (w3, 2 * w3):
            cv2.line(vis, (x, y0), (x, y1), (255, 255, 255), 1)

        def zone_color(v):
            if v is None:
                return (120, 120, 120)
            if v < CENTER_ALERT_M:
                return (0, 0, 255)
            if v < 1.5:
                return (0, 200, 255)
            return (0, 220, 0)

        labels = [
            ("L", zones.left, w3 // 2),
            ("C", zones.center, w // 2),
            ("R", zones.right, 2 * w3 + w3 // 2),
        ]
        for name, val, x in labels:
            txt = f"{name}:{val:.2f}m" if val is not None else f"{name}:—"
            _draw_label(
                vis,
                txt,
                (max(8, x - 50), int(36 * m["scale"])),
                m["zone_font"],
                zone_color(val),
                m["thickness"],
            )

        for o in objects:
            x1, y1b, x2, y2 = o.box
            color = (
                (0, 165, 255)
                if any(k in o.label.lower() for k in ("car", "bus", "truck"))
                else (0, 255, 180)
            )
            cv2.rectangle(vis, (x1, y1b), (x2, y2), color, 2)
            _draw_label(
                vis,
                f"{o.label} {o.dist_m:.1f}m",
                (x1, max(int(28 * m["scale"]), y1b - 8)),
                m["obj_font"],
                color,
                max(1, m["thickness"] - 1),
            )

        bar_h = m["bar_h"]
        cv2.rectangle(vis, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        help_line = "Space=PTT voice  d=scene  m=dist  p/c/f=find  q=quit"
        cv2.putText(
            vis,
            help_line,
            (12, h - bar_h + int(bar_h * 0.38)),
            cv2.FONT_HERSHEY_SIMPLEX,
            m["help_font"],
            (220, 220, 220),
            max(1, m["thickness"] - 1),
            cv2.LINE_AA,
        )
        status = f"{self._status}"
        if self.last_phrase:
            max_chars = max(40, int(70 * m["scale"]))
            status += " | " + self.last_phrase[:max_chars]
        cv2.putText(
            vis,
            status,
            (12, h - int(bar_h * 0.18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            m["status_font"],
            (0, 255, 200),
            max(1, m["thickness"] - 1),
            cv2.LINE_AA,
        )
        return vis

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        log("app", "assist starting")
        if self.llm.available:
            print(f"OpenRouter ready ({self.llm.model})")
            log("llm", "OpenRouter ready", model=self.llm.model)
        else:
            print("No OPENROUTER_API_KEY — offline templates for describe_scene.")
            log("llm", "no API key — offline templates")

        if self.voice is not None:
            print("Voice: Deepgram + OpenRouter PTT agent (Space).")
            log("realtime", "Deepgram + OpenRouter enabled")
        else:
            if voice_ready():
                print("Voice off — install Deepgram and sounddevice dependencies.")
            else:
                print("Voice off — set DEEPGRAM_API_KEY and OPENROUTER_API_KEY.")
            print("  Keyboard: d=scene  m=distance  p/c/f=find still work.")
            log("realtime", "disabled")
        print("  Logs: lines starting with [nekit ...] — copy those for debug.")

        speak("Assistant ready")

        try:
            while True:
                if self.voice is not None:
                    self.voice.tick()
                if not self.capture.wait_frame(timeout=0.3):
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break
                    continue

                frame = self.capture.grab()
                self.capture.clear_event()
                if frame is None:
                    continue

                self.frame_i += 1
                raw_zones = compute_zones(frame.depth, frame.conf)
                zones = self.zone_filter.update(raw_zones)

                if self.detector and self.frame_i % DETECT_EVERY_N == 0:
                    fresh = self.detector.detect(
                        frame.rgb_bgr, frame.depth, frame.conf
                    )
                    self.objects = self.detection_hold.update(fresh)
                elif not self.detector:
                    self.objects = []

                self.beep.update(zones, self.objects)
                self.scene = build_scene(
                    zones, self.objects, device=frame.device_name
                )
                self.last_rgb = frame.rgb_bgr
                self.store.publish(frame.rgb_bgr, self.scene, self.objects)

                vis = self._draw(frame.rgb_bgr, frame.depth, zones, self.objects)
                cv2.imshow(WINDOW, vis)
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self._handle_key(key):
                    break
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if self.voice is not None and self.voice.active:
                self.voice.end_session(announce=False)
            cv2.destroyAllWindows()
