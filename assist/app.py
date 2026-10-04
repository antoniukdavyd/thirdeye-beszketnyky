"""Main assist loop: waist-up beeps + Deepgram voice agent + on-demand tools."""

from __future__ import annotations

import os
import threading
import time
import traceback
import zlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

from .agent import Intent, voice_ready
from .capture import Record3DCapture
from .channel import DeepgramVoiceSession, create_voice_channel
from .crashlog import Watchdog, log_path
from .debuglog import log, log_exc
from .framelog import FrameLogger
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
from .voice.ptt_keys import is_space_down, space_hold_available

WINDOW = "nekit assist"
CENTER_ALERT_M = 0.8
# Space is sampled on its own thread. Sampling it once per render pass meant the
# PTT edge detector ran at whatever the render loop managed — a beep storm or a
# YOLO pass could stretch that past 100 ms and swallow a short tap entirely.
PTT_POLL_SEC = 0.015
# YOLO labeling cadence, measured pass start to pass start. 0.5 s = every 30th
# frame of the phone's 60 fps stream. Zones and beeps still run on every frame;
# this only paces labeling. Override with DETECT_INTERVAL_SEC.
DEFAULT_DETECT_INTERVAL_SEC = 0.5
# Floor so a typo cannot make the worker peg a core.
MIN_DETECT_INTERVAL_SEC = 0.05
# How long the last labels survive passes that find nothing.
DETECT_HOLD_SEC = 0.5
# The HUD is drawn and shown at most this tall. Record3D sends 1440x1920; drawing
# and imshow-ing that full size cost ~35 ms a frame, halving the window's fps.
# Perception and tools still get the full-resolution frame.
DISPLAY_MAX_H = 960
# One bad frame (odd size, truncated message, NaN-heavy depth) must not end a
# live demo: per-frame errors are logged and the frame is skipped. Only this
# many failures in a row — i.e. something persistently broken — stop the app.
MAX_FRAME_ERRORS_IN_ROW = 120
# YOLO on CPU can legitimately take ~0.4 s a pass; flag only real hangs.
DETECT_STALL_SEC = 3.0


# One fixed BGR color per detector class so boxes stay the same color from
# pass to pass (there is no tracking, so per-box colors would flicker).
LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 230, 0),
    "car": (0, 140, 255),
    "bus": (0, 215, 255),
    "truck": (0, 80, 180),
    "traffic light": (255, 0, 255),
    "door": (255, 160, 0),
    "chair": (180, 105, 255),
    "table": (0, 128, 128),
    "stairs": (0, 0, 255),
    "pole": (255, 255, 0),
    "wall": (160, 160, 160),
    "sofa": (130, 0, 130),
    "plant": (50, 205, 154),
    "bottle": (255, 0, 128),
    "cup": (128, 128, 255),
    "laptop": (255, 255, 255),
    "bag": (42, 42, 165),
}
# Labels outside LABEL_COLORS (custom classes) pick from this by a stable hash.
_FALLBACK_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 128, 0),
    (0, 255, 255),
    (128, 255, 0),
    (255, 0, 0),
    (0, 255, 128),
    (200, 200, 0),
)


def label_color(label: str) -> Tuple[int, int, int]:
    key = (label or "").strip().lower()
    color = LABEL_COLORS.get(key)
    if color is not None:
        return color
    # zlib.crc32, not hash(): str hashes change every run.
    return _FALLBACK_COLORS[zlib.crc32(key.encode("utf-8")) % len(_FALLBACK_COLORS)]


def detect_interval_sec() -> float:
    """DETECT_INTERVAL_SEC from env: seconds between YOLO pass starts."""
    raw = (os.getenv("DETECT_INTERVAL_SEC") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_DETECT_INTERVAL_SEC
    except ValueError:
        value = DEFAULT_DETECT_INTERVAL_SEC
    return max(MIN_DETECT_INTERVAL_SEC, value)


def hold_passes(interval_sec: float) -> int:
    """Empty passes to keep the last labels for, ~DETECT_HOLD_SEC in time."""
    return max(1, round(DETECT_HOLD_SEC / interval_sec))


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
        self._detect_interval = detect_interval_sec()
        self.detection_hold = DetectionHold(
            ttl_frames=hold_passes(self._detect_interval)
        )
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
        self._space_held = False
        self._space_hold_mode = space_hold_available()
        # Detection runs on a worker: a YOLO pass on the render thread stalled
        # the display (and everything else holding the GIL) for its duration.
        self._detect_stop = threading.Event()
        self._detect_wake = threading.Event()
        self._detect_thread: Optional[threading.Thread] = None
        self._detect_lock = threading.Lock()
        self._detect_input: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._ptt_thread: Optional[threading.Thread] = None
        self._ptt_stop = threading.Event()
        self.watchdog = Watchdog()
        self._frame_errors_in_row = 0
        self._frame_error_counts: Dict[tuple, int] = {}
        self.framelog = FrameLogger(
            stream_stats=getattr(self.capture, "take_stats", None)
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

        # Runs on the render thread: no frame copy, and never block on TTS here
        # (a blocking cloud TTS call froze the whole window while it spoke).
        snap = self.store.snapshot(copy_rgb=False)
        if not snap.has_frame or snap.rgb_bgr is None:
            speak("No frame")
            return

        self._run_tool_speak("describe_scene", {"focus": question})

    def _ptt_press(self, hold: bool = False) -> None:
        """Start/arm listen on Space press (hold) or toggle key."""
        if self.voice is None:
            log("app", "voice unavailable — need Deepgram + OpenRouter")
            speak("Voice unavailable. Need Deepgram and OpenRouter keys.")
            return
        log(
            "app",
            "PTT press",
            voice_state=self.voice.state.value,
            active=int(self.voice.active),
        )
        if not self.voice.available:
            speak("Voice unavailable.")
            return
        if not self.voice.active:
            self.voice.start_session()
            self.voice.arm_listen(hold=hold)
            return
        st = self.voice.state.value
        if st == "speaking":
            self.voice.request_ptt_barge_in(hold=hold)
            return
        if st == "listening":
            return
        self.voice.arm_listen(hold=hold)

    def _ptt_release(self) -> None:
        """End listen turn when Space is released (true hold-to-talk)."""
        if self.voice is None or not self.voice.active:
            return
        armed = bool(
            getattr(self.voice, "mic_gate", None) and self.voice.mic_gate.armed
        )
        log(
            "app",
            "PTT release",
            voice_state=self.voice.state.value,
            armed=int(armed),
        )
        # Keyed off the gate, not the HUD state: the agent can report "thinking"
        # mid-hold, and gating the release on state=="listening" then left the
        # mic armed with nobody to close it.
        if armed or self.voice.state.value in ("listening", "connecting"):
            self.voice.disarm_listen()

    def _toggle_voice(self) -> None:
        """Fallback toggle for `v` when Space hold is unavailable or unused."""
        if self.voice is None:
            log("app", "voice unavailable — need Deepgram + OpenRouter")
            speak("Voice unavailable. Need Deepgram and OpenRouter keys.")
            return
        log(
            "app",
            "key v toggle",
            voice_state=self.voice.state.value,
            active=int(self.voice.active),
        )
        if not self.voice.available:
            speak("Voice unavailable.")
            return
        if not self.voice.active:
            self.voice.start_session()
            self.voice.arm_listen(hold=False)
            return
        st = self.voice.state.value
        if st == "speaking":
            self.voice.request_ptt_barge_in(hold=False)
            return
        if st == "listening":
            self.voice.disarm_listen()
            return
        self.voice.arm_listen(hold=False)

    def _poll_space_hold(self) -> None:
        """True hold-to-talk: Space down = listen, Space up = end turn."""
        if not self._space_hold_mode:
            return
        down = is_space_down()
        if down and not self._space_held:
            self._ptt_press(hold=True)
        elif not down and self._space_held:
            self._ptt_release()
        self._space_held = down

    def _ptt_watch(self) -> None:
        """Sample Space on a fixed cadence, independent of render-loop speed."""
        while not self._ptt_stop.is_set():
            try:
                self._poll_space_hold()
            except Exception as exc:  # a PTT hiccup must not kill the thread
                log("app", f"ptt poll error: {type(exc).__name__}: {exc}")
            self._ptt_stop.wait(PTT_POLL_SEC)

    def _start_ptt_watch(self) -> None:
        if not self._space_hold_mode or self._ptt_thread is not None:
            return
        self._ptt_stop.clear()
        self._ptt_thread = threading.Thread(target=self._ptt_watch, daemon=True)
        self._ptt_thread.start()

    def _stop_ptt_watch(self) -> None:
        self._ptt_stop.set()
        if self._ptt_thread is not None:
            self._ptt_thread.join(timeout=1.0)
            self._ptt_thread = None

    def _submit_detect(self, rgb, depth, conf) -> None:
        """Hand the newest frame to the detector worker; latest wins.

        Overwrites any frame the worker has not picked up yet, so each pass
        labels the freshest frame available when it starts.
        """
        if self.detector is None:
            return
        with self._detect_lock:
            self._detect_input = (rgb, depth, conf)
        self._detect_wake.set()

    def _detect_watch(self) -> None:
        # Load the model and compile GPU kernels before frames flow, so the
        # first detections are not a multi-second hitch mid-demo.
        t0 = time.monotonic()
        try:
            self.detector.warmup()
            log("app", "detector warm", ms=round((time.monotonic() - t0) * 1000))
        except Exception as exc:
            log_exc("app", "detector warmup failed", exc)
        last_start = 0.0
        while not self._detect_stop.is_set():
            self.watchdog.beat("detect", limit_sec=DETECT_STALL_SEC)
            # Wait out the interval *before* taking a frame: taking it first and
            # then sleeping ran YOLO on a frame that was already stale.
            wait = self._detect_interval - (time.monotonic() - last_start)
            if wait > 0:
                self._detect_stop.wait(min(wait, 0.1))
                continue
            if not self._detect_wake.wait(0.1):
                continue
            self._detect_wake.clear()
            with self._detect_lock:
                job = self._detect_input
                self._detect_input = None
            if job is None:
                continue
            last_start = time.monotonic()
            try:
                fresh = self.detector.detect(*job)
                self.objects = self.detection_hold.update(fresh)
                self.framelog.detected((time.monotonic() - last_start) * 1000.0)
            except Exception as exc:
                log("app", f"detect error: {type(exc).__name__}: {exc}")
                print(traceback.format_exc(), flush=True)
        self.watchdog.forget("detect")

    def _start_detect_watch(self) -> None:
        if self.detector is None or self._detect_thread is not None:
            return
        self._detect_stop.clear()
        self._detect_thread = threading.Thread(target=self._detect_watch, daemon=True)
        self._detect_thread.start()

    def _stop_detect_watch(self) -> None:
        self._detect_stop.set()
        self._detect_wake.set()
        if self._detect_thread is not None:
            self._detect_thread.join(timeout=2.0)
            self._detect_thread = None

    def _handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            log("app", "quit key")
            return False

        # Space is handled by HID hold polling; ignore waitKey (incl. key-repeat).
        if key == ord(" "):
            if not self._space_hold_mode:
                self._toggle_voice()
            return True

        if key in (ord("v"), ord("V")):
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
        s = min(1.0, DISPLAY_MAX_H / float(rgb.shape[0]))
        if s < 1.0:
            size = (int(rgb.shape[1] * s), int(rgb.shape[0] * s))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
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
            x1, y1b, x2, y2 = (int(v * s) for v in o.box)
            color = label_color(o.label)
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
        help_line = "hold Space=talk  v=toggle  d/m/p/c/f  q=quit"
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
            if self._space_hold_mode:
                print("Voice: Deepgram + OpenRouter — hold Space to talk (v=toggle).")
            else:
                print("Voice: Deepgram + OpenRouter — Space/v toggle PTT.")
            log(
                "realtime",
                "Deepgram + OpenRouter enabled",
                space_hold=int(self._space_hold_mode),
            )
        else:
            if voice_ready():
                print("Voice off — install Deepgram and sounddevice dependencies.")
            else:
                print("Voice off — set DEEPGRAM_API_KEY and OPENROUTER_API_KEY.")
            print("  Keyboard: d=scene  m=distance  p/c/f=find still work.")
            log("realtime", "disabled")
        print("  Logs: lines starting with [nekit ...] — copy those for debug.")
        if log_path():
            print(f"  Full session log (survives crashes): {log_path()}")
        print("  Tip: click the OpenCV window so Space is tracked.")

        # Boot phrase uses offline `say` so startup never hangs on a slow/
        # blocked network reaching cloud TTS.
        speak("Assistant ready", local=True)

        self._start_ptt_watch()
        self._start_detect_watch()
        # Open the agent socket now, not on the first Space press, so the first
        # thing the user says is spoken into a live session.
        if self.voice is not None:
            self.voice.prewarm()

        self.watchdog.start()
        exit_reason = "quit key"
        try:
            while True:
                self.watchdog.beat("main")
                if self.voice is not None:
                    self.voice.tick()
                fl = self.framelog
                t0 = time.perf_counter()
                if not self.capture.wait_frame(timeout=0.3):
                    fl.wait_timeout()
                    fl.maybe_summary()
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break
                    continue
                t1 = time.perf_counter()
                fl.stage("wait", (t1 - t0) * 1000.0)

                # Clear before grabbing: clearing afterwards threw away the
                # notification for any frame that landed mid-grab, so the loop
                # kept waiting on an event that had already fired.
                self.capture.clear_event()
                try:
                    key = self._frame_step(fl, t1)
                except Exception as exc:
                    key = self._frame_failed(exc)
                else:
                    self._frame_errors_in_row = 0
                if key != 255 and not self._handle_key(key):
                    break
        except KeyboardInterrupt as exc:
            exit_reason = f"interrupted ({str(exc) or 'Ctrl+C'})"
            print("\nInterrupted.")
        except BaseException as exc:
            exit_reason = f"crash: {type(exc).__name__}: {exc}"
            raise
        finally:
            log("app", "main loop exit", reason=exit_reason, frames=self.frame_i)
            self.watchdog.forget("main")
            self._stop_ptt_watch()
            self._stop_detect_watch()
            if self.voice is not None and self.voice.active:
                self.voice.end_session(announce=False)
            self.beep.close()
            self.capture.stop()
            self.watchdog.stop()
            cv2.destroyAllWindows()
            print("Bye.")

    def _frame_step(self, fl: FrameLogger, t1: float) -> int:
        """Grab, process, draw and show one frame; returns the pressed key."""
        frame = self.capture.grab()
        if frame is None:
            fl.grab_failed()
            # Keep the window responsive even when frames cannot be used.
            return cv2.waitKey(1) & 0xFF
        t2 = time.perf_counter()
        fl.stage("grab", (t2 - t1) * 1000.0)
        fl.frame(frame)

        self.frame_i += 1
        raw_zones = compute_zones(frame.depth, frame.conf)
        zones = self.zone_filter.update(raw_zones)
        t3 = time.perf_counter()
        fl.stage("zones", (t3 - t2) * 1000.0)

        if self.detector:
            self._submit_detect(frame.rgb_bgr, frame.depth, frame.conf)
        else:
            self.objects = []

        objects = self.objects
        self.beep.update(zones, objects)
        self.scene = build_scene(zones, objects, device=frame.device_name)
        self.last_rgb = frame.rgb_bgr
        self.store.publish(frame.rgb_bgr, self.scene, objects)
        t4 = time.perf_counter()
        fl.stage("publish", (t4 - t3) * 1000.0)

        vis = self._draw(frame.rgb_bgr, frame.depth, zones, objects)
        t5 = time.perf_counter()
        fl.stage("draw", (t5 - t4) * 1000.0)
        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(1) & 0xFF
        fl.stage("show", (time.perf_counter() - t5) * 1000.0)
        fl.rendered()
        fl.maybe_summary(frame, zones, objects)
        return key

    def _frame_failed(self, exc: Exception) -> int:
        """Log a per-frame failure and skip the frame; re-raise if persistent.

        Must be called from inside the ``except`` block (uses bare ``raise``).
        """
        self._frame_errors_in_row += 1
        tb = traceback.extract_tb(exc.__traceback__)
        where = f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else "?"
        sig = (type(exc).__name__, where)
        count = self._frame_error_counts.get(sig, 0) + 1
        self._frame_error_counts[sig] = count
        # Full traceback the first few times, then a periodic count, so a
        # 60 fps failure cannot flood the log.
        if count <= 3 or count % 100 == 0:
            log(
                "app",
                f"frame error: {type(exc).__name__}: {exc}",
                at=where,
                count=count,
                in_row=self._frame_errors_in_row,
                frame=self.frame_i,
            )
            if count <= 3:
                print(traceback.format_exc().rstrip(), flush=True)
        if self._frame_errors_in_row >= MAX_FRAME_ERRORS_IN_ROW:
            log("app", "too many frame errors in a row — giving up", in_row=self._frame_errors_in_row)
            raise
        return cv2.waitKey(1) & 0xFF
