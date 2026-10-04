"""OpenRouter vision LLM with scene / distance / find modes."""

from __future__ import annotations

import base64
import json
import os
import threading
from typing import Optional

import cv2
import httpx
import numpy as np

from ..agent.modes import Intent
from ..perception.scene import compact_scene_for_llm

SCENE_PROMPT = """You are an orientation assistant for a blind person.

SCENE mode:
- Answer the user's actual question, to the point, in 2–3 short English sentences.
- Lead with the actionable part (what matters for walking: space type, path, key objects/people and their side).
- Use left / center / right for sides.
- No meters unless they are in SENSOR_JSON. Never invent objects or distances.
- No atmosphere, no long lists, skip trivial small items.
"""

DISTANCE_PROMPT = """You are an orientation assistant for a blind person.

DISTANCE mode:
- EXACTLY 1 short sentence.
- Meters and sides ONLY from SENSOR_JSON (zones_m / objects.dist_m).
- Name 1–2 nearest important objects or L/C/R zones.
- Start with “In front of you…”. Do not invent meters.
"""

FIND_PROMPT = """You are an orientation assistant for a blind person.

FIND mode:
- EXACTLY 1 sentence: found/not found, side, meters from JSON if present.
- If not found — “I don’t see …”.
- Start with “In front of you…” or “I don’t see”.
"""

# (question cues, object label substrings)
_FIND_MATCHES = (
    (("door", "двер"), ("door",)),
    (("person", "people", "человек", "люд"), ("person",)),
    (("car", "машин"), ("car",)),
    (("bus", "автобус"), ("bus", "truck")),
    (("table", "стол"), ("table",)),
    (("chair", "стул"), ("chair",)),
    (("traffic", "светофор"), ("traffic light", "traffic")),
    (("stairs", "лестниц"), ("stairs",)),
)


def _find_object_in_scene(question: str, objs: list) -> Optional[dict]:
    q = (question or "").lower()
    for cues, labels in _FIND_MATCHES:
        if not any(c in q for c in cues):
            continue
        for o in objs:
            lab = str(o.get("label", "")).lower()
            if any(l in lab for l in labels):
                return o
    return None


def _clip_spoken(text: str, max_chars: int = 320) -> str:
    """Keep TTS tight but useful: up to a few sentences, hard length cap.

    Earlier the answerer felt "light" because replies were cut to the first
    sentence. We now keep whole sentences up to ``max_chars`` so the user gets
    the actionable 2–3 sentence answer, then hard-cap length as a safety net.
    """
    t = (text or "").strip()
    if not t:
        return t
    if len(t) <= max_chars:
        return t
    # Keep as many whole sentences as fit under the cap.
    kept = ""
    rest = t
    while rest:
        nxt = -1
        for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
            i = rest.find(sep)
            if i >= 0 and (nxt < 0 or i < nxt):
                nxt = i
        if nxt < 0:
            break
        candidate = kept + rest[: nxt + 1]
        if len(candidate) > max_chars:
            break
        kept = candidate.rstrip() + " "
        rest = rest[nxt + 2 :]
    kept = kept.strip()
    if kept:
        return kept
    # No sentence boundary fit — hard cap on words.
    cut = t[: max_chars - 1].rsplit(" ", 1)[0]
    return (cut or t[: max_chars - 1]).rstrip(".,;:") + "."


# A walking user cannot wait 45 s for "what's ahead" — by then the answer
# describes a scene they already left. Fail over to the offline template instead.
VISION_CONNECT_TIMEOUT = 5.0
VISION_READ_TIMEOUT = 20.0


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = VISION_READ_TIMEOUT,
    ):
        self.api_key = (
            os.getenv("OPENROUTER_API_KEY", "") if api_key is None else api_key
        )
        self.model = model or os.getenv(
            "OPENROUTER_MODEL", "openai/gpt-4o-mini"
        )
        self.timeout = float(os.getenv("LLM_READ_TIMEOUT") or timeout)
        self.connect_timeout = float(
            os.getenv("LLM_CONNECT_TIMEOUT") or VISION_CONNECT_TIMEOUT
        )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        # One pooled client for the process: a fresh httpx.Client per call threw
        # away the TLS session and paid a full handshake on every question.
        self._client: Optional[httpx.Client] = None
        self._client_lock = threading.Lock()
        # Smaller image = faster upload / vision
        self.jpeg_max_side = int(os.getenv("LLM_JPEG_MAX_SIDE", "768"))
        self.jpeg_quality = int(os.getenv("LLM_JPEG_QUALITY", "80"))
        self.max_tokens_override = os.getenv("LLM_MAX_TOKENS")
        # phash gate cache (last frame+question → answer).
        self._cache_key: Optional[tuple] = None
        self._cache_answer: str = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(
                        self.timeout,
                        connect=self.connect_timeout,
                    ),
                    limits=httpx.Limits(
                        max_keepalive_connections=2, max_connections=4
                    ),
                )
            return self._client

    def close(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _encode_jpeg(self, rgb_bgr: np.ndarray) -> str:
        img = rgb_bgr
        h, w = img.shape[:2]
        scale = self.jpeg_max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _prompt_and_tokens(self, mode: Intent) -> tuple[str, int]:
        if self.max_tokens_override:
            cap = int(self.max_tokens_override)
        else:
            cap = 150
        # DISTANCE/FIND stay terse (single fact); SCENE gets room for 2–3 sentences.
        if mode == Intent.DISTANCE:
            return DISTANCE_PROMPT, min(cap, 60)
        if mode == Intent.FIND:
            return FIND_PROMPT, min(cap, 55)
        return SCENE_PROMPT, cap

    def describe(
        self,
        rgb_bgr: np.ndarray,
        scene: dict,
        user_question: str = "Describe the scene.",
        mode: Intent = Intent.SCENE,
    ) -> str:
        if not self.available:
            return _clip_spoken(self._offline_fallback(scene, user_question, mode))

        # phash gate: identical frame + same question → reuse cached answer and
        # skip a GPT-4o round-trip. Disabled transparently if imagehash absent.
        key = self._frame_cache_key(rgb_bgr, user_question, mode)
        if key is not None and key == self._cache_key:
            return self._cache_answer

        answer = self._describe_online(rgb_bgr, scene, user_question, mode)

        if key is not None:
            self._cache_key = key
            self._cache_answer = answer
        return answer

    def _frame_cache_key(
        self, rgb_bgr: np.ndarray, user_question: str, mode: Intent
    ) -> Optional[tuple]:
        try:
            import imagehash
            from PIL import Image
        except Exception:
            return None
        try:
            ph = imagehash.phash(Image.fromarray(rgb_bgr[:, :, ::-1]))
        except Exception:
            return None
        return (str(ph), (user_question or "").strip().lower(), mode)

    def _describe_online(
        self,
        rgb_bgr: np.ndarray,
        scene: dict,
        user_question: str,
        mode: Intent,
    ) -> str:
        system, max_tokens = self._prompt_and_tokens(mode)
        compact = compact_scene_for_llm(scene)
        # Fewer objects → shorter prompt / faster
        objs = compact.get("objects") or []
        compact["objects"] = objs[:6]
        b64 = self._encode_jpeg(rgb_bgr)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Question: {user_question}\n\n"
                                f"SENSOR_JSON:\n"
                                f"{json.dumps(compact, ensure_ascii=False)}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.15,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nekit-assist",
            "X-Title": "nekit-assist-mvp",
        }
        try:
            r = self._http().post(self.base_url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            raw = data["choices"][0]["message"]["content"].strip()
            return _clip_spoken(raw)
        except Exception as e:
            print(f"OpenRouter error: {e}")
            return _clip_spoken(self._offline_fallback(scene, user_question, mode))

    def _offline_fallback(
        self, scene: dict, question: str, mode: Intent
    ) -> str:
        zones = scene.get("zones_m") or {}
        objs = scene.get("objects") or []
        people = [o for o in objs if "person" in str(o.get("label", "")).lower()]
        cars = [
            o
            for o in objs
            if any(
                k in str(o.get("label", "")).lower()
                for k in ("car", "bus", "truck")
            )
        ]

        if mode == Intent.SCENE:
            bits = []
            if people:
                n = len(people)
                bits.append(f"{n} {'person' if n == 1 else 'people'} in frame")
            if cars:
                bits.append("vehicles nearby")
            big = [
                o
                for o in objs
                if o.get("label")
                in ("table", "door", "chair", "sofa", "stairs", "wall")
            ]
            if big:
                bits.append(f"large: {big[0]['label']}")
            if not bits:
                bits.append("a scene with no clear landmarks in the detector")
            return "In front of you " + "; ".join(bits) + "."

        if mode == Intent.DISTANCE:
            if people:
                p = people[0]
                return (
                    f"In front of you a person {p.get('bearing')}, "
                    f"about {p.get('dist_m')} meters."
                )
            parts = []
            for name, side in (
                ("left", "left"),
                ("center", "center"),
                ("right", "right"),
            ):
                v = zones.get(side)
                if v is not None:
                    parts.append(f"{name} {v} m")
            return (
                "In front of you zones: "
                + (", ".join(parts) if parts else "no data")
                + "."
            )

        hit = _find_object_in_scene(question, objs)
        if hit:
            label = hit.get("label")
            if "person" in str(label).lower():
                label = "a person"
            return (
                f"In front of you {label}: {hit.get('bearing')}, "
                f"about {hit.get('dist_m')} m."
            )
        return "I don't see that object in frame."
