# find_object: fresh focused detection pass — design

Date: 2026-10-04
Status: approved in chat (approach B + live-labeling pause), pending code review

## Problem

`find_object` (keys `p` / `c` / `f`, and the voice agent tool) never looks at
the image. It filters the object list left by the last background YOLO pass:
small `yolov8s-worldv2`, 17-class vocabulary, `imgsz=640`, `conf ≥ 0.25`,
every 0.5 s. Doors and people are sometimes missed, and the user hears
"I don't see door" while the door is in the picture. Objects without valid
LiDAR depth (glass doors, people beyond ~5 m) are dropped entirely.

## Goal

When find is triggered, run a new, stronger detection pass on the newest
frame, focused on the requested object, and answer from that.

Success:
- The answer comes from a pass started after the request, on the newest frame.
- Recall for door / person is higher than the stored labels (checked live).
- Weak hits are reported with doubt, never as certain.
- No new network dependency at run time; works offline once weights exist.

Out of scope: vision-LLM fallback, tracking, counting, measure_distances.

## Decisions (from the chat)

- Local only, stronger YOLO (no vision-model fallback).
- Weak hits are spoken with doubt ("possibly a door …").
- Bigger model now: `yolov8l-worldv2.pt` (swappable via `FIND_MODEL`).
- Live labeling pauses while the finder runs.

## Components

### `ObjectFinder` — new, `assist/perception/find.py`

Own YOLO-World instance, used only by find.

- `FIND_MODEL` env, default `yolov8l-worldv2.pt` (auto-downloads ~90 MB on
  first load; needs internet once).
- `FIND_IMGSZ` env, default `1280`. `conf` floor `0.10`.
- Device: same `yolo_device()` as the detector, with the same one-time
  MPS → CPU fallback on a predict error.
- `find(rgb_bgr, depth, conf_map, target) -> list[FindHit]`, sorted by score,
  highest first.
- Synonyms per target: `prompts_for(target)`, e.g.
  - `door` → door, doorway, glass door, entrance
  - `person` → person, pedestrian, child
  - `stairs` → stairs, staircase, steps
  - any other target → `[target]` (open vocabulary: "bench" works by voice)
- One fixed vocabulary `FIND_VOCAB` (all synonym groups + car, bus, truck,
  traffic light, chair, table, bench, pole, bicycle; no "wall"), set once at
  warm-up. Hits are filtered to the target's synonyms. A word not in it is
  appended once and kept.
  Measured on the M2 (2026-10-04): any `set_classes` call makes the next MPS
  predict ~1 s slower; switching vocabulary per target cost 1.2–3.2 s per
  find, a fixed vocabulary 261–348 ms. A new word costs ~1.9 s once.
- `warmup()` runs one pass on a dummy frame so the first real find does not
  pay model load + MPS kernel compile.
- Load failure (missing weights offline, etc.) → `find()` raises
  `FinderUnavailable`; the tool falls back (see Error handling).

### `FindHit` — new dataclass

`label` (canonical target), `prompt` (which synonym matched, e.g.
"glass door"), `score`, `dist_m: Optional[float]` (None = no valid depth;
kept, not dropped), `bearing` (same `bearing_from_cx` bins), `box`.

### `YOLO_LOCK` — new, module-level in `assist/perception/detect.py`

One `threading.Lock` around every YOLO predict (detector and finder), so two
models never run on MPS at the same time.

- Finder takes it blocking for the whole pass (waits at most one live pass).
- `ObjectDetector.detect(..., block=True)`; the background worker calls
  `block=False`: if the lock is busy it returns `None` (skipped) instead of
  waiting. This is the live-labeling pause: no flag that could stay stuck
  off after an error, and the detect thread keeps beating the watchdog
  instead of looking stalled during a slow find.

### `SceneStore` — changed

`publish(rgb, scene, objects, depth=None, conf=None)`; `SceneSnapshot` gains
`depth` and `conf` (references, no copy — the loop builds new arrays per
frame and never mutates published ones).

### `ToolRegistry` — changed

New optional `finder` dependency, injected like `ocr`.

### `AssistApp` — changed

- Creates `ObjectFinder` when YOLO is enabled; passes it to `ToolRegistry`.
- Publishes depth + conf with each frame.
- Warms the finder in the detect thread right after the detector warmup,
  before live labeling starts (logs `finder warm ms=…`). With `--no-yolo`:
  no finder; find uses stored labels as today.
- Worker treats `None` from `detect(block=False)` as "skipped": objects keep
  their current value, `det_fps` does not count it.

## Data flow

```
p / c / f key or voice tool call
  → ToolRegistry._find_object(label)
      snapshot = SceneStore.snapshot()          (newest frame + depth + conf)
      stale > 1.5 s → "No current camera frame" (unchanged)
      target = _match_labels(label)[0]          (existing alias table: "people" → person,
                                                 "дверь" → door, "автобус" → bus,
                                                 unknown word → itself, lowercased)
      finder present and depth present?
        yes → hits = finder.find(rgb, depth, conf, target)   [holds YOLO_LOCK]
              live labeling skips its passes meanwhile
        no  → stored-label lookup (current behavior)
      pick = select_hit(hits)
  → result dict → spoken phrase (keys) / JSON to agent (voice)
```

## Selection rules — `select_hit(hits)`

- Strong = `score ≥ 0.35`. Weak = `0.10 ≤ score < 0.35`.
- Any strong hit: nearest strong hit with a distance; if none has a distance,
  the highest-scoring strong hit.
- Only weak hits: the highest-scoring one.
- No hits: not found.

## Result and wording

Result dict (fresh path): `ok`, `found`, `label` (target), `prompt`,
`certain` (strong?), `score`, `bearing`, `dist_m` (number or null),
`source: "fresh"`, `age_ms`. Stored path keeps today's fields plus
`source: "stored"`, `certain: true`.

Spoken (keys; agent prompt tells it to use the same pattern):
- certain, distance: "In front of you a door: on the left, about 2.1 meters."
- uncertain, distance: "Possibly a door: on the left, about 2.1 meters."
- no distance: "… on the left, distance unknown."
- not found: "I don't see a door."

Agent prompt (`SYSTEM_INSTRUCTION`) find line gains: if `certain` is false say
"possibly"; if `dist_m` is null say the distance is unknown. Tool description
says it runs a fresh focused detection and accepts any object word.

## Error handling

- Finder raises (unavailable, predict error after CPU fallback) → log
  `find | finder failed`, fall back to the stored-label lookup; the answer is
  never an exception.
- Snapshot without depth (published by tests / older paths) → stored lookup.
- Slow pass (CPU fallback): no timeout; logged with `ms`. Proximity beeps
  are unaffected because zones run on the render thread.

## Logging

`find | pass target=door prompts=4 ms=… hits=… best=glass door 0.31 2.4m`

## Testing

Unit (fakes, no weights needed):
- `prompts_for`: synonyms, unknown passthrough.
- `select_hit`: strong-nearest, strong-without-distance, weak-best, empty.
- `ObjectFinder.find` with a fake model: `set_classes` called once per prompt
  change; hits without depth kept with `dist_m=None`; holds `YOLO_LOCK`.
- `ObjectDetector.detect(block=False)` returns `None` while `YOLO_LOCK` is held.
- Worker: a skipped pass leaves `objects` untouched and is not counted.
- `SceneStore` round-trips depth / conf.
- `ToolRegistry`: fresh path wording (certain / possibly / distance unknown /
  not found); finder exception → stored fallback; no finder → existing tests
  unchanged.

Real (needs weights, on this Mac):
- Benchmark script: load `yolov8l-worldv2.pt`, warm, time 5 finds at 1280 on
  MPS; record warm-up and per-find ms.
- Live check by the user: door and person that the stored labels miss.

## Config

`.env.example` + README: `FIND_MODEL`, `FIND_IMGSZ`.
