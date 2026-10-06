# Kaggle YOLO11n runtime — 6 October 2026

User-authorized model switch: `backend/.env` now selects the existing Kaggle
YOLO11n detection export at 640, CPU only, 2 inference threads. The previous
industoai exports remain available. Detection task is read from model metadata;
classification remains disabled because this model has only the class `egg`.

## Tracker comparison

Run `.venv/Scripts/python.exe -m backend.tune_tracker` from the project root.
The tool infers all 262 decoded frames once, then replays identical detections
through six isolated trackers and counters. It does not touch the live ledger.
JSON: `artifacts/model-evaluation/tracker-tuning/comparison.json`.

All profiles use confidence .10, NMS IoU .65, max detections 300, low threshold
.10, buffer 45, the existing orange polyline and unchanged arming/recovery rules.

| Profile | High / new | Match | Fuse score | Near-line track observations | Unique tracker IDs | Recovered / ambiguous | Crossing events |
|---|---|---|---|---:|---:|---|---:|
| Baseline | .25 / .30 | .80 | yes | 4,936 | 565 | 56 / 13 | 308 |
| Lower confidence | .20 / .25 | .80 | yes | 4,938 | 593 | 71 / 17 | 310 |
| **Selected: geometry** | **.25 / .30** | **.80** | **no** | **5,261** | **595** | **24 / 3** | **352** |
| Lower confidence geometry | .20 / .25 | .80 | no | 5,412 | 643 | 35 / 5 | 363 |
| Strict geometry | .20 / .25 | .70 | no | 5,395 | 699 | 62 / 17 | 362 |
| Sensitive geometry | .15 / .20 | .70 | no | 5,525 | 750 | 75 / 31 | 369 |

All six track 72/72 partial reference centres. Selected profile increases
near-line tracking coverage and reduces ambiguous physical-ID recovery on this
clip while retaining conservative birth thresholds. It also creates more IDs;
these proxies do not prove fewer identity switches or more accurate counts.
Median detector inference measured 76.6 ms/frame in this run, excluding tracker,
counter, JPEG, transport and rendering. Performance varies with machine load.

**352 is an algorithm output, not verified ground truth.** Kaggle training data
includes this stock clip and validation reused its training folder. No independent
per-egg crossing ledger is available. Use a held-out clip before accuracy claims.

## Viewer

- Video fills the window with aspect ratio preserved. One counter at top right.
- Blue rectangles: uncounted tracks. Green: counted tracks, not quality labels.
- Red dot: exact centre used by the line-crossing state machine (box centre for
  this detector). Shape contours are no longer drawn.
- Eggs: cumulative crossing events. Detections: raw model boxes in current frame.
  Tracked: confirmed observations; unmatched boxes do not trigger counting.
- Counter values and decoded JPEG are presented together. Reconnection takes
  authoritative backend totals, without incrementing again from replay events.
- Play/Pause control the actual worker. Replay creates a fresh counting session;
  prior SQLite events remain accessible. No automatic looping or frame skipping.
- Stream limit raised to 25 FPS to remove the old 10 FPS transport ceiling.
  This is a limit, not a claim of CPU throughput or full-speed 25 FPS playback.

Tracking parameter semantics: [Ultralytics documentation](https://docs.ultralytics.com/modes/track/).
