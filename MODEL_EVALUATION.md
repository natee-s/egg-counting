# Model evaluation — 5 October 2026

**Current configuration:** the user explicitly selected industoai for use after
reviewing these results. It is now active for counting only, at OpenVINO 416,
with the polyline traced from the user's orange guide. This selection does not
change the measured validation results or certify 100% physical accuracy.
Play/Pause controls the actual worker; Replay/Reset creates a fresh session.
Quality classification remains disabled.

**Earlier evaluation decision: no candidate passed the accuracy acceptance gate.** The strongest experimental
candidate is `industoai/Egg-Instance-Segmentation` (YOLOv8n-seg), exported at 416.
At that stage the application stayed in paused MP4 preview mode.

## Source and reference audit

- Video: `Eggs On A Conveyor Stock Video.mp4`, 854 × 480, nominal 25 FPS.
- OpenCV/FFmpeg reports 267 frames in metadata, but sequential decoding produces
  **262 frames**. Both decoding and a separate read-through reproduced this
  discrepancy. This is not a verified full-clip counting benchmark.
- Optical flow in the central conveyor: median approximately **(+5.99, −0.70)
  pixels per frame** over the first 50 frames, after excluding stationary points.
  The main image direction is left to right. The earlier experimental line was
  **vertical at x = 55% (469.7 px)**, across that flow, rather than horizontal
  along it. Perspective produces some vertical drift. The current application
  uses the user's orange polyline instead; those earlier totals are not comparable.
- The assistant visually marked **72 reference egg centres** on raw frames
  0, 120 and 240 (23, 25 and 24 points). These are partial central-conveyor
  annotations, not exhaustive ground truth, not independently reviewed, and not
  temporal egg identities. Coordinates and methodology are saved in
  `artifacts/model-evaluation/reference_points.json`.
- A reference matches only when its centre lies inside a predicted box **and
  instance mask** for segmentation. A prediction can explain only one reference
  egg. Merged masks therefore cannot pass for two eggs.
- These results measure **reference-point instance coverage**, not mAP, precision,
  overall detection accuracy, physical count accuracy, or defect accuracy.

## Candidates tested

| Candidate | Architecture | Result on the 72-point audit | Decision |
|---|---|---:|---|
| [industoai/Egg-Instance-Segmentation](https://huggingface.co/industoai/Egg-Instance-Segmentation) | YOLOv8n-seg, white/brown eggs | **71/72 (98.6%)**, OpenVINO 416 | Best experimental candidate; still misses an egg |
| [Yossefmohammed/EGGS-counting-Detection-with_YOLOV8-Model](https://github.com/Yossefmohammed/EGGS-counting-Detection-with_YOLOV8-Model) | YOLOv8s-seg, EGG | **43/72 (59.7%)**, PyTorch CPU 640 | Rejected for this footage |
| [Williamsanderson/PoultryVision](https://huggingface.co/Williamsanderson/PoultryVision/tree/main) | YOLO11m detection, chicken/egg | **0/72**, PyTorch CPU 640 | Rejected for this footage |

The point audit uses confidence 0.10 and NMS IoU 0.65. A separate screening run
at confidence 0.25 checked nine decoded frames (0, 30, …, 240); prediction
images and raw sample results are retained. The attempted sample at frame 266
could not be decoded and was not included in results.

Weights, pinned repository revisions, source URLs, byte counts and SHA-256
checksums are in `models/candidates/*.json`. The tested nano checkpoint is
6,777,261 bytes; its actual class names are `White-Egg` and `Brown-Egg`.

## OpenVINO + ByteTrack + crossing experiment

All 262 decodable frames were processed **sequentially**, without sampling,
looping, or quality classification. Counts below are unverified candidate
outputs, never official totals and never sent to the application dashboard.

| Nano configuration | Point coverage | Candidate crossings | Observed processing FPS | Median frame time |
|---|---:|---:|---:|---:|
| Whole frame, 640 | 69/72 | 276 | 1.93 | 517.6 ms |
| Whole frame, 416 | **71/72** | 222 | 3.54 | 287.5 ms |
| Central band x=260…679, 640 | 70/72 | 288 | 2.96 | 312.8 ms |

Runtime execution devices were verified as **CPU**, with two inference threads,
one stream, batch 1 and fixed input dimensions. Hardware reported by the local
runtime is Intel Core Ultra 5 225H. These are observed pipeline times on this
machine during development, not a controlled hardware benchmark or a speed
guarantee for another low-spec machine. All are below the source's 25 FPS.

Whole-frame 640 also reported two ambiguous identity recoveries; 416 had no
ambiguous recoveries in this experiment. Zero reported ambiguities does not prove
zero ID switches or double-counts. The 416 audit missed the reference at
**(484,184), frame 240**. Audit images show covered centres in green, missed
centres in red, and the transverse line in cyan.

The different totals cannot be used to choose the "correct" count. An independent
ledger of each physical egg crossing the line is still required to measure
missed crossings, duplicate crossings and wrong identities. Matching a final
total alone could hide cancelling misses and duplicates.

## Integration decision and licenses

- Current integration points directly to
  `models/candidates/egg_segmentor_416_openvino_model` in `EGG_MODE=real`.
  `White-Egg` and `Brown-Egg` are explicitly mapped as countable eggs.
  Classification is disabled; crossings use `status=unknown`,
  `label=not_inspected`, and `confidence=null`. Colour never declares an egg good.
- A full application run with the orange polyline processed 262 decoded frames
  and produced 204 crossing events with 204 distinct ledger identities.
  This is an algorithm output, not independently verified physical count accuracy.
  Pause stopped both frames and counts; completion did not replay automatically.
  Evidence: `artifacts/model-evaluation/real_orange_line_run.json`.
- The nano publisher labels its weights **CC BY-NC 4.0**. This restriction still
  applies to the source weights after export; Ultralytics' generated metadata
  must not be treated as replacing the publisher's license.
- PoultryVision is labelled **AGPL-3.0**. The GitHub counting project README
  claims MIT, but a separate license file/weight license was not verified.
- Next work required for acceptance: annotate representative conveyor instances
  and a physical crossing ledger; fine-tune the nano model on separate training
  footage; evaluate held-out footage for misses/duplicates; remeasure CPU speed.
  The current short, partial audit must not become both training data and the
  claimed independent acceptance test.

## Reproduce

The downloaded checkpoints and exports are already local. With `.venv` installed:

```powershell
.\.venv\Scripts\python.exe -m backend.evaluate_models --candidate egg_segmentor --size 416 --conf 0.10
.\.venv\Scripts\python.exe -m backend.evaluate_models --candidate egg_segmentor --size 640 --conf 0.10
.\.venv\Scripts\python.exe -m backend.evaluate_models --candidate egg_segmentor --size 640 --conf 0.10 --band
.\.venv\Scripts\python.exe -m backend.evaluate_models --candidate eggs_counting --size 640 --conf 0.10 --samples-only
.\.venv\Scripts\python.exe -m backend.evaluate_models --candidate poultryvision --size 640 --conf 0.10 --samples-only
```

Each run writes its frame counts, event coordinates, reference misses, runtime
measurements and `approved: false` to `artifacts/model-evaluation/*.json`.
It never changes application settings or the active model.

The dashboard line is configured in `backend/.env`:

```dotenv
EGG_MODE=real
EGG_SEG_MODEL=models/candidates/egg_segmentor_416_openvino_model
EGG_IMAGE_SIZE=416
EGG_EGG_LABELS=White-Egg,Brown-Egg
EGG_DEFECT_ENABLED=0
EGG_START_PAUSED=1
EGG_LINE_AXIS=x
EGG_LINE_POSITION=0.55
EGG_DIRECTION=1
EGG_LINE_POINTS=0.337:0,0.40:0.12,0.425:0.20,0.50:0.31,0.66:0.51,0.85:0.72,0.975:0.885
```
