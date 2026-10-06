# Verification — 5 October 2026

- Python 3.12.14 environment in `.venv`; CPU-only PyTorch 2.7.1 / torchvision 0.22.1.
- `python -m pytest backend/tests -q`: **22 passed** (28.61 seconds).
- Actual OpenVINO 2025.4.0 CPU inference verified with synthetic segmentation
  and classification IR graphs. Execution device reported `CPU` for both models.
- Actual Ultralytics 8.3.228 ByteTrack verified for empty-frame occlusion and
  low-confidence association. Instance-mask indices verified against tracker output.
- `pip check`: no broken requirements. Python source compilation passed.
- `npm run build`: React/Vite production bundle built successfully.
- Browser: live canvas feed, connected WebSocket, counters, particles,
  quality ring, defect filter, pause/resume display and recent-event ordering checked.
- Browser console: no errors/warnings at the checked state.
- Export button fetched the durable event API successfully (HTTP 200).
  The in-app browser did not expose a completed download event to automation,
  so automatic verification of the downloaded CSV file was not completed.
- PowerShell launch script parsed without syntax errors; background launcher
  was not executed because the backend and frontend were already running.

Screenshot: `artifacts/dashboard.jpg` (labelled demo data).

No trained egg weights or real conveyor footage were provided. Real detection
accuracy, defect accuracy, camera performance and the 100% physical-counting
target remain unverified. Synthetic IR tests verify integration, not model quality.

## Earlier update: user-supplied test video

- Added `Eggs On A Conveyor Stock Video.mp4` as the configured source in
  `backend/.env`; video preview is the default for this workspace.
- OpenCV decoded the actual file: 854 × 480, 25 FPS, 267 frames (approximately
  10.68 seconds). The browser reports approximately 10.675 seconds from MP4 metadata.
- `python -m pytest backend/tests -q`: **25 passed** (15.32 seconds), including
  preview metadata/poster, zero fabricated count events, HTTP 206 byte-range
  playback/seek, invalid range handling and missing-file behavior.
- Browser verified actual MP4 decoding, play, automatic stop at end without
  looping, explicit replay and pause. No video decode or console error was found.
- Video was left **paused**. No synthetic conveyor worker is running.
- The trained egg segmentation/classification models are still absent. Video
  preview does not perform AI counting; HUD statistics display `—` accordingly.

## Earlier update: public trained-model evaluation

- Downloaded three public trained egg candidates with pinned source revisions
  and SHA-256 manifests in `models/candidates/`. Full findings and source/license
  links are in `MODEL_EVALUATION.md`.
- Strongest experimental model: industoai YOLOv8n-seg, OpenVINO 416,
  **71/72 reference points covered**. This is a partial visual audit, not count
  accuracy. The other candidates covered 43/72 and 0/72 at the checked settings.
- Processed all **262 decodable frames** sequentially in three nano OpenVINO
  configurations. Metadata reports 267 frames; a separate full read-through
  confirmed the decoding discrepancy. No claim of full-clip count accuracy.
- Verified OpenVINO execution on CPU only, two threads, one stream; observed
  pipeline rates 1.93–3.54 FPS on the local Intel Core Ultra 5 225H.
- Candidate line-cross totals (222/276/288) are experimental outputs with no
  independently verified physical crossing ledger. They were not published to
  the live dashboard. No normal/defect results were fabricated.
- **No candidate activated.** Application remains in paused video preview mode.
- Measured main motion left to right; configured `EGG_LINE_AXIS=x` for a
  vertical transverse line at 55% width. Browser DOM and screenshot confirmed
  the vertical line, `paused=true`, `autoplay=false`, `loop=false`, zero time,
  and 854 × 480 decoded video dimensions. API confirms zero counts/events.
- `python -m pytest backend/tests -q`: **28 passed** (22.58 seconds), one
  existing Starlette/AnyIO deprecation warning. New audit tests prevent a merged
  instance from passing for two reference eggs and require mask coverage.
- `npm run build`: passed (6m35s). Sandbox initially blocked Vite's temporary
  config file; retry with authorized build permissions succeeded.
- Screenshot: `artifacts/model-evaluation/dashboard-line-vertical.png`.
- One temporary automatic-review capacity failure affected an edit attempt;
  retry succeeded. No work remains blocked by that review failure.

## Current update: user-selected model and orange counting line

- Activated industoai/Egg-Instance-Segmentation, exported at 416 with OpenVINO
  CPU inference and explicit White-Egg/Brown-Egg class mapping.
- Replaced the vertical line with a finite polyline traced from the user's
  orange guide. Backend intersection tests, annotated frames, the dashboard
  SVG, and crossing particle positions use the same normalized coordinates.
- Surface classification is disabled. Counts are marked Not Inspected, with
  null confidence; inspection totals/ratios display unavailable rather than
  fabricated good/defect results.
- `python -m pytest backend/tests -q`: **34 passed** (14.72 seconds), one
  existing Starlette/AnyIO deprecation warning. New tests cover finite segment
  intersections, polyline bends, reverse motion, count-only behavior, actual
  worker pause/resume/reset, and invalid/origin-restricted control requests.
- `npm run build`: passed (2m51s).
- Browser and actual model verified Play/Pause. At the pause check, frames
  stayed at 172 and counts at 135. Evidence: `real_pause_check.json`.
- Full run processed 262 decoded frames, stopped at EOF without automatic
  replay, and generated 204 events/204 distinct ledger identities. Classifier
  calls/errors were both zero. This verifies integration, not physical count
  accuracy or absence of physical duplicate counts. Evidence:
  `artifacts/model-evaluation/real_orange_line_run.json`.
- Browser Replay created a new session, with fresh tracking/counts and actual
  inference. Pausing that round stopped at 178 frames/141 counts. Browser
  Reset then created another session, paused at the first frame with zero counts.
  `real_reset_check.json` confirms no frames processed while paused, an empty
  new ledger, and retention of all 204 events in the original session.
- Browser console: no errors/warnings at the final checked state.
- Final screenshot: `artifacts/model-evaluation/ai-orange-line-ready.png`.
  Current server is in real mode, paused and ready for the user's Play action.

## 6 October 2026 — offline diagnosis and Kaggle YOLO11n comparison

- Retrieved Kaggle notebook version 1 `runs/egg_yolo11n/weights/best.pt`, verified
  checkpoint size and SHA-256, and exported fixed-size OpenVINO 640 Detect IR.
  The export and both model runs reported CPU execution only.
- Processed all 262 decoded frames separately with industoai/416, industoai/640,
  and Kaggle/640 using the same orange line and ByteTrack settings. Per-frame
  detection, tracking, line-state, partial reference, event and latency traces
  are saved in `artifacts/model-evaluation/pipeline-diagnostics/`.
- Baseline industoai/416 reproduced the application's 204 crossing events.
  See `KAGGLE_COMPARISON.md` for the full comparison and leakage caveat.
- `python -m pytest backend/tests -q -p no:cacheprovider
  --basetemp=artifacts/model-evaluation/pytest-temp-20261006`:
  **34 passed**, one existing Starlette/AnyIO deprecation warning.
  The explicit basetemp was needed because the default Windows sandbox temp
  directory did not permit pytest to create test directories in this session.
- Python compilation, model-file hash and generated OpenVINO files verified.
  The active `backend/.env` and live model selection were not changed.
