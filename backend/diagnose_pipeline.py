r"""Offline, read-only egg pipeline comparison on the configured MP4.

Usage (from project root)::

    .\.venv\Scripts\python.exe -m backend.diagnose_pipeline --model current
    .\.venv\Scripts\python.exe -m backend.diagnose_pipeline --model kaggle

This does not write the live SQLite ledger or change backend/.env. The 72
reference centres are a partial visual check, not a physical crossing ledger.
The Kaggle training set contains frames from this same stock video, so its
results here are diagnostic and cannot establish out-of-sample accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/ultralytics"))

import cv2
import numpy as np
import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from backend.main import Config, LineCounter, Observation, path_crossing, pixel_line

VIDEO = ROOT / "Eggs On A Conveyor Stock Video.mp4"
OUT = ROOT / "artifacts/model-evaluation/pipeline-diagnostics"
REFERENCES = ROOT / "artifacts/model-evaluation/reference_points.json"
MODELS = {
    "current": ("segment", ROOT / "models/candidates/egg_segmentor_416_openvino_model", 416),
    "current640": ("segment", ROOT / "models/candidates/egg_segmentor_640_openvino_model", 640),
    "kaggle": ("detect", ROOT / "models/candidates/kaggle_egg_yolo11n_openvino_model", 640),
    "kaggle416": ("detect", ROOT / "models/candidates/kaggle_egg_yolo11n_416_openvino_model", 416),
}


def unique_reference_matches(points, boxes, polygons):
    """Match one reference centre to at most one prediction, and vice versa."""
    candidates = []
    for point in points:
        candidates.append([
            i for i, (x1, y1, x2, y2) in enumerate(boxes)
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2
            and (not polygons or (i < len(polygons) and len(polygons[i]) >= 3
                 and cv2.pointPolygonTest(polygons[i].astype(np.float32), tuple(point), False) >= 0))
        ])
    prediction_to_reference = {}

    def assign(reference, visited):
        for prediction in candidates[reference]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in prediction_to_reference or assign(prediction_to_reference[prediction], visited):
                prediction_to_reference[prediction] = reference
                return True
        return False

    for reference in range(len(points)):
        assign(reference, set())
    return {reference: prediction for prediction, reference in prediction_to_reference.items()}


def load_model(name, cfg):
    task, directory, size = MODELS[name]
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing OpenVINO export: {directory}")
    # Reuse the live CPU-only compilation path, but keep tracker and counter
    # isolated from the running application.
    from backend.main import OpenVINOModels
    loader = OpenVINOModels.__new__(OpenVINOModels)
    loader.cfg = replace(cfg, image_size=size)
    model = loader.load(YOLO, ov, directory, task, size)
    tracker = BYTETracker(SimpleNamespace(
        track_high_thresh=.25, track_low_thresh=.10,
        new_track_thresh=.30, track_buffer=cfg.track_buffer,
        match_thresh=.80, fuse_score=True,
    ), frame_rate=30)
    return model, tracker, task, size


def analyze(name, limit=None):
    cfg = Config.from_env()
    if not cfg.line_points:
        raise ValueError("Expected the active orange polyline in backend/.env")
    torch.set_num_threads(cfg.cpu_threads)
    cv2.setNumThreads(1)
    model, tracker, task, size = load_model(name, cfg)
    counter = LineCounter(cfg)
    references = json.loads(REFERENCES.read_text(encoding="utf-8"))["frames"]
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")
    OUT.mkdir(parents=True, exist_ok=True)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rows, near_line, reference_rows, crossings = [], [], [], []
    reason_counts = Counter()
    stages = Counter()
    durations = []
    number = 0
    try:
        while limit is None or number < limit:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            started = time.perf_counter()
            result = model.predict(
                frame, imgsz=size, device="intel:cpu", conf=.10, iou=.65,
                classes=None, max_det=300, verbose=False, rect=False,
            )[0]
            boxes = result.boxes.cpu().numpy()
            polygons = result.masks.xy if result.masks is not None else []
            predicted = len(boxes)
            after_inference = time.perf_counter()
            tracks = tracker.update(boxes, frame)
            observations = []
            index_to_observation = {}
            for track in tracks:
                index = int(track[-1])  # Ultralytics' original detection index.
                contour = polygons[index] if index < len(polygons) else None
                x1, y1, x2, y2, track_id = track[:5]
                center = ((x1+x2)/2, (y1+y2)/2)
                if contour is not None and len(contour) >= 3:
                    moments = cv2.moments(contour.astype(np.float32))
                    if moments["m00"]:
                        center = (moments["m10"]/moments["m00"], moments["m01"]/moments["m00"])
                obs = Observation(int(track_id), tuple(map(float, center)),
                                  tuple(map(float, track[:4])), contour)
                observations.append(obs)
                index_to_observation[index] = obs

            # Snapshot the pre-update state so a rejected crossing can be
            # distinguished from a missing detector/tracker observation.
            before = {}
            for obs in observations:
                physical_id = counter.aliases.get(obs.tracker_id)
                state = counter.states.get(physical_id)
                if state is not None:
                    before[obs.tracker_id] = {
                        "distance": state.distance, "center": state.center,
                        "phase": state.phase.value, "last_frame": state.last_frame,
                    }
            current_crossings = counter.update(observations, number + 1, width, height)
            crossed = {obs.tracker_id: state.physical_id for state, obs, _ in current_crossings}
            for state, obs, point in current_crossings:
                crossings.append({"frame": number, "media_seconds": round(number / 25, 3),
                                  "physical_id": state.physical_id, "tracker_id": obs.tracker_id,
                                  "x": round(point[0], 1), "y": round(point[1], 1)})
            stages["raw_detections_all_frames"] += predicted
            stages["tracked_detections_all_frames"] += len(index_to_observation)
            line = pixel_line(cfg, width, height)
            for index, box in enumerate(boxes):
                xyxy = box.xyxy[0]
                raw_center = ((xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)
                obs = index_to_observation.get(index)
                center = obs.center if obs else raw_center
                distance = counter.distance(center, width, height)
                if abs(distance) > 35:
                    continue
                stages["raw_detections_near_line"] += 1
                row = {"frame": number, "detection_index": index,
                       "confidence": round(float(box.conf[0]), 3),
                       "x": round(float(center[0]), 1), "y": round(float(center[1]), 1),
                       "distance_px": round(float(distance), 1),
                       "tracked": obs is not None}
                if obs is None:
                    row["reason"] = "detected_but_not_tracked"
                else:
                    stages["tracked_detections_near_line"] += 1
                    physical_id = counter.aliases[obs.tracker_id]
                    state = counter.states[physical_id]
                    previous = before.get(obs.tracker_id)
                    row.update({"tracker_id": obs.tracker_id, "physical_id": physical_id,
                                "phase": state.phase.value,
                                "previous_phase": previous["phase"] if previous else None})
                    if obs.tracker_id in crossed:
                        row["reason"] = "counted"
                    elif previous and previous["distance"] < 0 <= distance:
                        if number + 1 - previous["last_frame"] > cfg.max_cross_gap:
                            row["reason"] = "stale_track_gap"
                        elif previous["phase"] == "counted":
                            row["reason"] = "already_counted"
                        elif previous["phase"] != "armed":
                            row["reason"] = "crossed_before_armed"
                        elif path_crossing(previous["center"], center, line) is None:
                            row["reason"] = "outside_finite_line"
                        else:
                            row["reason"] = "crossing_unexplained"
                    else:
                        row["reason"] = "not_crossing"
                reason_counts[row["reason"]] += 1
                near_line.append(row)

            if str(number) in references:
                points = references[str(number)]
                matched = unique_reference_matches(points, boxes.xyxy, polygons)
                for ref_index, point in enumerate(points):
                    prediction = matched.get(ref_index)
                    obs = index_to_observation.get(prediction) if prediction is not None else None
                    reference_rows.append({"frame": number, "reference_index": ref_index + 1,
                                           "point": point, "detected": prediction is not None,
                                           "tracked": obs is not None,
                                           "confidence": round(float(boxes[prediction].conf[0]), 3)
                                           if prediction is not None else None,
                                           "detection_index": prediction,
                                           "tracker_id": obs.tracker_id if obs else None,
                                           "near_line": abs(counter.distance(point, width, height)) <= 35})
                preview = frame.copy()
                cv2.polylines(preview, [np.array(line, np.int32)], False, (55, 145, 255), 2)
                for ref_index, point in enumerate(points):
                    prediction = matched.get(ref_index)
                    color = ((0, 190, 0) if prediction in index_to_observation
                             else (0, 215, 255) if prediction is not None else (0, 0, 255))
                    cv2.circle(preview, tuple(point), 5, color, 2)
                    cv2.putText(preview, str(ref_index + 1), (point[0]+5, point[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
                cv2.imwrite(str(OUT / f"{name}_reference_{number:03d}.jpg"), preview)
            finished = time.perf_counter()
            durations.append((after_inference-started, finished-started))
            frame_rows.append({"frame": number, "detections": predicted,
                               "tracked": len(index_to_observation),
                               "crossings": len(current_crossings),
                               "inference_ms": round(1000*(after_inference-started), 1),
                               "pipeline_ms": round(1000*(finished-started), 1)})
            number += 1
            if number % 50 == 0:
                print(f"{name}: {number} frames, {len(crossings)} crossing events", flush=True)
    finally:
        capture.release()

    frames = len(frame_rows)
    raw_matched = sum(row["detected"] for row in reference_rows)
    track_matched = sum(row["tracked"] for row in reference_rows)
    times = [entry[1] for entry in durations]
    infer_times = [entry[0] for entry in durations]
    report = {
        "model": name, "task": task, "input_size": size,
        "model_path": str(MODELS[name][1]),
        "execution_devices": model.predictor.model.ov_compiled_model.get_property("EXECUTION_DEVICES"),
        "source": str(VIDEO), "decoded_frames": frames,
        "metadata_frames": declared_frames,
        "video_source_fps": 25,
        "line_points": cfg.line_points,
        "settings": {"confidence": .10, "iou": .65, "max_det": 300,
                     "tracker_high": .25, "tracker_low": .10, "new_track": .30,
                     "hysteresis_px": cfg.hysteresis, "arm_frames": cfg.arm_frames,
                     "near_line_band_px": 35},
        "reference_points": len(reference_rows),
        "reference_detected": raw_matched,
        "reference_tracked": track_matched,
        "reference_diagnostic_only": True,
        "raw_detections_all_frames": stages["raw_detections_all_frames"],
        "tracked_detections_all_frames": stages["tracked_detections_all_frames"],
        "raw_detections_near_line": stages["raw_detections_near_line"],
        "tracked_detections_near_line": stages["tracked_detections_near_line"],
        "near_line_reasons": dict(reason_counts),
        "crossing_events": len(crossings),
        "distinct_counted_physical_ids": len({row["physical_id"] for row in crossings}),
        "recovered_ids": counter.recovered, "ambiguous_ids": counter.ambiguous,
        "pipeline_fps": round(frames/sum(times), 2) if times else 0,
        "inference_median_ms": round(statistics.median(infer_times)*1000, 1) if times else 0,
        "pipeline_median_ms": round(statistics.median(times)*1000, 1) if times else 0,
        "pipeline_p95_ms": round(float(np.percentile(times, 95))*1000, 1) if times else 0,
        "accuracy_verified": False,
        "limits": ("Kaggle trained on frames from this same video; " if name.startswith("kaggle") else "")
                  + "Partial point references; no independent physical crossing IDs.",
        "frames": frame_rows, "references": reference_rows,
        "near_line_trace": near_line, "crossings": crossings,
    }
    destination = OUT / (f"{name}_smoke_{limit}.json" if limit is not None
                         else f"{name}_diagnostics.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"frames", "references", "near_line_trace", "crossings"}}, indent=2), flush=True)
    print("Saved", destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Smoke test only; omit for the full 262 decoded frames")
    options = parser.parse_args()
    analyze(options.model, options.max_frames)
