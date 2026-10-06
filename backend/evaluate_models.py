"""Offline CPU experiment: python -m backend.evaluate_models.

Reference-point coverage is a partial visual audit, NOT mAP/count accuracy.
This experiment cannot select or enable a production model.
"""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data/ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/ultralytics"))

import cv2
import numpy as np
import openvino as ov
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from backend.main import Config, LineCounter, OpenVINOModels, Observation


def match_references(points, boxes, polygons):
    """One prediction may cover only one reference egg; merged eggs fail.

    Unlabelled eggs are not negatives, so precision cannot be reported.
    """
    candidates = []
    for point in points:
        candidates.append([i for i, (x1, y1, x2, y2) in enumerate(boxes)
                           if x1 <= point[0] <= x2 and y1 <= point[1] <= y2
                           and (not polygons or cv2.pointPolygonTest(
                               polygons[i].astype(np.float32), tuple(point), False) >= 0)])
    assigned = {}
    def assign(reference, visited):
        for prediction in candidates[reference]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in assigned or assign(assigned[prediction], visited):
                assigned[prediction] = reference
                return True
        return False
    for reference in range(len(points)):
        assign(reference, set())
    return set(assigned.values())


def evaluate_samples(label, size, confidence):
    """Reproduce the partial CPU spot-check for any downloaded candidate."""
    out = ROOT / "artifacts/model-evaluation"
    references = json.loads((out / "reference_points.json").read_text())["frames"]
    model = YOLO(str(ROOT / f"models/candidates/{label}.pt"))
    audit = []
    for number, points in references.items():
        frame = cv2.imread(str(out / f"raw_{int(number):03d}.jpg"))
        result = model.predict(frame, imgsz=size, device="cpu", conf=confidence,
            iou=.65, classes=[1] if label == "poultryvision" else None,
            verbose=False, rect=False)[0]
        polygons = result.masks.xy if result.masks is not None else []
        matched = match_references(points, result.boxes.xyxy.cpu().numpy(), polygons)
        audit.append({"frame": int(number), "references": len(points),
                      "matched": len(matched), "detections": len(result.boxes)})
    report = {"candidate": label, "size": size, "confidence": confidence,
              "backend": "PyTorch CPU", "audit": audit,
              "reference_points": sum(len(p) for p in references.values()),
              "matched_references": sum(a["matched"] for a in audit), "approved": False}
    (out / f"{label}_reference_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def evaluate(label, size, confidence, band=False):
    out = ROOT / "artifacts/model-evaluation"
    references = json.loads((out / "reference_points.json").read_text())["frames"]
    cfg = Config(axis="x", direction=1, line_position=.55, image_size=size, cpu_threads=2)
    adapter = OpenVINOModels.__new__(OpenVINOModels)
    adapter.cfg = cfg
    model = adapter.load(YOLO, ov, ROOT / f"models/candidates/{label}_{size}_openvino_model", "segment", size)
    adapter.seg = model
    adapter.egg_classes = list(model.names)  # Both white and brown are eggs.
    adapter.tracker = BYTETracker(SimpleNamespace(track_high_thresh=.25, track_low_thresh=.10,
        new_track_thresh=.30, track_buffer=45, match_thresh=.80, fuse_score=True), frame_rate=30)
    counter = LineCounter(cfg)
    capture = cv2.VideoCapture(str(ROOT / "Eggs On A Conveyor Stock Video.mp4"))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    name = f"{label}_{size}_c{confidence:g}" + ("_band" if band else "")
    times, events, rows, audit = [], [], [], []
    number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        offset = 260 if band else 0
        start = time.perf_counter()
        result = model.predict(frame[:, 260:680] if band else frame, imgsz=size,
            device="intel:cpu", conf=confidence, iou=.65, verbose=False, rect=False)[0]
        boxes = result.boxes.cpu().numpy()
        if offset:
            boxes.data[:, [0, 2]] += offset
        polygons = [p + np.array([offset, 0]) for p in result.masks.xy] if result.masks is not None else []
        tracks = adapter.tracker.update(boxes, frame)
        observations = []
        for row in tracks:
            index = int(row[-1])
            contour = polygons[index] if index < len(polygons) else None
            center = ((row[0]+row[2])/2, (row[1]+row[3])/2)
            if contour is not None:
                m = cv2.moments(contour.astype(np.float32))
                if m["m00"]:
                    center = (m["m10"]/m["m00"], m["m01"]/m["m00"])
            observations.append(Observation(int(row[4]), center, tuple(map(float, row[:4])), contour))
        for state, obs, point in counter.update(observations, number+1, w, h):
            events.append({"id": state.physical_id, "tracker_id": obs.tracker_id,
                "frame": number, "media_seconds": number/25, "x": float(point[0]), "y": float(point[1])})
        times.append(time.perf_counter()-start)
        rows.append({"frame": number, "detections": len(boxes), "tracks": len(tracks)})
        if str(number) in references:
            points = references[str(number)]
            matched = match_references(points, boxes.xyxy, polygons)
            audit.append({"frame": number, "references": len(points), "matched": len(matched),
                "missed": [p for i, p in enumerate(points) if i not in matched]})
            preview = frame.copy()
            for obs in observations:
                cv2.rectangle(preview, tuple(map(int, obs.box[:2])), tuple(map(int, obs.box[2:])), (180, 180, 0), 1)
            cv2.line(preview, (round(w*.55), 0), (round(w*.55), h), (255, 255, 0), 2)
            for i, p in enumerate(points):
                color = (0, 220, 0) if i in matched else (0, 0, 255)
                cv2.circle(preview, tuple(p), 5, color, 2)
                cv2.putText(preview, str(i+1), (p[0]+6, p[1]), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
            cv2.imwrite(str(out/f"{name}_audit_{number:03d}.jpg"), preview)
        number += 1
        if number % 50 == 0:
            print(name, number, "frames;", len(events), "candidate crossings", flush=True)
    capture.release()
    total = sum(a["references"] for a in audit)
    matched = sum(a["matched"] for a in audit)
    report = {"candidate": label, "size": size, "confidence": confidence, "band": band,
        "frames_processed": number, "declared_frames": declared_frames,
        "decode_matches_metadata": number == declared_frames, "source_fps": 25, "cpu_threads": 2,
        "execution_devices": model.predictor.model.ov_compiled_model.get_property("EXECUTION_DEVICES"),
        "processing_fps": len(times)/sum(times), "median_frame_ms": float(np.median(times)*1000),
        "p95_frame_ms": float(np.percentile(times, 95)*1000),
        "reference_points": total, "matched_references": matched, "reference_coverage_percent": 100*matched/total,
        "candidate_crossings": len(events), "recovered_ids": counter.recovered, "ambiguous_ids": counter.ambiguous,
        "approved": False, "reason": "Partial references and unresolved identities cannot certify counting accuracy. No independent full-clip crossing ledger.",
        "audit": audit, "events": events, "frames": rows}
    (out/f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"audit", "events", "frames"}}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["egg_segmentor", "eggs_counting", "poultryvision"], default="egg_segmentor")
    parser.add_argument("--size", type=int, choices=[416, 640], default=640)
    parser.add_argument("--conf", type=float, default=.10)
    parser.add_argument("--band", action="store_true")
    parser.add_argument("--samples-only", action="store_true", help="Partial PyTorch CPU reference audit")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if args.samples_only:
        evaluate_samples(args.candidate, args.size, args.conf)
    elif args.candidate == "poultryvision":
        parser.error("PoultryVision has no segmentation export here; use --samples-only")
    else:
        evaluate(args.candidate, args.size, args.conf, args.band)
