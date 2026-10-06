"""Compare ByteTrack settings on identical Kaggle detections, without live counts.

Run: .venv/Scripts/python.exe -m backend.tune_tracker
These are continuity diagnostics, NOT counting accuracy or ground truth.
Each invocation infers the complete file once, then replays cached detections.
"""
from dataclasses import replace
import json
import time

import cv2
import numpy as np

from backend.main import Config, LineCounter, Observation, OpenVINOModels, ROOT


def main():
    cfg = replace(Config.from_env(), seg_model=ROOT / "models/candidates/kaggle_egg_yolo11n_openvino_model",
                  image_size=640, egg_labels=("egg",), defect_enabled=0)
    model = OpenVINOModels(cfg)
    from types import SimpleNamespace
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from backend.diagnose_pipeline import unique_reference_matches

    references = json.loads((ROOT / "artifacts/model-evaluation/reference_points.json").read_text())["frames"]
    cap = cv2.VideoCapture(str(ROOT / cfg.source))
    if not cap.isOpened():
        raise RuntimeError("Cannot read the configured test video")
    cached, elapsed = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            started = time.perf_counter()
            result = model.seg.predict(frame, imgsz=640, device="intel:cpu", rect=False, verbose=False,
                                       conf=cfg.detection_conf, iou=cfg.detection_iou,
                                       classes=model.egg_classes, max_det=cfg.max_detections)[0]
            elapsed.append(time.perf_counter() - started)
            cached.append((result.boxes.cpu().numpy().data.copy(), frame.shape[:2]))
            if len(cached) % 50 == 0:
                print(f"Inferred {len(cached)} frames", flush=True)
    finally:
        cap.release()

    profiles = {
        "baseline": (.25, .30, .80, True),
        "lower_conf": (.20, .25, .80, True),
        "geometry": (.25, .30, .80, False),
        "lower_conf_geometry": (.20, .25, .80, False),
        "strict_geometry": (.20, .25, .70, False),
        "sensitive_geometry": (.15, .20, .70, False),
    }
    reports = []
    for name, (high, new, match, fuse) in profiles.items():
        tracker = BYTETracker(SimpleNamespace(track_high_thresh=high, track_low_thresh=.10,
            new_track_thresh=new, match_thresh=match, fuse_score=fuse, track_buffer=cfg.track_buffer), frame_rate=30)
        counter = LineCounter(cfg)
        crossings = near_raw = near_tracked = tracked = reference_matches = reference_total = 0
        ids = set()
        for index, (data, (height, width)) in enumerate(cached):
            boxes = Boxes(data.copy(), (height, width))
            tracks = tracker.update(boxes)
            observations = [Observation(int(row[4]), ((float(row[0])+float(row[2]))/2,
                             (float(row[1])+float(row[3]))/2), tuple(map(float, row[:4]))) for row in tracks]
            crossings += len(counter.update(observations, index + 1, width, height))
            ids.update(o.tracker_id for o in observations)
            tracked += len(observations)
            near_raw += sum(abs(counter.distance(((b[0]+b[2])/2, (b[1]+b[3])/2), width, height)) <= 35 for b in data)
            near_tracked += sum(abs(counter.distance(o.center, width, height)) <= 35 for o in observations)
            ref = references.get(str(index))
            if ref:
                points = ref
                reference_total += len(points)
                reference_matches += len(unique_reference_matches(points, [o.box for o in observations], []))
        reports.append(dict(profile=name, high=high, new=new, match=match, fuse_score=fuse,
            frames=len(cached), crossings=crossings, unique_tracker_ids=len(ids), tracked_observations=tracked,
            near_line_detections=int(near_raw), near_line_tracks=int(near_tracked),
            reference_matches=reference_matches, reference_total=reference_total,
            recovered_ids=counter.recovered, ambiguous_ids=counter.ambiguous))
        print(json.dumps(reports[-1]), flush=True)
    out = ROOT / "artifacts/model-evaluation/tracker-tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        "warning": "Same-video diagnostics only; no physical crossing ground truth. Counts are algorithm outputs.",
        "model": str(cfg.seg_model), "conf": cfg.detection_conf, "iou": cfg.detection_iou,
        "median_inference_ms": float(np.median(elapsed)*1000), "profiles": reports}, indent=2))


if __name__ == "__main__":
    main()
