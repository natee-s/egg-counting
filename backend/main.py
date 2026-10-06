"""Egg counting, OpenVINO inference, ByteTrack, and a native JSON WebSocket.

Run from the project root: python -m uvicorn backend.main:app --host 127.0.0.1
The single CV worker owns capture, tracking and SQLite writes. FastAPI's event
loop only handles clients: a slow browser must never stall the conveyor worker.
Demo is a labelled simulation; real mode refuses missing/untrained class labels.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / "backend" / ".env")
LOG = logging.getLogger("egg-counter")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    mode: str = "demo"
    source: str = "0"
    seg_model: Path = ROOT / "models/egg_seg_openvino_model"
    cls_model: Path = ROOT / "models/egg_cls_openvino_model"
    db: Path = ROOT / "data/events.sqlite3"
    image_size: int = 416
    cls_image_size: int = 224
    cpu_threads: int = 2
    detection_conf: float = 0.10
    detection_iou: float = 0.65
    max_detections: int = 300
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.30
    match_thresh: float = 0.80
    fuse_score: int = 1
    axis: str = "y"
    line_position: float = 0.55
    line_points: tuple[tuple[float, float], ...] = ()
    direction: int = 1
    hysteresis: float = 10
    arm_frames: int = 2
    track_buffer: int = 45
    recover_frames: int = 12
    recover_distance: float = 24
    max_cross_gap: int = 45
    stream_fps: float = 10
    jpeg_quality: int = 70
    max_width: int = 960
    egg_label: str = "egg"
    egg_labels: tuple[str, ...] = ()
    defect_enabled: int = 1
    start_paused: int = 1
    model_name: str = "Custom egg model"
    good_labels: tuple[str, ...] = ("good", "normal")
    defect_labels: tuple[str, ...] = ("defect", "cracked", "crack", "dirty", "dirt")
    origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    @classmethod
    def from_env(cls) -> "Config":
        mapping = {
            "mode": "MODE", "source": "SOURCE", "seg_model": "SEG_MODEL",
            "cls_model": "CLS_MODEL", "db": "DB", "image_size": "IMAGE_SIZE",
            "cls_image_size": "CLS_IMAGE_SIZE", "cpu_threads": "CPU_THREADS",
            "detection_conf": "DETECTION_CONF", "detection_iou": "DETECTION_IOU",
            "max_detections": "MAX_DETECTIONS", "track_high_thresh": "TRACK_HIGH_THRESH",
            "track_low_thresh": "TRACK_LOW_THRESH", "new_track_thresh": "NEW_TRACK_THRESH",
            "match_thresh": "MATCH_THRESH", "fuse_score": "FUSE_SCORE",
            "axis": "LINE_AXIS", "line_position": "LINE_POSITION", "direction": "DIRECTION",
            "line_points": "LINE_POINTS", "egg_labels": "EGG_LABELS",
            "defect_enabled": "DEFECT_ENABLED", "start_paused": "START_PAUSED", "model_name": "MODEL_NAME",
            "hysteresis": "HYSTERESIS_PX", "arm_frames": "ARM_FRAMES",
            "track_buffer": "TRACK_BUFFER", "recover_frames": "RECOVER_FRAMES",
            "recover_distance": "RECOVER_DISTANCE_PX", "max_cross_gap": "MAX_CROSS_GAP",
            "stream_fps": "STREAM_FPS", "jpeg_quality": "JPEG_QUALITY", "max_width": "MAX_WIDTH",
            "egg_label": "EGG_LABEL", "good_labels": "GOOD_LABELS",
            "defect_labels": "DEFECT_LABELS", "origins": "ORIGINS",
        }
        default = cls()
        kwargs = {}
        for name, key in mapping.items():
            raw = os.getenv("EGG_" + key)
            if raw is None:
                continue
            value = getattr(default, name)
            if name == "line_points":
                kwargs[name] = tuple(tuple(float(v) for v in point.split(":")) for point in raw.split(",") if point.strip())
            elif isinstance(value, Path):
                path = Path(raw)
                kwargs[name] = path if path.is_absolute() else ROOT / path
            elif isinstance(value, tuple):
                kwargs[name] = tuple(s.strip() for s in raw.split(",") if s.strip())
            else:
                kwargs[name] = type(value)(raw)
        cfg = cls(**kwargs)
        if cfg.mode not in {"demo", "real", "video"} or cfg.axis not in {"x", "y"} or cfg.direction not in {-1, 1}:
            raise ValueError("Invalid mode, line axis, or direction")
        if not 0 < cfg.line_position < 1 or cfg.hysteresis < 0 or cfg.arm_frames < 1:
            raise ValueError("Invalid line position / arming configuration")
        if min(cfg.stream_fps, cfg.cpu_threads, cfg.max_width, cfg.track_buffer, cfg.max_cross_gap) <= 0:
            raise ValueError("FPS, threads, frame dimensions and buffers must be positive")
        if cfg.recover_frames < 0 or cfg.recover_distance < 0 or not 1 <= cfg.jpeg_quality <= 100:
            raise ValueError("Invalid recovery / JPEG configuration")
        if any(size < 32 or size % 32 for size in (cfg.image_size, cfg.cls_image_size)):
            raise ValueError("Inference image sizes must be positive multiples of 32")
        if set(cfg.good_labels) & set(cfg.defect_labels):
            raise ValueError("Good and defect labels must be disjoint")
        if cfg.defect_enabled not in {0, 1} or cfg.start_paused not in {0, 1}:
            raise ValueError("DEFECT_ENABLED and START_PAUSED must be 0 or 1")
        if not (0 <= cfg.detection_conf <= cfg.track_low_thresh < cfg.track_high_thresh
                <= cfg.new_track_thresh <= 1):
            raise ValueError("Require 0 <= DETECTION_CONF <= TRACK_LOW_THRESH < TRACK_HIGH_THRESH <= NEW_TRACK_THRESH <= 1")
        if not (0 < cfg.detection_iou <= 1 and 0 < cfg.match_thresh <= 1
                and cfg.max_detections > 0 and cfg.fuse_score in {0, 1}):
            raise ValueError("Invalid detection IoU, max detections or ByteTrack matching settings")
        if cfg.line_points:
            if len(cfg.line_points) < 2 or any(len(p) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in cfg.line_points):
                raise ValueError("LINE_POINTS needs at least two normalized x:y points")
            if any(a == b for a, b in zip(cfg.line_points, cfg.line_points[1:])):
                raise ValueError("Consecutive line points must be distinct")
        return cfg


def normalized_line(cfg: Config):
    """One geometry shared by counting, backend annotation and the browser."""
    if cfg.line_points:
        return cfg.line_points
    return ((cfg.line_position, 0), (cfg.line_position, 1)) if cfg.axis == "x" else ((0, cfg.line_position), (1, cfg.line_position))


def pixel_line(cfg: Config, width: int, height: int):
    return [(x * width, y * height) for x, y in normalized_line(cfg)]


def path_crossing(start, end, points):
    """Intersection with FINITE segments, not the infinite line extension.

    Return the first intersection along the egg's movement. This also supplies
    an exact particle position at corners of the user-defined polyline.
    """
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    movement = (end[0] - start[0], end[1] - start[1])
    hits = []
    for a, b in zip(points, points[1:]):
        segment = (b[0] - a[0], b[1] - a[1])
        denominator = cross(movement, segment)
        if abs(denominator) < 1e-9:
            continue
        relative = (a[0] - start[0], a[1] - start[1])
        t, u = cross(relative, segment) / denominator, cross(relative, movement) / denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            hits.append((t, (start[0] + t * movement[0], start[1] + t * movement[1])))
    return min(hits, key=lambda item: item[0])[1] if hits else None


class Phase(str, Enum):
    UNARMED = "unarmed"
    ARMED = "armed"
    COUNTED = "counted"  # Terminal even if the egg rolls back over the line.


@dataclass
class Observation:
    tracker_id: int
    center: tuple[float, float]
    box: tuple[float, float, float, float]
    contour: np.ndarray | None = None


@dataclass
class EggState:
    physical_id: int
    tracker_id: int
    center: tuple[float, float]
    box: tuple[float, float, float, float]
    last_frame: int
    distance: float
    velocity: tuple[float, float] = (0, 0)
    phase: Phase = Phase.UNARMED
    upstream_frames: int = 0
    status: str | None = None


class LineCounter:
    """Directed crossing state machine plus conservative short ID recovery.

    A new track must be observed beyond the upstream hysteresis band for N
    frames. Only ARMED tracks can cross (previous distance < 0, current >= 0).
    COUNTED never re-arms. No cooldown: adjacent eggs may count in one frame.
    Recovery transfers the terminal state, rather than inventing a new identity.
    Motion is not appearance ReID; ambiguous matches are exposed for review.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.states: dict[int, EggState] = {}
        self.aliases: dict[int, int] = {}
        self.next_id = 1
        self.recovered = 0
        self.ambiguous = 0
        self.long_gaps = 0

    def distance(self, center, width: int, height: int) -> float:
        if self.cfg.line_points:
            # Signed perpendicular distance to the nearest finite segment.
            # Top -> bottom points give a positive right-facing normal.
            nearest = None
            for a, b in zip(pixel_line(self.cfg, width, height), pixel_line(self.cfg, width, height)[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                length = math.hypot(dx, dy)
                fraction = max(0, min(1, ((center[0]-a[0])*dx + (center[1]-a[1])*dy) / length**2))
                projection = (a[0]+fraction*dx, a[1]+fraction*dy)
                error = math.dist(center, projection)
                signed = ((center[0]-a[0])*dy - (center[1]-a[1])*dx) / length
                if nearest is None or error < nearest[0]:
                    nearest = (error, signed)
            return self.cfg.direction * nearest[1]
        coordinate = center[0] if self.cfg.axis == "x" else center[1]
        dimension = width if self.cfg.axis == "x" else height
        return self.cfg.direction * (coordinate - dimension * self.cfg.line_position)

    def update(self, observations: list[Observation], frame: int, width: int, height: int):
        present = {o.tracker_id for o in observations}
        new = [o for o in observations if o.tracker_id not in self.aliases]
        # Never merge into a physical identity whose tracker is visible this frame.
        missing = [s for s in self.states.values() if s.tracker_id not in present
                   and 0 < frame - s.last_frame <= self.cfg.recover_frames]
        candidates: dict[int, list[int]] = {}
        reverse: dict[int, list[int]] = {}
        for obs in new:
            for state in missing:
                gap = frame - state.last_frame
                predicted = tuple(state.center[i] + state.velocity[i] * gap for i in (0, 1))
                error = math.dist(obs.center, predicted)
                old_size = (state.box[2] - state.box[0], state.box[3] - state.box[1])
                size = (obs.box[2] - obs.box[0], obs.box[3] - obs.box[1])
                similar = all(0.65 <= size[i] / max(1, old_size[i]) <= 1.55 for i in (0, 1))
                if error <= self.cfg.recover_distance and similar:
                    candidates.setdefault(obs.tracker_id, []).append(state.physical_id)
                    reverse.setdefault(state.physical_id, []).append(obs.tracker_id)
        for obs in new:
            options = candidates.get(obs.tracker_id, [])
            if len(options) == 1 and len(reverse[options[0]]) == 1:
                state = self.states[options[0]]
                # Remove the old alias so two later tracks cannot share this state.
                self.aliases.pop(state.tracker_id, None)
                state.tracker_id = obs.tracker_id
                self.aliases[obs.tracker_id] = state.physical_id
                self.recovered += 1
            else:
                if options:
                    self.ambiguous += 1
                distance = self.distance(obs.center, width, height)
                state = EggState(self.next_id, obs.tracker_id, obs.center, obs.box, frame, distance)
                self.states[self.next_id] = state
                self.aliases[obs.tracker_id] = self.next_id
                self.next_id += 1

        crossings = []
        for obs in observations:
            state = self.states[self.aliases[obs.tracker_id]]
            distance = self.distance(obs.center, width, height)
            gap = frame - state.last_frame
            if gap > self.cfg.max_cross_gap and state.phase != Phase.COUNTED:
                state.phase, state.upstream_frames = Phase.UNARMED, 0
                self.long_gaps += 1
            if state.phase == Phase.UNARMED:
                state.upstream_frames = state.upstream_frames + 1 if distance <= -self.cfg.hysteresis else 0
                if state.upstream_frames >= self.cfg.arm_frames:
                    state.phase = Phase.ARMED
            elif state.phase == Phase.ARMED and state.distance < 0 <= distance:
                crossing = path_crossing(state.center, obs.center, pixel_line(self.cfg, width, height))
                # Crossing beyond either endpoint must not increment a count.
                if crossing is not None:
                # Mark terminal BEFORE classification, so repeated observations
                # or retries cannot cause multiple classifier calls/count events.
                    state.phase = Phase.COUNTED
                    crossings.append((state, obs, crossing))
            if gap > 0:
                state.velocity = tuple((obs.center[i] - state.center[i]) / gap for i in (0, 1))
            state.center, state.box = obs.center, obs.box
            state.distance, state.last_frame = distance, frame
        # Retain terminal states while the same track might return; SQLite keeps
        # the permanent event ledger. Tracker identities are scoped to a run.
        retention = max(self.cfg.track_buffer, self.cfg.max_cross_gap, self.cfg.recover_frames) * 4
        for physical_id, state in list(self.states.items()):
            if frame - state.last_frame > retention:
                self.aliases.pop(state.tracker_id, None)
                del self.states[physical_id]
        return crossings


class EventStore:
    """Idempotent, durable counts. A fresh run_id prevents restart ID collisions.

    Counts are session-specific; restarting capture starts a new run because
    persistent totals cannot prove the identity of eggs still on the conveyor.
    Historical runs are retained and exportable through /api/events.
    """
    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id = path, run_id
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL, physical_id INTEGER NOT NULL, payload TEXT NOT NULL,
                UNIQUE(run_id, physical_id))""")
        self.total = self.normal = self.defect = self.unclassified = 0

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def record(self, event: dict) -> dict | None:
        with self.connect() as db:
            cursor = db.execute("INSERT OR IGNORE INTO events(run_id,physical_id,payload) VALUES(?,?,?)",
                                (self.run_id, event["id"], json.dumps(event)))
            if not cursor.rowcount:
                return None
            self.total += 1
            self.normal += int(event["status"] == "good")
            self.defect += int(event["status"] == "defect")
            self.unclassified += int(event["status"] == "unknown")
            event = {**event, "sequence": cursor.lastrowid, "total_count": self.total,
                     "run_id": self.run_id}
            db.execute("UPDATE events SET payload=? WHERE sequence=?", (json.dumps(event), cursor.lastrowid))
        return event

    def after(self, cursor: int, limit: int = 500, run_id: str | None = None):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?",
                              (run_id or self.run_id, cursor, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]


class OpenVINOModels:
    """Pinned Ultralytics preprocessing/postprocessing; strictly CPU compilation.

    Ultralytics' default device=cpu may compile OpenVINO for AUTO. Explicit
    intel:cpu and a verified CPU-only recompile prevent unintended GPU selection.
    Batch=1, LATENCY, one stream, bounded inference threads suit small machines.
    """
    def __init__(self, cfg: Config):
        # Report missing models immediately, before loading heavyweight libraries.
        directories = (cfg.seg_model, cfg.cls_model) if cfg.defect_enabled else (cfg.seg_model,)
        for directory in directories:
            xmls = list(directory.glob("*.xml"))
            if len(xmls) != 1 or not xmls[0].with_suffix(".bin").exists() or not (directory / "metadata.yaml").exists():
                raise FileNotFoundError(f"Need one XML + BIN + metadata.yaml in {directory}; run backend/export_models.py")
        settings = ROOT / "data/ultralytics"
        settings.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
        from ultralytics import YOLO
        import openvino as ov
        import torch
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.cfg = cfg
        torch.set_num_threads(cfg.cpu_threads)
        cv2.setNumThreads(1)
        # Accept a trained detection OR segmentation export. Never force a
        # detector through the segmentation postprocessor (different outputs).
        from ultralytics.utils import YAML
        self.task = YAML.load(cfg.seg_model / "metadata.yaml").get("task")
        if self.task not in {"detect", "segment"}:
            raise ValueError(f"Expected detect or segment egg model, got {self.task}")
        self.seg = self.load(YOLO, ov, cfg.seg_model, self.task, cfg.image_size)
        self.cls = self.load(YOLO, ov, cfg.cls_model, "classify", cfg.cls_image_size) if cfg.defect_enabled else None
        names = {int(k): str(v).lower() for k, v in self.seg.names.items()}
        accepted = {label.lower() for label in (cfg.egg_labels or (cfg.egg_label,))}
        self.egg_classes = [i for i, name in names.items() if name in accepted]
        if not self.egg_classes:
            raise ValueError(f"Egg model must include {accepted}, got {names}")
        if self.cls is not None:
            labels = {str(v).lower() for v in self.cls.names.values()}
            if not labels & set(cfg.good_labels) or not labels & set(cfg.defect_labels):
                raise ValueError(f"Classifier must contain good AND defect labels, got {labels}")
            if labels - set(cfg.good_labels) - set(cfg.defect_labels):
                raise ValueError(f"Unmapped classifier labels: {labels}")
        args = SimpleNamespace(track_high_thresh=cfg.track_high_thresh, track_low_thresh=cfg.track_low_thresh,
                               new_track_thresh=cfg.new_track_thresh, track_buffer=cfg.track_buffer,
                               match_thresh=cfg.match_thresh, fuse_score=bool(cfg.fuse_score))
        # frame_rate=30 means track_buffer is exactly the number of processed
        # frames, independent of variable CPU throughput or source frame rate.
        self.tracker = BYTETracker(args, frame_rate=30)
        self.last_detection_count = 0

    def load(self, YOLO, ov, directory: Path, task: str, size: int):
        xmls = list(directory.glob("*.xml"))
        if len(xmls) != 1 or not xmls[0].with_suffix(".bin").exists() or not (directory / "metadata.yaml").exists():
            raise FileNotFoundError(f"Need one XML + BIN + metadata.yaml in {directory}; run backend/export_models.py")
        core = ov.Core()
        graph = core.read_model(str(xmls[0]))
        shape = graph.input(0).partial_shape
        if shape.is_dynamic or list(shape.to_shape()) != [1, 3, size, size]:
            raise ValueError(f"Expected fixed model input [1,3,{size},{size}], got {shape}; match .env to export sizes")
        from ultralytics.utils import YAML
        metadata = YAML.load(directory / "metadata.yaml")
        if metadata.get("task") != task:
            raise ValueError(f"Expected {task} model in {directory}, got {metadata.get('task')}")
        model = YOLO(str(directory), task=task)
        # Initialize the predictor with a blank image. This is startup warmup,
        # not a per-frame egg classification and does not produce count events.
        model.predict(np.zeros((size, size, 3), np.uint8), imgsz=size,
                      device="intel:cpu", verbose=False, rect=False)
        backend = model.predictor.model
        graph.get_parameters()[0].set_layout(ov.Layout("NCHW"))
        compiled = core.compile_model(graph, "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1",
            "INFERENCE_NUM_THREADS": self.cfg.cpu_threads,
            "CACHE_DIR": str(ROOT / "data/ov_cache"),
        })
        # Deliberately version-pinned adapter to retain Ultralytics' mask/NMS
        # implementation while controlling OpenVINO CPU scheduling ourselves.
        backend.ov_compiled_model = compiled
        backend.inference_mode = "LATENCY"
        if any(not device.startswith("CPU") for device in compiled.get_property("EXECUTION_DEVICES")):
            raise RuntimeError("CPU-only invariant violated")
        return model

    def detect(self, frame: np.ndarray) -> list[Observation]:
        result = self.seg.predict(frame, imgsz=self.cfg.image_size, device="intel:cpu",
                                  conf=self.cfg.detection_conf, iou=self.cfg.detection_iou,
                                  max_det=self.cfg.max_detections, classes=self.egg_classes,
                                  verbose=False, rect=False)[0]
        self.last_detection_count = len(result.boxes)
        # update even with no detections: ByteTrack must age occluded tracks.
        tracks = self.tracker.update(result.boxes.cpu().numpy(), frame)
        observations = []
        polygons = result.masks.xy if result.masks is not None else []
        for row in tracks:
            # Last column is the ORIGINAL detection index, critical for matching
            # the right instance mask after ByteTrack reorders detections.
            x1, y1, x2, y2, track_id = row[:5]
            idx = int(row[-1])
            contour = polygons[idx] if idx < len(polygons) else None
            center = ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)
            if contour is not None and len(contour) >= 3:
                moments = cv2.moments(contour.astype(np.float32))
                if moments["m00"]:
                    center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            observations.append(Observation(int(track_id), center,
                                           tuple(float(v) for v in row[:4]), contour))
        return observations

    def classify(self, crop: np.ndarray) -> tuple[str, str, float]:
        if self.cls is None:
            raise RuntimeError("Surface inspection is disabled for this counting-only model")
        result = self.cls.predict(crop, imgsz=self.cfg.cls_image_size,
                                  device="intel:cpu", verbose=False)[0]
        label = str(result.names[result.probs.top1]).lower()
        status = "good" if label in self.cfg.good_labels else "defect"
        return status, label, float(result.probs.top1conf)


class DemoConveyor:
    """Deterministic labelled synthetic feed; never passed off as model inference."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.width, self.height = 960, 540

    def read(self, frame_number: int):
        canvas = np.full((self.height, self.width, 3), (23, 28, 33), np.uint8)
        for y in range(-40, self.height + 40, 60):
            offset = int(frame_number * 3) % 60
            cv2.line(canvas, (0, y + offset), (self.width, y + offset), (46, 55, 63), 8)
            cv2.line(canvas, (0, y + offset - 5), (self.width, y + offset - 5), (66, 72, 78), 1)
        observations = []
        vertical = self.cfg.axis == "y"
        span = self.height if vertical else self.width
        travel_frames = math.ceil((span + 100) / 4)
        for lane in range(5):
            for batch in range(max(0, frame_number // 42 - travel_frames // 42 - 2), frame_number // 42 + 1):
                age = frame_number - batch * 42 - lane * 6
                coordinate = age * 4 - 50
                if not -50 <= coordinate <= span + 50:
                    continue
                if self.cfg.direction == -1:
                    coordinate = span - coordinate
                cross = (lane + 0.5) * ((self.width if vertical else self.height) / 5)
                x, y = (cross, coordinate) if vertical else (coordinate, cross)
                identity = batch * 5 + lane + 1
                color = (182, 203, 216) if identity % 11 == 0 else (227, 235, 239)
                cv2.ellipse(canvas, (int(x), int(y)), (26, 35), -12, 0, 360, color, -1, cv2.LINE_AA)
                cv2.ellipse(canvas, (int(x - 7), int(y - 9)), (9, 16), -12, 0, 360, (244, 248, 250), -1)
                if identity % 11 == 0:
                    cv2.line(canvas, (int(x - 9), int(y - 12)), (int(x + 8), int(y + 8)), (54, 57, 72), 2)
                observations.append(Observation(identity, (x, y), (x - 27, y - 36, x + 27, y + 36)))
        return canvas, observations


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.run_id = uuid.uuid4().hex
        self.store = EventStore(cfg.db, self.run_id)
        self.counter = LineCounter(cfg)
        self.started = time.monotonic()
        self.frame_number = 0
        self.fps = 0.0
        self.epm = deque()
        self.latest_sequence = 0
        self.classifier_calls = 0
        self.classifier_errors = 0
        self.capture = None
        self.models = None
        self.state = "starting"
        self.error = None
        self.source_fps = 0.0
        self.media_time = 0.0
        self.width, self.height = 960, 540
        self.video = None
        self.preview_frame_data = None
        self.tracked_count = 0
        self.detection_count = 0
        self.play_event = threading.Event()
        self.pause_ack = threading.Event()
        if cfg.mode != "real" or not cfg.start_paused:
            self.play_event.set()

    def snapshot(self) -> dict:
        now = time.monotonic()
        while self.epm and self.epm[0] < now - 60:
            self.epm.popleft()
        return {
            "type": "frame_update", "event": "frame_update", "run_id": self.run_id,
            "mode": self.cfg.mode, "state": self.state, "error": self.error,
            "timestamp": utc_now(), "sequence": self.latest_sequence,
            "frame_data": self.preview_frame_data, "events": [], "video": self.video,
            "frame_size": {"width": self.width, "height": self.height},
            "line": {"axis": self.cfg.axis, "position": self.cfg.line_position, "direction": self.cfg.direction,
                     "points": normalized_line(self.cfg), "custom": bool(self.cfg.line_points)},
            "inspection_enabled": self.cfg.mode == "demo" or (self.cfg.mode == "real" and bool(self.cfg.defect_enabled)),
            "model": {"name": self.cfg.model_name, "device": "CPU", "ready": self.models is not None},
            "stats": {"fps": round(self.fps, 1), "total_count": self.store.total,
                      "detection_count": self.detection_count, "tracked_count": self.tracked_count,
                      "processed_frames": self.frame_number,
                      "normal_count": self.store.normal, "defect_count": self.store.defect,
                      "unclassified_count": self.store.unclassified, "eggs_per_minute": len(self.epm),
                      "elapsed_seconds": round(now - self.started, 1), "source_fps": round(self.source_fps, 1),
                      "media_seconds": round(self.media_time, 2),
                      "classifier_calls": self.classifier_calls, "classifier_errors": self.classifier_errors,
                      "recovered_ids": self.counter.recovered, "ambiguous_ids": self.counter.ambiguous,
                      "long_gaps": self.counter.long_gaps},
        }

    def process(self, frame: np.ndarray, observations: list[Observation], classifier: Callable | None) -> list[dict]:
        self.height, self.width = frame.shape[:2]
        self.tracked_count = len(observations)
        self.detection_count = getattr(self.models, "last_detection_count", len(observations))
        events = []
        # Use the RAW frame for classification, before masks/text/laser overlays.
        for state, obs, crossing in self.counter.update(observations, self.frame_number, self.width, self.height):
            x1, y1, x2, y2 = obs.box
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(self.width, math.ceil(x2)), min(self.height, math.ceil(y2))
            crop = frame[top:bottom, left:right].copy()
            if classifier is None:
                # Count-only operation is explicit, not a classifier failure.
                # White/Brown-Egg labels carry NO evidence of surface quality.
                status, label, confidence = "unknown", "not_inspected", None
            else:
                try:
                    if not crop.size:
                        raise ValueError("Empty crossing crop")
                    self.classifier_calls += 1
                    status, label, confidence = classifier(crop, obs.tracker_id)
                    if status not in {"good", "defect"}:
                        raise ValueError("Classifier returned invalid status")
                except Exception:
                # A classification failure must neither lose a physical count
                # nor falsely label the egg good. Unknown is a protocol extension.
                    LOG.exception("Classification failed for egg %s", state.physical_id)
                    self.classifier_errors += 1
                    status, label, confidence = "unknown", "classification_failed", 0.0
            state.status = status
            event = self.store.record({
                "id": state.physical_id, "tracker_id": obs.tracker_id, "type": "count_triggered",
                "status": status, "legacy_status": "normal" if status == "good" else status,
                "label": label, "confidence": confidence, "x": round(crossing[0], 1), "y": round(crossing[1], 1),
                "fps": round(self.fps, 1), "timestamp": utc_now(), "frame_number": self.frame_number,
                "media_seconds": round(self.media_time, 3),
            })
            if event:
                self.latest_sequence = event["sequence"]
                self.epm.append(time.monotonic())
                events.append(event)
        return events

    def annotate(self, frame: np.ndarray, observations: list[Observation]):
        for obs in observations:
            state = self.counter.states.get(self.counter.aliases.get(obs.tracker_id))
            # Red = not yet counted / tracking, gold = counted.
            color = (0, 215, 255) if state and state.phase == Phase.COUNTED else (0, 0, 255)
            x1, y1, x2, y2 = map(int, obs.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            text = f"ID:{state.physical_id if state else obs.tracker_id}"
            cv2.putText(frame, text, (max(0, x1), max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .28, color, 1, cv2.LINE_AA)
            # Exactly the same centre used by LineCounter, not a decorative dot.
            cv2.circle(frame, tuple(round(v) for v in obs.center), 2, (0, 0, 255), -1, cv2.LINE_AA)
        # Laser rendering is inexpensive; the dashboard adds its CSS glow.
        points = np.array(pixel_line(self.cfg, self.width, self.height), np.int32)
        color = (0, 0, 255) if self.cfg.line_points else (220, 220, 40)
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)

    def prepare_video(self):
        """Read only file metadata and a poster. Playback belongs to the browser.

        This explicit preview mode has no detector, tracker, classifier or
        synthetic eggs. Counts remain empty until trained models are available.
        Native MP4 playback saves the CPU cost of JPEG re-encoding every frame.
        """
        path = Path(self.cfg.source)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise FileNotFoundError(f"Video preview requires a local MP4: {path}")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot read video: {path.name}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, poster = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode the first video frame: {path.name}")
            self.height, self.width = poster.shape[:2]
            self.source_fps = fps
            self.video = {"name": path.name, "fps": fps, "frame_count": count,
                          "duration_seconds": count / fps if fps > 0 else 0,
                          "url": "/api/video", "counting_enabled": False}
            encoded_ok, encoded = cv2.imencode(".jpg", poster, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
            if encoded_ok:
                self.preview_frame_data = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
            self.state = "preview"
        finally:
            capture.release()

    def run(self, stop: threading.Event, publish: Callable[[dict], None]):
        pending_events = []
        last_publish = 0.0
        try:
            if self.cfg.mode == "video":
                self.prepare_video()
                return  # finally publishes a paused-video snapshot, with zero events.
            if self.cfg.mode == "real":
                self.models = OpenVINOModels(self.cfg)
                source = int(self.cfg.source) if self.cfg.source.isdecimal() else self.cfg.source
                if isinstance(source, str) and "://" not in source:
                    source = str((ROOT / source).resolve()) if not Path(source).is_absolute() else source
                    if not Path(source).is_file():
                        raise FileNotFoundError(f"Video not found: {source}")
                self.capture = cv2.VideoCapture(source)
                if not self.capture.isOpened():
                    raise RuntimeError(f"Cannot open source {source}")
                self.source_fps = self.capture.get(cv2.CAP_PROP_FPS) or 0
                is_file = isinstance(source, str) and "://" not in source
                ok, first_frame = self.capture.read()
                if not ok:
                    raise RuntimeError("Cannot decode the first source frame")
                self.height, self.width = first_frame.shape[:2]
                if is_file:
                    count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video = {"name": Path(source).name, "fps": self.source_fps,
                                  "frame_count": count, "duration_seconds": count / self.source_fps if self.source_fps > 0 else 0,
                                  "counting_enabled": True}
                # A poster is safe before Play: no detector/tracker/count runs.
                ok, encoded = cv2.imencode(".jpg", first_frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
                if ok:
                    self.preview_frame_data = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
            else:
                demo = DemoConveyor(self.cfg)
                self.source_fps = 15
                is_file = False
            self.state = "running"
            self.started = time.monotonic()
            while not stop.is_set():
                if self.cfg.mode == "real" and not self.play_event.is_set():
                    if self.state != "paused":
                        self.state = "paused"
                        publish(self.snapshot())
                    self.pause_ack.set()
                    stop.wait(.05)
                    continue
                self.pause_ack.clear()
                self.state = "running"
                started = time.monotonic()
                if self.cfg.mode == "real":
                    if first_frame is not None:
                        ok, frame, first_frame = True, first_frame, None
                    else:
                        ok, frame = self.capture.read()
                    if not ok:
                        if is_file:
                            self.state = "completed"  # Do not loop files: that would count again.
                            self.video["decoded_frames"] = self.frame_number
                            self.video["decode_matches_metadata"] = self.frame_number == self.video["frame_count"]
                            break
                        raise RuntimeError("Camera stream lost; stop conveyor before restarting capture")
                    if frame.shape[1] > self.cfg.max_width:
                        scale = self.cfg.max_width / frame.shape[1]
                        frame = cv2.resize(frame, (self.cfg.max_width, round(frame.shape[0] * scale)))
                    observations = self.models.detect(frame)
                    classifier = (lambda crop, tracker_id: self.models.classify(crop)) if self.cfg.defect_enabled else None
                    self.frame_number += 1
                    self.media_time = (self.frame_number - 1) / self.source_fps if self.source_fps > 0 else 0
                else:
                    self.frame_number += 1
                    frame, observations = demo.read(self.frame_number)
                    classifier = lambda crop, tracker_id: ("defect", "cracked", 0.96) if tracker_id % 11 == 0 else ("good", "normal", 0.99)
                    self.media_time = (self.frame_number - 1) / self.source_fps
                pending_events.extend(self.process(frame, observations, classifier))
                duration = time.monotonic() - started
                instantaneous = 1 / max(duration, 1e-6)
                if self.cfg.mode == "demo":
                    instantaneous = min(self.source_fps, instantaneous)
                self.fps = instantaneous if self.fps == 0 else .85 * self.fps + .15 * instantaneous
                now = time.monotonic()
                if now - last_publish >= 1 / self.cfg.stream_fps or pending_events:
                    self.annotate(frame, observations)
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
                    packet = self.snapshot()
                    packet["events"] = pending_events
                    packet["frame_data"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii") if ok else None
                    self.preview_frame_data = packet["frame_data"]
                    publish(packet)
                    pending_events = []
                    last_publish = now
                # File processing reads EVERY frame sequentially; only streaming
                # is throttled. Camera drivers may still drop frames under load.
                if self.cfg.mode == "demo":
                    stop.wait(max(0, 1 / self.source_fps - (time.monotonic() - started)))
            if stop.is_set():
                self.state = "stopped"
        except Exception as exc:
            LOG.exception("CV pipeline stopped")
            self.state, self.error = "error", str(exc)
        finally:
            self.pause_ack.set()
            if self.capture is not None:
                self.capture.release()
            final = self.snapshot()
            final["events"] = pending_events
            publish(final)


@dataclass(eq=False)
class Subscriber:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    overflow: bool = False


class Hub:
    def __init__(self, initial: dict):
        self.latest = initial
        self.clients: set[Subscriber] = set()

    def publish(self, packet: dict):
        # Called only on the event loop. Overflow closes the slow client; the
        # client reconnects with its SQLite cursor rather than losing events.
        self.latest = packet
        for subscriber in tuple(self.clients):
            if subscriber.overflow:
                continue
            try:
                subscriber.queue.put_nowait(packet)
            except asyncio.QueueFull:
                subscriber.overflow = True
                while not subscriber.queue.empty():
                    subscriber.queue.get_nowait()
                subscriber.queue.put_nowait(None)


def create_app(config: Config | None = None, start_worker: bool = True) -> FastAPI:
    cfg = config or Config.from_env()

    def start_engine(app, engine):
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        app.state.stop = stop
        app.state.worker = None
        def publish_if_current(packet):
            # A completed old worker must not overwrite a replay's new session.
            if app.state.engine is engine:
                app.state.hub.publish(packet)
        if start_worker:
            worker = threading.Thread(target=engine.run,
                args=(stop, lambda packet: loop.call_soon_threadsafe(publish_if_current, packet)),
                name="egg-cv", daemon=True)
            app.state.worker = worker
            worker.start()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = Engine(cfg)
        hub = Hub(engine.snapshot())
        app.state.engine, app.state.hub = engine, hub
        app.state.control_lock = asyncio.Lock()
        start_engine(app, engine)
        try:
            yield
        finally:
            app.state.stop.set()
            if app.state.worker:
                await asyncio.to_thread(app.state.worker.join, 10)

    app = FastAPI(title="Egg Counting · CPU OpenVINO", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.origins), allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.post("/api/control")
    async def control(request: Request):
        """Pause the CV worker itself. Explicit replay creates a fresh ledger run."""
        if cfg.mode != "real":
            raise HTTPException(409, "Pipeline controls are available in real mode")
        origin = request.headers.get("origin")
        if origin and origin not in cfg.origins:
            raise HTTPException(403, "Origin not allowed")
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "Expected a JSON action")
        action = body.get("action") if isinstance(body, dict) else None
        if action not in {"resume", "pause", "reset", "replay"}:
            raise HTTPException(400, "Action must be resume, pause, reset or replay")
        async with app.state.control_lock:
            engine = app.state.engine
            if action in {"reset", "replay"}:
                if engine.state not in {"paused", "completed", "error"}:
                    raise HTTPException(409, "Pause or finish the current run before restarting")
                app.state.stop.set()
                if app.state.worker:
                    await asyncio.to_thread(app.state.worker.join, 10)
                    if app.state.worker.is_alive():
                        raise HTTPException(503, "Previous worker is still stopping")
                engine = Engine(cfg)
                if action == "replay":
                    engine.play_event.set()  # The user explicitly requested replay.
                app.state.engine = engine
                app.state.hub.publish(engine.snapshot())
                start_engine(app, engine)
            elif engine.state in {"starting", "completed", "error", "stopped"}:
                raise HTTPException(409, f"Cannot {action} while pipeline is {engine.state}")
            elif action == "resume":
                engine.pause_ack.clear()
                engine.play_event.set()
            else:
                engine.play_event.clear()
                # Return only once an in-flight frame finishes and capture stops.
                if not await asyncio.to_thread(engine.pause_ack.wait, 10):
                    raise HTTPException(503, "Pause is pending; a frame is still processing")
            return {"run_id": engine.run_id, "state": engine.state, "action": action}

    @app.get("/api/health")
    async def health():
        packet = app.state.hub.latest
        return {"state": packet["state"], "mode": cfg.mode, "error": packet["error"], "run_id": packet["run_id"]}

    @app.get("/api/stats")
    async def stats():
        return {**app.state.hub.latest, "frame_data": None, "events": []}

    @app.get("/api/video")
    async def video():
        # Only expose the specifically configured source, never a client path.
        # FileResponse supports HTTP byte ranges for browser seek/pause/replay.
        if cfg.mode != "video":
            raise HTTPException(409, "Native video playback is available only in video preview mode")
        path = Path(cfg.source)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise HTTPException(404, "Configured MP4 not found")
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-cache"})

    @app.get("/api/events")
    async def events(after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=1000), run_id: str | None = None):
        rows = await asyncio.to_thread(app.state.engine.store.after, after, limit, run_id)
        return {"events": rows, "next_cursor": rows[-1]["sequence"] if rows else after,
                "has_more": len(rows) == limit}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket, after: int = Query(0, ge=0), run_id: str | None = None):
        origin = ws.headers.get("origin")
        if origin and origin not in cfg.origins:
            await ws.close(code=1008)
            return
        await ws.accept()
        hub: Hub = app.state.hub
        subscriber = Subscriber()
        # Subscribe BEFORE replay query; queued live events are filtered by the
        # final replay cursor. This closes the replay/live race without CV locks.
        hub.clients.add(subscriber)
        cursor = after if run_id == hub.latest["run_id"] else 0

        async def send(packet):
            await asyncio.wait_for(ws.send_json(packet), timeout=5)

        async def receive_disconnect():
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    return

        disconnect = asyncio.create_task(receive_disconnect())
        try:
            await send({**hub.latest, "events": [], "type": "snapshot"})
            while True:
                rows = await asyncio.to_thread(app.state.engine.store.after, cursor, 500)
                if not rows:
                    break
                await send({"type": "event_replay", "run_id": hub.latest["run_id"], "events": rows})
                cursor = rows[-1]["sequence"]
                if len(rows) < 500:
                    break
            while not disconnect.done():
                get_packet = asyncio.create_task(subscriber.queue.get())
                done, _ = await asyncio.wait({get_packet, disconnect}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
                if get_packet not in done:
                    get_packet.cancel()
                    await asyncio.gather(get_packet, return_exceptions=True)
                    if disconnect.done():
                        break
                    await send({**hub.latest, "frame_data": None, "events": [], "type": "heartbeat"})
                    continue
                packet = get_packet.result()
                if packet is None:
                    await ws.close(code=1013, reason="Slow client: reconnect to replay durable events")
                    break
                fresh = [e for e in packet["events"] if e["sequence"] > cursor]
                await send({**packet, "events": fresh})
                if fresh:
                    cursor = fresh[-1]["sequence"]
        except (WebSocketDisconnect, TimeoutError, RuntimeError):
            pass
        finally:
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)
            hub.clients.discard(subscriber)

    return app


app = create_app()
