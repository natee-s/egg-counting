"""Real OpenVINO + Ultralytics + ByteTrack API integration with synthetic IR.

No pretrained weights, network, camera, or claimed egg detection accuracy.
Constant model outputs test format, preprocessing, mask indexing, and CPU runtime.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
pytest.importorskip("ultralytics")
from openvino import opset13 as op
import yaml
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

from backend.main import Config, OpenVINOModels


def write_ir(path, task, size):
    path.mkdir()
    parameter = op.parameter([1, 3, size, size], np.float32, name="images")
    zero = op.multiply(op.reduce_sum(parameter, op.constant(np.array([0, 1, 2, 3], np.int64)), False), op.constant(np.float32(0)))
    if task == "segment":
        output = np.zeros((1, 37, 2), np.float32)
        output[0, :5, 0] = [120, 160, 60, 80, .95]
        output[0, :5, 1] = [280, 160, 60, 80, .90]
        output[0, 5:, :] = 1
        outputs = [op.add(op.constant(output), zero), op.add(op.constant(np.ones((1, 32, size // 4, size // 4), np.float32)), zero)]
        names = {0: "egg"}
    elif task == "detect":
        output = np.zeros((1, 5, 2), np.float32)
        output[0, :, 0] = [120, 160, 60, 80, .95]
        output[0, :, 1] = [280, 160, 60, 80, .90]
        outputs = [op.add(op.constant(output), zero)]
        names = {0: "egg"}
    else:
        outputs = [op.add(op.constant(np.array([[.95, .05]], np.float32)), zero)]
        names = {0: "normal", 1: "defect"}
    graph = ov.Model(outputs, [parameter], task)
    ov.save_model(graph, str(path / "model.xml"), compress_to_fp16=False)
    (path / "metadata.yaml").write_text(yaml.safe_dump({"task": task, "names": names, "stride": 32,
                                                        "batch": 1, "imgsz": [size, size], "args": {"dynamic": False}}), encoding="utf-8")


def test_cpu_segmentation_classifier_and_mask_tracker_order(tmp_path):
    seg = tmp_path / "egg_seg_openvino_model"
    cls = tmp_path / "egg_cls_openvino_model"
    write_ir(seg, "segment", 416)
    write_ir(cls, "classify", 224)
    models = OpenVINOModels(Config(seg_model=seg, cls_model=cls, cpu_threads=2))
    frame = np.zeros((416, 416, 3), np.uint8)
    observations = models.detect(frame)
    assert len(observations) == 2
    assert all(o.contour is not None for o in observations)
    assert sorted(o.center[0] for o in observations) == pytest.approx([120, 280], abs=3)
    status, label, confidence = models.classify(frame[:80, :60])
    assert (status, label) == ("good", "normal")
    assert confidence == pytest.approx(.95)
    for model in (models.seg, models.cls):
        assert model.predictor.model.ov_compiled_model.get_property("EXECUTION_DEVICES") == ["CPU"]


def test_actual_bytetrack_retains_id_across_occlusion_and_low_score():
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=.25, track_low_thresh=.1,
                                         new_track_thresh=.3, track_buffer=45,
                                         match_thresh=.8, fuse_score=True), frame_rate=30)
    def boxes(x, score):
        return Boxes(np.array([[x, 10, x + 30, 50, score, 0]], np.float32), (100, 200))
    first = tracker.update(boxes(10, .95))
    assert len(first) == 1
    identity = int(first[0, 4])
    tracker.update(Boxes(np.empty((0, 6), np.float32), (100, 200)))
    recovered = tracker.update(boxes(12, .95))
    assert int(recovered[0, 4]) == identity
    low_score = tracker.update(boxes(14, .2))
    assert int(low_score[0, 4]) == identity


def test_wrong_export_size_is_actionable(tmp_path):
    seg = tmp_path / "egg_seg_openvino_model"
    cls = tmp_path / "egg_cls_openvino_model"
    write_ir(seg, "segment", 416)
    write_ir(cls, "classify", 224)
    with pytest.raises(ValueError, match="match .env to export sizes"):
        OpenVINOModels(Config(seg_model=seg, cls_model=cls, image_size=448))


def test_detection_export_cpu_box_centres_and_tracker_configuration(tmp_path):
    directory = tmp_path / "egg_detect_openvino_model"
    write_ir(directory, "detect", 416)
    models = OpenVINOModels(Config(seg_model=directory, defect_enabled=0,
                                   track_high_thresh=.2, new_track_thresh=.25, fuse_score=0))
    observations = models.detect(np.zeros((416, 416, 3), np.uint8))
    assert models.task == "detect"
    assert models.cls is None
    assert models.last_detection_count == len(observations) == 2
    assert all(o.contour is None for o in observations)
    assert sorted(o.center[0] for o in observations) == pytest.approx([120, 280])
    assert all(o.center[1] == pytest.approx(160) for o in observations)
    assert models.tracker.args.track_high_thresh == .2
    assert models.tracker.args.new_track_thresh == .25
    assert models.tracker.args.fuse_score is False
    assert models.seg.predictor.model.ov_compiled_model.get_property("EXECUTION_DEVICES") == ["CPU"]
