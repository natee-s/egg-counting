"""Real file preview and HTTP range requests must never generate demo counts."""
from dataclasses import replace
import threading
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import Config, Engine, create_app


def video_config(tmp_path):
    path = tmp_path / "Eggs On A Conveyor Stock Video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 96))
    assert writer.isOpened()
    for number in range(10):
        writer.write(np.full((96, 160, 3), number * 20, np.uint8))
    writer.release()
    return Config(mode="video", source=str(path), db=tmp_path / "events.sqlite3")


def test_video_mode_has_real_poster_metadata_and_zero_synthetic_events(tmp_path):
    config = video_config(tmp_path)
    engine = Engine(config)
    packets = []
    engine.run(threading.Event(), packets.append)
    assert len(packets) == 1
    packet = packets[0]
    assert packet["state"] == "preview"
    assert packet["mode"] == "video"
    assert packet["frame_size"] == {"width": 160, "height": 96}
    assert packet["frame_data"].startswith("data:image/jpeg;base64,")
    assert packet["video"]["name"] == "Eggs On A Conveyor Stock Video.mp4"
    assert packet["video"]["fps"] == 25
    assert packet["video"]["duration_seconds"] == .4
    assert packet["video"]["counting_enabled"] is False
    assert packet["stats"]["total_count"] == packet["stats"]["classifier_calls"] == 0
    assert packet["events"] == []
    assert engine.models is None
    assert engine.store.after(0) == []


def test_http_range_streams_only_configured_mp4_and_supports_seek(tmp_path):
    config = video_config(tmp_path)
    with TestClient(create_app(config, start_worker=False)) as client:
        response = client.get("/api/video", headers={"Range": "bytes=0-63"})
        assert response.status_code == 206
        assert response.headers["content-type"] == "video/mp4"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.content == open(config.source, "rb").read(64)
        assert client.get("/api/video", headers={"Range": "bytes=99999999-"}).status_code == 416


def test_video_preview_rejects_missing_file_and_other_modes(tmp_path):
    missing = Config(mode="video", source=str(tmp_path / "missing.mp4"), db=tmp_path / "events.sqlite3")
    engine = Engine(missing)
    packets = []
    engine.run(threading.Event(), packets.append)
    assert packets[0]["state"] == "error"
    assert packets[0]["stats"]["total_count"] == 0
    with TestClient(create_app(missing, start_worker=False)) as client:
        assert client.get("/api/video").status_code == 404
    with TestClient(create_app(replace(missing, mode="demo"), start_worker=False)) as client:
        assert client.get("/api/video").status_code == 409
