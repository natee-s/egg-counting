"""Behavior tests for counting, not mocked claims of real camera accuracy."""
from dataclasses import replace
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import Config, Engine, EventStore, LineCounter, Observation, Phase, create_app


def obs(identity, x, y, size=20):
    return Observation(identity, (x, y), (x - size / 2, y - size / 2, x + size / 2, y + size / 2))


@pytest.fixture
def config(tmp_path):
    return Config(db=tmp_path / "events.sqlite3", line_position=.5, hysteresis=5,
                  arm_frames=2, recover_distance=6)


def test_forward_then_roll_backward_counts_once(config):
    counter = LineCounter(config)
    hits = []
    for frame, y in enumerate([30, 35, 48, 52, 45, 32, 48, 55], 1):
        hits += counter.update([obs(1, 30, y)], frame, 100, 100)
    assert len(hits) == 1
    assert hits[0][2] == pytest.approx((30, 50))
    assert counter.states[1].phase == Phase.COUNTED


def test_birth_downstream_does_not_count(config):
    counter = LineCounter(config)
    for frame, y in enumerate([60, 70, 80], 1):
        assert counter.update([obs(1, 30, y)], frame, 100, 100) == []


def test_jitter_in_band_never_arms(config):
    counter = LineCounter(config)
    for frame, y in enumerate([48, 51, 49, 52, 48, 51], 1):
        assert counter.update([obs(1, 30, y)], frame, 100, 100) == []


def test_simultaneous_adjacent_eggs_no_global_cooldown(config):
    counter = LineCounter(config)
    counter.update([obs(1, 30, 30), obs(2, 55, 30)], 1, 100, 100)
    counter.update([obs(1, 30, 35), obs(2, 55, 35)], 2, 100, 100)
    crossings = counter.update([obs(1, 30, 55), obs(2, 55, 55)], 3, 100, 100)
    assert len(crossings) == 2
    assert len({state.physical_id for state, _, _ in crossings}) == 2


def test_occlusion_and_id_switch_transfer_armed_state(config):
    counter = LineCounter(config)
    counter.update([obs(10, 30, 30)], 1, 100, 100)
    counter.update([obs(10, 30, 40)], 2, 100, 100)
    counter.update([], 3, 100, 100)
    crossings = counter.update([obs(99, 30, 60)], 4, 100, 100)
    assert len(crossings) == 1
    assert crossings[0][0].physical_id == 1
    assert counter.recovered == 1


def test_counted_id_switch_does_not_classify_or_count_again(config):
    counter = LineCounter(config)
    counter.update([obs(10, 30, 30)], 1, 100, 100)
    counter.update([obs(10, 30, 40)], 2, 100, 100)
    assert len(counter.update([obs(10, 30, 52)], 3, 100, 100)) == 1
    assert counter.update([obs(99, 30, 64)], 4, 100, 100) == []
    assert counter.states[1].phase == Phase.COUNTED
    assert counter.recovered == 1


def test_ambiguous_id_recovery_does_not_merge_two_eggs(config):
    counter = LineCounter(replace(config, recover_distance=20))
    counter.update([obs(1, 40, 30), obs(2, 60, 30)], 1, 100, 100)
    counter.update([obs(1, 40, 40), obs(2, 60, 40)], 2, 100, 100)
    crossings = counter.update([obs(3, 50, 52)], 3, 100, 100)
    assert crossings == []
    assert counter.ambiguous == 1
    assert counter.recovered == 0


def test_one_missing_track_cannot_recover_as_two_new_tracks(config):
    counter = LineCounter(replace(config, recover_distance=20))
    counter.update([obs(1, 50, 30)], 1, 100, 100)
    counter.update([obs(1, 50, 40)], 2, 100, 100)
    counter.update([obs(2, 45, 50), obs(3, 55, 50)], 3, 100, 100)
    assert counter.recovered == 0
    assert counter.aliases[2] != counter.aliases[3]


@pytest.mark.parametrize("axis,direction,coordinates", [
    ("x", 1, [30, 40, 55]), ("x", -1, [70, 60, 45]), ("y", -1, [70, 60, 45]),
])
def test_other_line_orientations(config, axis, direction, coordinates):
    counter = LineCounter(replace(config, axis=axis, direction=direction))
    hits = []
    for frame, coordinate in enumerate(coordinates, 1):
        observation = obs(1, coordinate, 30) if axis == "x" else obs(1, 30, coordinate)
        hits += counter.update([observation], frame, 100, 100)
    assert len(hits) == 1


def test_large_observation_gap_requires_rearming(config):
    counter = LineCounter(replace(config, max_cross_gap=3))
    counter.update([obs(1, 30, 30)], 1, 100, 100)
    counter.update([obs(1, 30, 40)], 2, 100, 100)
    assert counter.update([obs(1, 30, 60)], 7, 100, 100) == []
    assert counter.long_gaps == 1


def test_classifier_only_at_crossing_and_raw_crop(config):
    engine = Engine(config)
    frame = np.full((100, 100, 3), 123, np.uint8)
    calls = []
    def classify(crop, identity):
        calls.append(identity)
        assert crop.shape == (20, 20, 3)
        assert np.all(crop == 123)  # No annotation contaminates classification.
        return "good", "normal", .99
    for number, y in enumerate([30, 40, 55, 40, 55], 1):
        engine.frame_number = number
        engine.process(frame, [obs(1, 30, y)], classify)
    assert calls == [1]
    assert engine.store.total == engine.store.normal == 1
    assert len(engine.store.after(0)) == 1


def test_classifier_failure_preserves_count_as_unknown(config):
    engine = Engine(config)
    frame = np.zeros((100, 100, 3), np.uint8)
    def fail(crop, identity):
        raise RuntimeError("inference unavailable")
    for number, y in enumerate([30, 40, 55], 1):
        engine.frame_number = number
        engine.process(frame, [obs(1, 30, y)], fail)
    assert engine.store.total == engine.store.unclassified == 1
    assert engine.store.normal == engine.store.defect == 0


def test_store_is_idempotent_and_run_scoped(config):
    store = EventStore(config.db, "run-one")
    event = {"id": 1, "status": "good", "timestamp": "test"}
    saved = store.record(event)
    assert store.record(event) is None
    assert store.total == 1
    assert store.after(saved["sequence"]) == []
    second = EventStore(config.db, "run-two")
    assert second.record(event)["sequence"] > saved["sequence"]
    assert len(second.after(0, run_id="run-one")) == 1


def test_websocket_snapshot_replay_and_reconnect_cursor(config):
    app = create_app(config, start_worker=False)
    with TestClient(app) as client:
        engine = app.state.engine
        first = engine.store.record({"id": 1, "status": "good", "timestamp": "test"})
        second = engine.store.record({"id": 2, "status": "defect", "timestamp": "test"})
        engine.latest_sequence = second["sequence"]
        app.state.hub.publish(engine.snapshot())
        with client.websocket_connect("/ws") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["stats"]["total_count"] == 2
            replay = ws.receive_json()
            assert replay["type"] == "event_replay"
            assert [e["id"] for e in replay["events"]] == [1, 2]
        with client.websocket_connect(f"/ws?run_id={engine.run_id}&after={first['sequence']}") as ws:
            ws.receive_json()
            assert [e["id"] for e in ws.receive_json()["events"]] == [2]
        assert client.get("/api/events?limit=1").json()["has_more"] is True
        assert client.get("/api/health").json()["mode"] == "demo"


def test_disallowed_websocket_origin(config):
    with TestClient(create_app(config, start_worker=False)) as client:
        with pytest.raises(Exception) as failure:
            with client.websocket_connect("/ws", headers={"origin": "http://untrusted.example"}):
                pass
        assert getattr(failure.value, "code", None) == 1008


def test_real_missing_models_are_reported_not_silently_demo(config):
    real = replace(config, mode="real", seg_model=config.db.parent / "missing")
    # Dependency may not be installed in the minimal demo environment. Both
    # dependency/model failures must be visible and must not fabricate eggs.
    with TestClient(create_app(real)) as client:
        for _ in range(100):
            result = client.get("/api/health").json()
            if result["state"] == "error":
                break
            time.sleep(.05)
        assert result["state"] == "error"
        assert result["error"]
        assert client.get("/api/stats").json()["stats"]["total_count"] == 0


def test_demo_end_to_end_frame_and_event(config):
    with TestClient(create_app(config)) as client:
        with client.websocket_connect("/ws") as ws:
            for _ in range(150):
                packet = ws.receive_json()
                if packet.get("events"):
                    assert packet["events"][0]["status"] in {"good", "defect"}
                    assert packet["stats"]["total_count"] > 0
                    assert packet["frame_data"].startswith("data:image/jpeg;base64,")
                    break
            else:
                pytest.fail("Demo never emitted a crossing event")
