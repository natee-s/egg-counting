"""Finite user lines, count-only operation and actual worker pause."""
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.main import Config, Engine, LineCounter, Observation, create_app, path_crossing


def observation(identity, x, y):
    return Observation(identity, (x, y), (x-5, y-5, x+5, y+5))


def test_sloping_line_counts_forward_once_after_rollback():
    counter = LineCounter(Config(line_points=((.2, 0), (.8, 1)), hysteresis=3))
    hits = []
    for frame, x in enumerate([30, 35, 48, 55, 40, 35, 60], 1):
        hits += counter.update([observation(1, x, 50)], frame, 100, 100)
    assert len(hits) == 1
    assert hits[0][2] == pytest.approx((50, 50))


def test_finite_line_does_not_count_beyond_endpoint():
    counter = LineCounter(Config(line_points=((.3, .2), (.7, .8)), hysteresis=3))
    hits = []
    for frame, x in enumerate([70, 75, 80, 90], 1):
        hits += counter.update([observation(1, x, 95)], frame, 100, 100)
    assert hits == []


def test_polyline_intersection_uses_segment_not_endpoint_chord():
    points = [(10, 0), (70, 50), (80, 100)]
    assert path_crossing((20, 25), (60, 25), points) == pytest.approx((40, 25))
    assert path_crossing((70, 110), (95, 110), points) is None


def test_count_only_does_not_call_or_fail_classifier(tmp_path):
    engine = Engine(Config(db=tmp_path/'events.sqlite3', axis='x', line_position=.5, hysteresis=3))
    frame = np.zeros((100, 100, 3), np.uint8)
    for number, x in enumerate([30, 35, 55], 1):
        engine.frame_number = number
        engine.process(frame, [observation(1, x, 50)], None)
    assert engine.store.total == engine.store.unclassified == 1
    assert engine.store.normal == engine.store.defect == 0
    assert engine.classifier_calls == engine.classifier_errors == 0
    event = engine.store.after(0)[0]
    assert (event['status'], event['label'], event['confidence']) == ('unknown', 'not_inspected', None)


def wait_state(client, states):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get('/api/stats').json()
        if result['state'] in states:
            return result
        time.sleep(.01)
    pytest.fail(f'Expected {states}, got {result}')


def test_real_video_starts_paused_and_controls_stop_capture_and_reset(monkeypatch, tmp_path):
    path = tmp_path/'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (160, 96))
    assert writer.isOpened()
    for i in range(40):
        writer.write(np.full((96, 160, 3), 50, np.uint8))
    writer.release()
    class Models:
        def __init__(self, cfg):
            self.number = 0
        def detect(self, frame):
            time.sleep(.02)
            self.number += 1
            return [observation(1, min(130, 25+self.number*4), 50)]
        def classify(self, frame):
            pytest.fail('Disabled classifier was called')
    monkeypatch.setattr('backend.main.OpenVINOModels', Models)
    cfg = Config(mode='real', source=str(path), db=tmp_path/'events.sqlite3',
                 axis='x', start_paused=1, defect_enabled=0)
    with TestClient(create_app(cfg)) as client:
        initial = wait_state(client, {'paused'})
        assert initial['stats']['processed_frames'] == initial['stats']['total_count'] == 0
        with client.websocket_connect('/ws') as ws:
            assert ws.receive_json()['frame_data'].startswith('data:image/jpeg;base64,')
        assert client.post('/api/control', json={'action':'resume'}).status_code == 200
        wait_state(client, {'running'})
        assert client.post('/api/control', json={'action':'pause'}).status_code == 200
        paused = wait_state(client, {'paused'})
        frames = paused['stats']['processed_frames']
        time.sleep(.1)
        assert client.get('/api/stats').json()['stats']['processed_frames'] == frames
        assert client.post('/api/control', json={'action':'resume'}).status_code == 200
        completed = wait_state(client, {'completed'})
        assert completed['stats']['total_count'] == completed['stats']['unclassified_count'] == 1
        assert completed['stats']['processed_frames'] == 40
        assert completed['stats']['classifier_calls'] == completed['stats']['classifier_errors'] == 0
        run = completed['run_id']
        time.sleep(.1)
        assert client.get('/api/stats').json()['run_id'] == run
        assert client.post('/api/control', json={'action':'reset'}).status_code == 200
        reset = wait_state(client, {'paused'})
        assert reset['run_id'] != run
        assert reset['stats']['total_count'] == reset['stats']['processed_frames'] == 0
        assert len(client.get('/api/events', params={'run_id':run}).json()['events']) == 1


def test_control_rejects_other_origins_and_bad_actions(tmp_path):
    cfg = Config(mode='real', db=tmp_path/'events.sqlite3')
    with TestClient(create_app(cfg, start_worker=False)) as client:
        assert client.post('/api/control', json={'action':'resume'}, headers={'Origin':'http://untrusted.example'}).status_code == 403
        assert client.post('/api/control', json={'action':'bogus'}).status_code == 400
        assert client.post('/api/control', json={'action':'resume'}).status_code == 409


def test_annotation_shows_counting_centre_and_box_without_contour(tmp_path):
    engine = Engine(Config(db=tmp_path/'events.sqlite3', axis='x', line_position=.8))
    frame = np.zeros((100, 100, 3), np.uint8)
    egg = Observation(1, (30, 50), (20, 40, 40, 60),
                      np.array([[25, 45], [35, 45], [35, 55], [25, 55]], np.float32))
    engine.frame_number = 1
    engine.process(frame, [egg], None)
    engine.annotate(frame, [egg])
    assert frame[50, 30].tolist() == [0, 0, 255]  # Counting centre in red.
    assert frame[50, 20].sum() > 0  # Rectangular box.
    assert frame[45, 25].sum() == 0  # No mask contour inside the box.
    assert engine.snapshot()['stats']['tracked_count'] == 1
    assert engine.snapshot()['stats']['total_count'] == 0


def test_detection_threshold_cannot_discard_bytetrack_rescue(monkeypatch):
    monkeypatch.setenv('EGG_DETECTION_CONF', '.5')
    monkeypatch.setenv('EGG_TRACK_LOW_THRESH', '.1')
    with pytest.raises(ValueError, match='DETECTION_CONF'):
        Config.from_env()
