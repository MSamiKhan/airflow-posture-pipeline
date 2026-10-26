"""Tests for the classify -> evaluate steps using synthetic keypoints (no YOLO needed)."""

import json

from posture import pipeline


def _person(shoulder, hip, knee, ankle, spread=5.0, vertical=True):
    """17 x [x, y, conf] array with only the keypoints the classifier uses filled in."""
    kpts = [[0.0, 0.0, 0.0] for _ in range(17)]
    dx, dy = (spread, 0) if vertical else (0, spread)
    for (left_idx, right_idx), mid in (
        ((5, 6), shoulder), ((11, 12), hip), ((13, 14), knee), ((15, 16), ankle),
    ):
        kpts[left_idx] = [mid[0] - dx, mid[1] - dy, 0.9]
        kpts[right_idx] = [mid[0] + dx, mid[1] + dy, 0.9]
    return kpts


def _record(person_id, keypoints):
    return {
        "video": "demo.mp4", "frame": person_id * 15, "person": person_id,
        "frame_height": 480, "frame_width": 640, "infer_ms": 40.0 + person_id,
        "keypoints": keypoints,
    }


def test_classify_and_evaluate(tmp_path):
    records = [
        _record(0, _person((100, 100), (100, 200), (100, 300), (100, 400))),
        _record(1, _person((100, 300), (200, 300), (300, 300), (400, 300), vertical=False)),
        _record(2, [[0.0, 0.0, 0.0] for _ in range(17)]),
    ]
    keypoints_path = tmp_path / "keypoints.json"
    keypoints_path.write_text(json.dumps(records))

    predictions_path = pipeline.classify_posture(keypoints_path, tmp_path)
    labels = [p["label"] for p in json.loads(predictions_path.read_text())]
    assert labels == ["Standing", "Lying", "Insufficient_Keypoints"]

    report = json.loads(pipeline.evaluate(predictions_path, tmp_path).read_text())
    assert report["num_detections"] == 3
    assert report["label_distribution"]["Lying"] == 1
    assert report["insufficient_keypoints_rate"] == 0.333


def test_upload_skips_without_bucket(tmp_path):
    assert pipeline.upload_to_s3(tmp_path, bucket=None) == []


def test_run_dir_sanitises_run_id(tmp_path):
    path = pipeline.run_dir(tmp_path, "manual__2026-01-01T00:00:00+00:00")
    assert path.exists()
    assert ":" not in path.name and "+" not in path.name
