"""Unit tests for the posture classifier using hand-built keypoints."""

from posture.classifier import PostureClassifier
from posture.config import PostureLabel


def _kpts(shoulder, hip, knee, ankle, spread=5.0, vertical=True):
    """Build a keypoint dict from four body-part midpoints, with left/right offsets."""

    def pair(mid):
        dx, dy = (spread, 0) if vertical else (0, spread)
        return (mid[0] - dx, mid[1] - dy), (mid[0] + dx, mid[1] + dy)

    kpts = {}
    for name, mid in (("shoulder", shoulder), ("hip", hip), ("knee", knee), ("ankle", ankle)):
        left, right = pair(mid)
        kpts[f"{name}_left"], kpts[f"{name}_right"], kpts[f"{name}_mid"] = left, right, mid
    return kpts


def test_standing():
    kpts = _kpts((100, 100), (100, 200), (100, 300), (100, 400))
    label, conf = PostureClassifier().classify(kpts, 480)
    assert label == PostureLabel.STANDING
    assert conf >= 0.8


def test_lying():
    kpts = _kpts((100, 300), (200, 300), (300, 300), (400, 300), vertical=False)
    label, _ = PostureClassifier().classify(kpts, 480)
    assert label == PostureLabel.LYING


def test_insufficient_keypoints():
    kpts = {"shoulder_mid": None, "hip_mid": None}
    label, conf = PostureClassifier().classify(kpts, 480)
    assert label == PostureLabel.INSUFFICIENT
    assert conf == 0.0
