"""Extract and validate the keypoints the posture classifier needs."""

from typing import Dict, List, Optional, Tuple

from posture.config import KeypointIndex, PostureConfig
from posture.geometry import GeometryUtils


class KeypointExtractor:
    def __init__(self, config: Optional[PostureConfig] = None):
        self.config = config or PostureConfig()
        self.geo = GeometryUtils()

    def extract(
        self,
        person_keypoints: List[List[float]],
        confidence_scores: Optional[List[float]] = None,
    ) -> Dict:
        """Return named keypoints (None when missing/low-confidence) plus midpoints."""

        def get_kpt_if_valid(idx: int) -> Optional[Tuple[float, float]]:
            if idx >= len(person_keypoints):
                return None
            if confidence_scores is not None:
                if idx >= len(confidence_scores):
                    return None
                if confidence_scores[idx] < self.config.min_keypoint_confidence:
                    return None
            kpt = person_keypoints[idx]
            if kpt is None or len(kpt) < 2:
                return None
            if kpt[0] == 0 and kpt[1] == 0:
                return None
            return (float(kpt[0]), float(kpt[1]))

        kpts = {
            "shoulder_left": get_kpt_if_valid(KeypointIndex.LEFT_SHOULDER),
            "shoulder_right": get_kpt_if_valid(KeypointIndex.RIGHT_SHOULDER),
            "hip_left": get_kpt_if_valid(KeypointIndex.LEFT_HIP),
            "hip_right": get_kpt_if_valid(KeypointIndex.RIGHT_HIP),
            "knee_left": get_kpt_if_valid(KeypointIndex.LEFT_KNEE),
            "knee_right": get_kpt_if_valid(KeypointIndex.RIGHT_KNEE),
            "ankle_left": get_kpt_if_valid(KeypointIndex.LEFT_ANKLE),
            "ankle_right": get_kpt_if_valid(KeypointIndex.RIGHT_ANKLE),
        }

        for part in ("shoulder", "hip", "knee", "ankle"):
            left, right = kpts[f"{part}_left"], kpts[f"{part}_right"]
            kpts[f"{part}_mid"] = self.geo.calculate_midpoint(left, right) if left and right else None

        return kpts
