"""Posture detection thresholds, labels and YOLO keypoint indices."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PostureConfig:
    # Angle thresholds (degrees)
    vertical_deviation_threshold: float = 12.0
    alignment_threshold: float = 10.0
    bending_threshold: float = 20.0

    # Height ratio thresholds (sitting detection)
    sitting_hip_height_ratio_max: float = 0.45

    # Knee angle thresholds (degrees, straight leg = 180)
    sitting_knee_angle_max: float = 120.0
    standing_knee_angle_min: float = 135.0
    squatting_knee_angle_range: Tuple[float, float] = (60.0, 120.0)

    # Horizontal orientation threshold (lying detection)
    horizontal_deviation_threshold: float = 30.0

    # Confidence thresholds
    min_keypoint_confidence: float = 0.3
    lying_duration_threshold: float = 0.5


class PostureLabel:
    STANDING = "Standing"
    SITTING = "Sitting"
    BENDING = "Bending"
    SQUATTING = "Squatting"
    LYING = "Lying"
    UNKNOWN = "Unknown"
    INSUFFICIENT = "Insufficient_Keypoints"
    FALLING = "Falling"


class KeypointIndex:
    """COCO-17 indices used by YOLO pose models."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
