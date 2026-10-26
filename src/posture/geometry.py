"""Geometry helpers for posture detection (angles, distances, alignment)."""

import math
from typing import Tuple

import numpy as np


class GeometryUtils:
    @staticmethod
    def calculate_midpoint(
        point_left: Tuple[float, float], point_right: Tuple[float, float]
    ) -> Tuple[float, float]:
        return (
            (point_left[0] + point_right[0]) / 2.0,
            (point_left[1] + point_right[1]) / 2.0,
        )

    @staticmethod
    def calculate_angle_from_horizontal(
        point_start: Tuple[float, float], point_end: Tuple[float, float]
    ) -> float:
        """Angle from the x-axis in degrees (image coords: up = -90, down = +90)."""
        delta_x = point_end[0] - point_start[0]
        delta_y = point_end[1] - point_start[1]
        return math.degrees(math.atan2(delta_y, delta_x))

    @staticmethod
    def calculate_angle_between_three_points(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_c: Tuple[float, float],
    ) -> float:
        """Angle at point_b formed by a-b-c, in degrees (0-180)."""
        ba = np.array([point_a[0] - point_b[0], point_a[1] - point_b[1]])
        bc = np.array([point_c[0] - point_b[0], point_c[1] - point_b[1]])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @staticmethod
    def normalize_angle_to_180(angle_degrees: float) -> float:
        return (angle_degrees + 180) % 360 - 180

    @staticmethod
    def calculate_angular_difference(angle_a: float, angle_b: float) -> float:
        return abs(GeometryUtils.normalize_angle_to_180(angle_a - angle_b))

    @staticmethod
    def calculate_deviation_from_vertical(angle_from_horizontal: float) -> float:
        up = GeometryUtils.calculate_angular_difference(angle_from_horizontal, -90.0)
        down = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 90.0)
        return min(up, down)

    @staticmethod
    def calculate_deviation_from_horizontal(angle_from_horizontal: float) -> float:
        right = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 0.0)
        left = GeometryUtils.calculate_angular_difference(angle_from_horizontal, 180.0)
        return min(right, left)

    @staticmethod
    def are_segments_aligned(angle_a: float, angle_b: float, threshold: float = 15.0) -> bool:
        same = GeometryUtils.calculate_angular_difference(angle_a, angle_b)
        opposite = GeometryUtils.calculate_angular_difference(angle_a, angle_b + 180.0)
        return min(same, opposite) < threshold
