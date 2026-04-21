"""Calibration logic for the eye-tracking research application.

This module isolates calibration-related data models and fitting logic from the
tracking runtime. It can be imported independently by both the GUI layer and the
tracker backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(slots=True)
class CalibrationSample:
    """One accepted feature sample for a given on-screen target."""

    target_x: float
    target_y: float
    features: list[float]
    timestamp: str


@dataclass(slots=True)
class CalibrationData:
    """Serializable calibration payload stored in YAML."""

    timestamp: str
    camera_resolution: list[int]
    feature_names: list[str]
    model_type: str
    transformation_matrix: list[list[float]]
    target_layout: list[list[float]]
    validation_error_px: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CalibrationModel:
    """Second-order polynomial regression mapping features to screen space.

    The model uses six input features and expands them to a fixed design vector.
    This is intentionally simple and interpretable, while still handling common
    non-linearities in gaze mapping.
    """

    FEATURE_NAMES = [
        "left_iris_x",
        "left_iris_y",
        "right_iris_x",
        "right_iris_y",
        "head_yaw",
        "head_pitch",
    ]

    def __init__(self) -> None:
        self.coefficients: Optional[np.ndarray] = None
        self.target_layout = self.default_target_layout()
        self.validation_error_px: Optional[float] = None

    @staticmethod
    def default_target_layout() -> list[list[float]]:
        return [
            [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
            [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
            [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
        ]

    @staticmethod
    def expand_features(features: Iterable[float]) -> np.ndarray:
        vals = np.asarray(list(features), dtype=np.float64)
        if vals.size != 6:
            raise ValueError("CalibrationModel expects exactly 6 features.")
        x1, x2, x3, x4, x5, x6 = vals
        return np.asarray(
            [
                1.0,
                x1, x2, x3, x4, x5, x6,
                x1 * x1, x2 * x2, x3 * x3, x4 * x4, x5 * x5, x6 * x6,
                x1 * x2, x1 * x3, x1 * x4, x2 * x3, x2 * x4, x3 * x4,
            ],
            dtype=np.float64,
        )

    def fit(self, samples: list[CalibrationSample], screen_size: tuple[int, int]) -> CalibrationData:
        if len(samples) < 9:
            raise ValueError("At least 9 calibration samples are required.")

        design = np.vstack([self.expand_features(sample.features) for sample in samples])
        targets = np.asarray(
            [[sample.target_x * screen_size[0], sample.target_y * screen_size[1]] for sample in samples],
            dtype=np.float64,
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
        self.coefficients = coefficients

        predictions = design @ coefficients
        error_px = np.linalg.norm(predictions - targets, axis=1)
        self.validation_error_px = float(np.mean(error_px))

        return CalibrationData(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            camera_resolution=[screen_size[0], screen_size[1]],
            feature_names=self.FEATURE_NAMES,
            model_type="polynomial_regression_order_2",
            transformation_matrix=coefficients.tolist(),
            target_layout=self.target_layout,
            validation_error_px=self.validation_error_px,
        )

    def load_from_dict(self, payload: dict[str, Any]) -> None:
        matrix = payload.get("transformation_matrix")
        self.coefficients = np.asarray(matrix, dtype=np.float64) if matrix else None
        self.validation_error_px = payload.get("validation_error_px")
        self.target_layout = payload.get("target_layout", self.default_target_layout())

    def map_to_screen(
        self,
        features: list[float],
        screen_size: tuple[int, int],
    ) -> tuple[float, float, bool]:
        if self.coefficients is None:
            return 0.0, 0.0, False
        expanded = self.expand_features(features)
        output = expanded @ self.coefficients
        width, height = screen_size
        x_px = float(np.clip(output[0], 0, width))
        y_px = float(np.clip(output[1], 0, height))
        return x_px / max(width, 1), y_px / max(height, 1), True
