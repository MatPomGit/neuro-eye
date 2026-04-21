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
    feature_mean: Optional[list[float]] = None
    feature_std: Optional[list[float]] = None

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
        self.feature_mean = np.zeros(6, dtype=np.float64)
        self.feature_std = np.ones(6, dtype=np.float64)
        self._ridge_lambda = 1e-3

    @staticmethod
    def default_target_layout() -> list[list[float]]:
        return [
            [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
            [0.1, 0.5], [0.5, 0.5], [0.9, 0.5],
            [0.1, 0.9], [0.5, 0.9], [0.9, 0.9],
        ]

    @staticmethod
    def _legacy_expand_features(features: Iterable[float]) -> np.ndarray:
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

    @staticmethod
    def expand_features(features: Iterable[float]) -> np.ndarray:
        """Rozszerza cechy do wielomianu 2. rzędu z pełnymi interakcjami."""
        vals = np.asarray(list(features), dtype=np.float64)
        if vals.size != 6:
            raise ValueError("CalibrationModel expects exactly 6 features.")
        terms: list[float] = [1.0]
        terms.extend(vals.tolist())
        terms.extend((vals * vals).tolist())
        # Dzięki interakcjom model lepiej kompensuje błędy zależne od pozy głowy.
        for i in range(vals.size):
            for j in range(i + 1, vals.size):
                terms.append(float(vals[i] * vals[j]))
        return np.asarray(terms, dtype=np.float64)

    def _fit_weighted_ridge(
        self,
        design: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Rozwiązuje ważoną regresję grzbietową dla współczynników kalibracji."""
        xtwx = design.T @ (weights[:, None] * design)
        reg = np.eye(design.shape[1], dtype=np.float64) * self._ridge_lambda
        reg[0, 0] = 0.0
        xtwy = design.T @ (weights[:, None] * targets)
        return np.linalg.pinv(xtwx + reg) @ xtwy

    def fit(self, samples: list[CalibrationSample], screen_size: tuple[int, int]) -> CalibrationData:
        if len(samples) < 9:
            raise ValueError("At least 9 calibration samples are required.")

        raw_features = np.asarray([sample.features for sample in samples], dtype=np.float64)
        self.feature_mean = raw_features.mean(axis=0)
        self.feature_std = raw_features.std(axis=0)
        self.feature_std = np.clip(self.feature_std, 1e-4, None)
        normalized = (raw_features - self.feature_mean) / self.feature_std

        design = np.vstack([self.expand_features(row) for row in normalized])
        targets = np.asarray(
            [[sample.target_x * screen_size[0], sample.target_y * screen_size[1]] for sample in samples],
            dtype=np.float64,
        )
        weights = np.ones(design.shape[0], dtype=np.float64)
        coefficients = self._fit_weighted_ridge(design, targets, weights)

        # Iteracyjne ważenie Huber ogranicza wpływ pojedynczych błędnych punktów kalibracji.
        for _ in range(4):
            residual = np.linalg.norm((design @ coefficients) - targets, axis=1)
            median_residual = float(np.median(residual))
            scale = max(1.4826 * median_residual, 1.0)
            huber_limit = 1.5 * scale
            weights = np.where(residual <= huber_limit, 1.0, huber_limit / np.maximum(residual, 1e-6))
            coefficients = self._fit_weighted_ridge(design, targets, weights)
        self.coefficients = coefficients

        predictions = design @ coefficients
        error_px = np.linalg.norm(predictions - targets, axis=1)
        self.validation_error_px = float(np.mean(error_px))

        return CalibrationData(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            camera_resolution=[screen_size[0], screen_size[1]],
            feature_names=self.FEATURE_NAMES,
            model_type="polynomial_regression_order_2_normalized_huber_ridge",
            transformation_matrix=coefficients.tolist(),
            target_layout=self.target_layout,
            validation_error_px=self.validation_error_px,
            feature_mean=self.feature_mean.tolist(),
            feature_std=self.feature_std.tolist(),
        )

    def load_from_dict(self, payload: dict[str, Any]) -> None:
        matrix = payload.get("transformation_matrix")
        self.coefficients = np.asarray(matrix, dtype=np.float64) if matrix else None
        self.validation_error_px = payload.get("validation_error_px")
        self.target_layout = payload.get("target_layout", self.default_target_layout())
        mean = payload.get("feature_mean")
        std = payload.get("feature_std")
        model_type = str(payload.get("model_type", ""))
        is_normalized_model = "normalized" in model_type or (
            self.coefficients is not None and self.coefficients.shape[0] == 28
        )
        # Dla nowego modelu z normalizacją brak statystyk jest błędem krytycznym.
        if is_normalized_model and (mean is None or std is None):
            raise ValueError(
                "Calibration file is missing feature_mean/feature_std required by normalized model."
            )
        self.feature_mean = np.asarray(mean, dtype=np.float64) if mean is not None else np.zeros(6, dtype=np.float64)
        self.feature_std = np.asarray(std, dtype=np.float64) if std is not None else np.ones(6, dtype=np.float64)
        if self.feature_mean.shape != (6,) or self.feature_std.shape != (6,):
            raise ValueError("Calibration feature_mean/feature_std must each contain exactly 6 values.")
        self.feature_std = np.clip(self.feature_std, 1e-4, None)

    def map_to_screen(
        self,
        features: list[float],
        screen_size: tuple[int, int],
    ) -> tuple[float, float, bool]:
        if self.coefficients is None:
            return 0.0, 0.0, False
        if self.coefficients.shape[0] == 20:
            # Obsługa starszych kalibracji zapisanych bez normalizacji cech.
            expanded = self._legacy_expand_features(features)
        else:
            normalized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_std
            expanded = self.expand_features(normalized)
        output = expanded @ self.coefficients
        width, height = screen_size
        x_px = float(np.clip(output[0], 0, width))
        y_px = float(np.clip(output[1], 0, height))
        return x_px / max(width, 1), y_px / max(height, 1), True
