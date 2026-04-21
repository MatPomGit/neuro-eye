"""Tracking backend for the modular eye-tracking research application.

This module provides the backend contract expected by ``main.py``:
- ``TrackerController`` as the GUI-facing facade.
- ``CameraWorker`` running in a ``QThread`` to keep the UI responsive.
- A practical blink detector based on eye-aspect-ratio style features.
- A fullscreen 9-point calibration window and session manager.
- YAML calibration persistence and CSV/JSON recording export helpers.

The implementation is intentionally robust to partially missing runtime
dependencies. If OpenCV or MediaPipe are unavailable, the module still exposes a
working interface and degrades gracefully with informative status messages.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml
from PyQt6.QtCore import QMutex, QObject, QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - environment dependent
    mp = None


def _resolve_mediapipe_face_mesh() -> tuple[Any, str | None]:
    """Resolve a MediaPipe Face Mesh implementation across package variants."""
    if mp is None:
        return None, "MediaPipe is not installed."

    candidates: list[tuple[str, str]] = [
        ("top-level solutions API", "mediapipe.solutions.face_mesh"),
        ("python solutions API", "mediapipe.python.solutions.face_mesh"),
        ("legacy package API", "mediapipe.python.solutions.face_mesh"),
    ]

    for label, module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["FaceMesh"])
            if hasattr(module, "FaceMesh"):
                return module, None
        except Exception:
            continue

    return None, (
        "Installed MediaPipe package does not expose Face Mesh / Iris landmarks. "
        "The app will run in preview-only mode. Install a standard MediaPipe build "
        "that includes the solutions API, for example mediapipe 0.10.x."
    )


MP_FACE_MESH, MP_FACE_MESH_ERROR = _resolve_mediapipe_face_mesh()


DEFAULT_CAMERA_INDEX = 0
DEFAULT_CAMERA_RESOLUTION = (1280, 720)
DEFAULT_CAMERA_FPS = 30.0
DATA_DIR = Path("data")
SESSIONS_DIR = DATA_DIR / "sessions"
CALIBRATIONS_DIR = DATA_DIR / "calibrations"


def _candidate_camera_indices(max_indices: int = 6) -> list[int]:
    """Return likely camera indices without aggressively probing invalid devices."""
    if os.name == "posix":
        found: list[int] = []
        for device in sorted(glob.glob("/dev/video*")):
            name = os.path.basename(device)
            suffix = name.replace("video", "", 1)
            if suffix.isdigit():
                found.append(int(suffix))
        if found:
            return found[:max_indices]
    return list(range(max_indices))
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10
LEFT_TEMPLE_INDEX = 234
RIGHT_TEMPLE_INDEX = 454


@dataclass(slots=True)
class FrameMetrics:
    """Single-frame telemetry emitted to the GUI and recorder."""

    timestamp: str
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    pupil_dilation: float = 0.0
    blink_rate: float = 0.0
    tracking_ready: bool = False
    blink_detected: bool = False
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    raw_feature_vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CalibrationSample:
    """Collected sample for one calibration point."""

    target_x: float
    target_y: float
    features: list[float]
    timestamp: str


@dataclass(slots=True)
class CalibrationData:
    """Persisted calibration model."""

    timestamp: str
    camera_resolution: list[int]
    feature_names: list[str]
    model_type: str
    transformation_matrix: list[list[float]]
    target_layout: list[list[float]]
    validation_error_px: Optional[float] = None
    feature_mean: Optional[list[float]] = None
    feature_std: Optional[list[float]] = None


class CalibrationStorage:
    """Read and write calibration data in YAML format."""

    @staticmethod
    def save(path: str | Path, payload: CalibrationData) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(payload), handle, sort_keys=False)

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data


class RecordingWriter:
    """Buffered gaze recording export supporting CSV and JSON."""

    def __init__(self) -> None:
        self._is_recording = False
        self._session_name = ""
        self._export_path: Optional[Path] = None
        self._rows: list[dict[str, Any]] = []

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, session_name: str, export_path: str | None = None) -> str:
        self._session_name = session_name
        if export_path:
            self._export_path = Path(export_path)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._export_path = SESSIONS_DIR / f"{session_name}_{timestamp}.csv"
        self._rows = []
        self._is_recording = True
        return str(self._export_path)

    def append(self, metrics: FrameMetrics) -> None:
        if not self._is_recording:
            return
        row = asdict(metrics)
        row["session_name"] = self._session_name
        self._rows.append(row)

    def stop(self) -> Optional[str]:
        if not self._is_recording or self._export_path is None:
            self._is_recording = False
            return None

        export_path = self._export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = export_path.suffix.lower()

        if suffix == ".json":
            with export_path.open("w", encoding="utf-8") as handle:
                json.dump(self._rows, handle, indent=2)
        else:
            fieldnames = list(self._rows[0].keys()) if self._rows else [
                "session_name",
                "timestamp",
                "gaze_x",
                "gaze_y",
                "pupil_dilation",
                "blink_rate",
                "tracking_ready",
                "blink_detected",
                "head_yaw",
                "head_pitch",
                "raw_feature_vector",
            ]
            with export_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)

        self._is_recording = False
        self._rows = []
        self._session_name = ""
        self._export_path = None
        return str(export_path)


class BlinkDetector:
    """State-based blink detector using EAR-style thresholds with hysteresis."""

    def __init__(
        self,
        close_threshold: float = 0.21,
        open_threshold: float = 0.24,
        min_consecutive_closed: int = 2,
        min_blink_duration_ms: int = 80,
        max_blink_duration_ms: int = 400,
        refractory_ms: int = 120,
    ) -> None:
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_consecutive_closed = min_consecutive_closed
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms
        self.refractory_ms = refractory_ms
        self._closed_frames = 0
        self._is_blink_active = False
        self._blink_start_ms: Optional[float] = None
        self._last_blink_end_ms = 0.0
        self._blink_timestamps_ms: list[float] = []

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray) -> float:
        if points.shape[0] < 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = points[:6]
        vertical_a = np.linalg.norm(p2 - p6)
        vertical_b = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal <= 1e-6:
            return 0.0
        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    def update(self, left_eye: np.ndarray, right_eye: np.ndarray, head_pose_ok: bool) -> tuple[bool, float]:
        now_ms = time.perf_counter() * 1000.0
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if not head_pose_ok:
            self._closed_frames = 0
            if self._is_blink_active:
                self._is_blink_active = False
            return False, self._blink_rate(now_ms)

        blink_detected = False

        if avg_ear < self.close_threshold:
            self._closed_frames += 1
            if (
                not self._is_blink_active
                and self._closed_frames >= self.min_consecutive_closed
                and (now_ms - self._last_blink_end_ms) > self.refractory_ms
            ):
                self._is_blink_active = True
                self._blink_start_ms = now_ms
        elif avg_ear > self.open_threshold:
            self._closed_frames = 0
            if self._is_blink_active and self._blink_start_ms is not None:
                duration_ms = now_ms - self._blink_start_ms
                if self.min_blink_duration_ms <= duration_ms <= self.max_blink_duration_ms:
                    blink_detected = True
                    self._blink_timestamps_ms.append(now_ms)
                self._is_blink_active = False
                self._last_blink_end_ms = now_ms
                self._blink_start_ms = None
        else:
            self._closed_frames = max(0, self._closed_frames - 1)

        return blink_detected, self._blink_rate(now_ms)

    def _blink_rate(self, now_ms: float) -> float:
        window_ms = 60_000.0
        self._blink_timestamps_ms = [ts for ts in self._blink_timestamps_ms if now_ms - ts <= window_ms]
        return float(len(self._blink_timestamps_ms))


class CalibrationModel:
    """Second-order polynomial calibration mapping eye features to screen space."""

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
        f = np.asarray(list(features), dtype=np.float64)
        if f.size != 6:
            raise ValueError("Calibration model expects exactly 6 features.")
        x1, x2, x3, x4, x5, x6 = f
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
        """Rozszerza cechy do wielomianu 2. rzędu z pełnym zestawem interakcji."""
        f = np.asarray(list(features), dtype=np.float64)
        if f.size != 6:
            raise ValueError("Calibration model expects exactly 6 features.")
        terms: list[float] = [1.0]
        terms.extend(f.tolist())
        terms.extend((f * f).tolist())
        # Interakcje między wszystkimi parami cech poprawiają korekcję paralaksy i rotacji głowy.
        for i in range(f.size):
            for j in range(i + 1, f.size):
                terms.append(float(f[i] * f[j]))
        return np.asarray(terms, dtype=np.float64)

    def _fit_weighted_ridge(
        self,
        design: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Dopasowuje model ważoną regresją grzbietową (stabilniejszą numerycznie)."""
        xtwx = design.T @ (weights[:, None] * design)
        reg = np.eye(design.shape[1], dtype=np.float64) * self._ridge_lambda
        reg[0, 0] = 0.0  # Wyraz wolny nie powinien być tłumiony regularizacją.
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

        # Iteracyjna redukcja wpływu odchyleń (np. mrugnięcie w trakcie jednego punktu kalibracji).
        for _ in range(4):
            residual = np.linalg.norm((design @ coefficients) - targets, axis=1)
            median_residual = float(np.median(residual))
            scale = max(1.4826 * median_residual, 1.0)
            huber_limit = 1.5 * scale
            weights = np.where(residual <= huber_limit, 1.0, huber_limit / np.maximum(residual, 1e-6))
            coefficients = self._fit_weighted_ridge(design, targets, weights)
        self.coefficients = coefficients

        predictions = design @ coefficients
        error = np.linalg.norm(predictions - targets, axis=1)
        self.validation_error_px = float(np.mean(error))

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
        # Dla modeli z normalizacją wymagamy statystyk cech, aby uniknąć cichej degradacji jakości.
        if is_normalized_model and (mean is None or std is None):
            raise ValueError(
                "Calibration file is missing feature_mean/feature_std required by normalized model."
            )
        self.feature_mean = np.asarray(mean, dtype=np.float64) if mean is not None else np.zeros(6, dtype=np.float64)
        self.feature_std = np.asarray(std, dtype=np.float64) if std is not None else np.ones(6, dtype=np.float64)
        if self.feature_mean.shape != (6,) or self.feature_std.shape != (6,):
            raise ValueError("Calibration feature_mean/feature_std must each contain exactly 6 values.")
        self.feature_std = np.clip(self.feature_std, 1e-4, None)

    def map_to_screen(self, features: list[float], screen_size: tuple[int, int]) -> tuple[float, float, bool]:
        if self.coefficients is None:
            return 0.0, 0.0, False
        if self.coefficients.shape[0] == 20:
            # Kompatybilność z wcześniejszymi kalibracjami zapisanymi w starym formacie.
            expanded = self._legacy_expand_features(features)
        else:
            normalized = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_std
            expanded = self.expand_features(normalized)
        output = expanded @ self.coefficients
        width, height = screen_size
        x = float(np.clip(output[0], 0, width))
        y = float(np.clip(output[1], 0, height))
        nx = x / max(width, 1)
        ny = y / max(height, 1)
        return nx, ny, True


class KalmanGazeFilter:
    """Filtr Kalmana stabilizujący współrzędne spojrzenia w przestrzeni [0, 1]."""

    def __init__(
        self,
        process_noise: float = 2.5e-3,
        measurement_noise: float = 2.0e-2,
        innovation_gate: float = 6.0,
    ) -> None:
        # Model stanu: [x, y, vx, vy], gdzie x/y to pozycja wzroku,
        # a vx/vy odpowiadają prędkości zmiany spojrzenia.
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._covariance = np.eye(4, dtype=np.float64)
        self._process_noise = float(process_noise)
        self._measurement_noise = float(measurement_noise)
        self._innovation_gate = float(max(innovation_gate, 1.0))
        self._initialized = False

        # Macierz pomiaru: obserwujemy wyłącznie pozycję (x, y).
        self._measurement_matrix = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )

    def reset(self) -> None:
        """Resetuje filtr po utracie twarzy lub kalibracji."""
        self._state.fill(0.0)
        self._covariance = np.eye(4, dtype=np.float64)
        self._initialized = False

    def update(
        self,
        gaze_x: float,
        gaze_y: float,
        dt: float,
        measurement_scale: float = 1.0,
    ) -> tuple[float, float]:
        """Aktualizuje filtr i zwraca wygładzone współrzędne spojrzenia."""
        measurement = np.asarray([[float(gaze_x)], [float(gaze_y)]], dtype=np.float64)
        dt = float(np.clip(dt, 1e-3, 0.2))
        measurement_scale = float(np.clip(measurement_scale, 0.5, 6.0))

        if not self._initialized:
            self._state = np.asarray([[measurement[0, 0]], [measurement[1, 0]], [0.0], [0.0]])
            self._covariance = np.eye(4, dtype=np.float64) * 0.2
            self._initialized = True
            return float(measurement[0, 0]), float(measurement[1, 0])

        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Szum procesu rośnie wraz z krokiem czasowym, dzięki czemu filtr
        # lepiej reaguje na szybkie ruchy gałki ocznej.
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process_cov = self._process_noise * np.asarray(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        measurement_cov = np.eye(2, dtype=np.float64) * (self._measurement_noise * measurement_scale)

        predicted_state = transition @ self._state
        predicted_covariance = transition @ self._covariance @ transition.T + process_cov

        innovation = measurement - (self._measurement_matrix @ predicted_state)
        innovation_covariance = (
            self._measurement_matrix @ predicted_covariance @ self._measurement_matrix.T
            + measurement_cov
        )

        # Bramka innowacji ogranicza wpływ pojedynczych odskoków punktu spojrzenia.
        try:
            mahalanobis = float(
                innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation
            )
        except Exception:
            mahalanobis = 0.0
        if mahalanobis > self._innovation_gate:
            innovation *= 0.35

        kalman_gain = (
            predicted_covariance
            @ self._measurement_matrix.T
            @ np.linalg.pinv(innovation_covariance)
        )

        self._state = predicted_state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self._covariance = (identity - kalman_gain @ self._measurement_matrix) @ predicted_covariance

        filtered_x = float(np.clip(self._state[0, 0], 0.0, 1.0))
        filtered_y = float(np.clip(self._state[1, 0], 0.0, 1.0))
        self._state[0, 0] = filtered_x
        self._state[1, 0] = filtered_y
        return filtered_x, filtered_y


class CalibrationWindow(QWidget):
    """Fullscreen target presenter for the 9-point calibration routine."""

    calibration_finished = pyqtSignal(list)
    status_message = pyqtSignal(str)

    def __init__(self, controller: "TrackerController") -> None:
        super().__init__()
        self.controller = controller
        self.points = CalibrationModel.default_target_layout()
        self.current_index = -1
        self.collecting = False
        self.collected_samples: list[CalibrationSample] = []
        self._point_samples: list[list[float]] = []
        self._settle_timer = QTimer(self)
        self._collect_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._collect_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._begin_collection_phase)
        self._collect_timer.timeout.connect(self._finalize_current_point)
        self.setWindowTitle("Calibration")
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.BlankCursor)

    def start(self) -> None:
        self.collected_samples = []
        self.current_index = -1
        self.status_message.emit("Calibration started. Follow the fullscreen targets.")
        self.showFullScreen()
        self._advance_to_next_point()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.status_message.emit("Calibration cancelled.")
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        self._settle_timer.stop()
        self._collect_timer.stop()
        self.controller.unregister_calibration_window(self)
        super().closeEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        if 0 <= self.current_index < len(self.points):
            px, py = self.points[self.current_index]
            x = int(self.width() * px)
            y = int(self.height() * py)
            painter.setPen(QPen(QColor("white"), 3))
            radius = 18
            painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
            painter.drawLine(x - 28, y, x + 28, y)
            painter.drawLine(x, y - 28, x, y + 28)

            progress_rect = QRectF(20.0, self.height() - 40.0, max(0.0, (self.current_index + 1) / len(self.points) * (self.width() - 40.0)), 12.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2563EB"))
            painter.drawRoundedRect(progress_rect, 6.0, 6.0)

    def _advance_to_next_point(self) -> None:
        self.current_index += 1
        if self.current_index >= len(self.points):
            self.status_message.emit("Calibration sample collection complete.")
            self.calibration_finished.emit(self.collected_samples)
            self.close()
            return

        self.collecting = False
        self._point_samples = []
        self.status_message.emit(f"Calibration point {self.current_index + 1}/9")
        self.update()
        self._settle_timer.start(650)

    def _begin_collection_phase(self) -> None:
        self.collecting = True
        self.status_message.emit(f"Collecting point {self.current_index + 1}/9")
        self._collect_timer.start(1000)

    def push_feature_vector(self, features: list[float], quality_ok: bool) -> None:
        if not self.collecting or not quality_ok:
            return
        self._point_samples.append(list(features))

    def _finalize_current_point(self) -> None:
        self.collecting = False
        if not self._point_samples:
            self.status_message.emit(
                f"Point {self.current_index + 1}/9 had no valid samples. Retrying."
            )
            self._settle_timer.start(500)
            return

        features = np.median(np.asarray(self._point_samples, dtype=np.float64), axis=0).tolist()
        target_x, target_y = self.points[self.current_index]
        self.collected_samples.append(
            CalibrationSample(
                target_x=target_x,
                target_y=target_y,
                features=features,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )
        self._advance_to_next_point()


class CameraWorker(QObject):
    """Background worker that captures frames and computes gaze telemetry."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        camera_resolution: tuple[int, int] = DEFAULT_CAMERA_RESOLUTION,
        camera_fps: float = DEFAULT_CAMERA_FPS,
        tracking_stride: int = 1,
    ) -> None:
        super().__init__()
        self.camera_index = int(camera_index)
        self.camera_resolution = (int(camera_resolution[0]), int(camera_resolution[1]))
        self.camera_fps = float(camera_fps)
        self.tracking_stride = max(1, min(3, int(tracking_stride)))
        self._running = False
        self._mutex = QMutex()
        self.blink_detector = BlinkDetector()
        self._screen_size = self._current_screen_size()
        self._calibration_model = CalibrationModel()
        self._gaze_filter = KalmanGazeFilter()
        self._last_metrics_timestamp: Optional[float] = None
        self._gaze_prefilter_window: deque[tuple[float, float]] = deque(maxlen=5)

    def stop(self) -> None:
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def update_calibration_model(self, model: CalibrationModel) -> None:
        self._calibration_model = model

    def run(self) -> None:
        self._mutex.lock()
        self._running = True
        self._mutex.unlock()

        if cv2 is None:
            self.status_changed.emit("OpenCV not available. Camera stream cannot start.")
            self.finished.emit()
            return

        cap = self._open_capture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

        if not cap.isOpened():
            self.status_changed.emit("Unable to open camera device.")
            self.finished.emit()
            return

        if MP_FACE_MESH is None:
            preview_msg = MP_FACE_MESH_ERROR or (
                "MediaPipe Face Mesh not available. Running preview-only camera mode."
            )
            self.status_changed.emit(preview_msg)
            self._run_preview_only(cap)
            return

        try:
            face_mesh = MP_FACE_MESH.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        except Exception as exc:
            self.status_changed.emit(f"Failed to initialize MediaPipe Face Mesh: {exc}")
            self._run_preview_only(cap)
            return

        self.status_changed.emit(
            "Camera worker started "
            f"(camera={self.camera_index}, resolution={self.camera_resolution[0]}x{self.camera_resolution[1]}, "
            f"fps={self.camera_fps:.0f}, stride={self.tracking_stride})."
        )
        frame_counter = 0
        last_results: Any = None
        last_metrics: Optional[FrameMetrics] = None
        try:
            while self._is_running():
                ok, frame = cap.read()
                if not ok:
                    self.status_changed.emit("Camera frame read failed.")
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_counter += 1
                should_process = ((frame_counter - 1) % self.tracking_stride) == 0
                if should_process or last_results is None:
                    try:
                        last_results = face_mesh.process(image_rgb)
                        last_metrics = self._build_metrics(image_rgb, last_results)
                    except Exception as exc:
                        self.status_changed.emit(f"MediaPipe processing error: {exc}")
                        last_results = None
                        last_metrics = FrameMetrics(
                            timestamp=datetime.now().isoformat(timespec="milliseconds")
                        )
                    metrics = last_metrics
                else:
                    if last_metrics is None:
                        metrics = FrameMetrics(
                            timestamp=datetime.now().isoformat(timespec="milliseconds")
                        )
                    else:
                        metrics = replace(
                            last_metrics,
                            timestamp=datetime.now().isoformat(timespec="milliseconds"),
                        )
                rendered = self._draw_overlay(image_rgb.copy(), last_results, metrics)
                self.frame_ready.emit(self._to_qimage(rendered))
                self.metrics_ready.emit(asdict(metrics))
        except Exception as exc:
            self.status_changed.emit(f"Camera worker crashed: {exc}")
        finally:
            try:
                face_mesh.close()
            except Exception:
                pass
            cap.release()
            self.status_changed.emit("Camera worker stopped.")
            self.finished.emit()

    @staticmethod
    def _open_capture(camera_index: int) -> Any:
        if cv2 is None:
            return None
        if os.name == "posix" and hasattr(cv2, "CAP_V4L2"):
            return cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)
        return cv2.VideoCapture(int(camera_index))

    def _run_preview_only(self, cap: Any) -> None:
        try:
            while self._is_running():
                ok, frame = cap.read()
                if not ok:
                    self.status_changed.emit("Camera frame read failed.")
                    break
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                metrics = FrameMetrics(timestamp=datetime.now().isoformat(timespec="milliseconds"))
                self.frame_ready.emit(self._to_qimage(image_rgb))
                self.metrics_ready.emit(asdict(metrics))
        finally:
            cap.release()
            self.finished.emit()

    def _build_metrics(self, image_rgb: np.ndarray, results: Any) -> FrameMetrics:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        now_perf = time.perf_counter()
        metrics = FrameMetrics(timestamp=timestamp)
        if not results.multi_face_landmarks:
            self._gaze_filter.reset()
            self._last_metrics_timestamp = None
            self._gaze_prefilter_window.clear()
            return metrics

        face_landmarks = results.multi_face_landmarks[0]
        points = self._landmarks_to_pixels(face_landmarks, image_rgb.shape[1], image_rgb.shape[0])
        left_eye = points[LEFT_EYE_INDICES]
        right_eye = points[RIGHT_EYE_INDICES]
        left_iris = points[LEFT_IRIS_INDICES]
        right_iris = points[RIGHT_IRIS_INDICES]

        head_yaw, head_pitch = self._estimate_head_pose(points)
        head_pose_ok = abs(head_yaw) < 18.0 and abs(head_pitch) < 18.0
        blink_detected, blink_rate = self.blink_detector.update(left_eye, right_eye, head_pose_ok)
        features = self._extract_feature_vector(left_eye, right_eye, left_iris, right_iris, head_yaw, head_pitch)
        gaze_x, gaze_y, tracking_ready = self._calibration_model.map_to_screen(features, self._screen_size)
        if tracking_ready:
            # Prefiltr medianowy usuwa krótkie skoki charakterystyczne dla webcam + MediaPipe.
            self._gaze_prefilter_window.append((gaze_x, gaze_y))
            stacked = np.asarray(self._gaze_prefilter_window, dtype=np.float64)
            median_x = float(np.median(stacked[:, 0]))
            median_y = float(np.median(stacked[:, 1]))

            # Adaptacyjny szum pomiaru: im gorsza jakość geometrii twarzy,
            # tym słabiej ufamy bieżącemu pomiarowi.
            pose_penalty = min(abs(head_yaw) / 20.0 + abs(head_pitch) / 20.0, 2.0)
            blink_penalty = 1.25 if blink_detected else 0.0
            measurement_scale = 1.0 + pose_penalty + blink_penalty

            if self._last_metrics_timestamp is None:
                dt = 1.0 / max(self.camera_fps, 1.0)
            else:
                dt = now_perf - self._last_metrics_timestamp
            gaze_x, gaze_y = self._gaze_filter.update(
                median_x,
                median_y,
                dt,
                measurement_scale=measurement_scale,
            )
            self._last_metrics_timestamp = now_perf
        else:
            self._gaze_filter.reset()
            self._last_metrics_timestamp = None
            self._gaze_prefilter_window.clear()
        pupil_dilation = self._estimate_pupil_dilation(left_iris, right_iris)

        return FrameMetrics(
            timestamp=timestamp,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            pupil_dilation=pupil_dilation,
            blink_rate=blink_rate,
            tracking_ready=tracking_ready,
            blink_detected=blink_detected,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            raw_feature_vector=features,
        )

    @staticmethod
    def _landmarks_to_pixels(face_landmarks: Any, width: int, height: int) -> np.ndarray:
        coords = []
        for landmark in face_landmarks.landmark:
            coords.append([landmark.x * width, landmark.y * height])
        return np.asarray(coords, dtype=np.float64)

    @staticmethod
    def _estimate_head_pose(points: np.ndarray) -> tuple[float, float]:
        left = points[LEFT_TEMPLE_INDEX]
        right = points[RIGHT_TEMPLE_INDEX]
        nose = points[NOSE_TIP_INDEX]
        forehead = points[FOREHEAD_INDEX]
        chin = points[CHIN_INDEX]

        head_width = np.linalg.norm(right - left)
        head_height = np.linalg.norm(chin - forehead)
        if head_width <= 1e-6 or head_height <= 1e-6:
            return 0.0, 0.0

        center_x = (left[0] + right[0]) / 2.0
        center_y = (forehead[1] + chin[1]) / 2.0
        yaw = ((nose[0] - center_x) / head_width) * 90.0
        pitch = ((nose[1] - center_y) / head_height) * 90.0
        return float(yaw), float(pitch)

    @staticmethod
    def _extract_feature_vector(
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        left_iris: np.ndarray,
        right_iris: np.ndarray,
        head_yaw: float,
        head_pitch: float,
    ) -> list[float]:
        def iris_relative(eye: np.ndarray, iris: np.ndarray) -> tuple[float, float]:
            min_xy = eye.min(axis=0)
            max_xy = eye.max(axis=0)
            iris_center = iris.mean(axis=0)
            width = max(max_xy[0] - min_xy[0], 1e-6)
            height = max(max_xy[1] - min_xy[1], 1e-6)
            rel_x = float((iris_center[0] - min_xy[0]) / width)
            rel_y = float((iris_center[1] - min_xy[1]) / height)
            return rel_x, rel_y

        l_ix, l_iy = iris_relative(left_eye, left_iris)
        r_ix, r_iy = iris_relative(right_eye, right_iris)
        return [l_ix, l_iy, r_ix, r_iy, float(head_yaw), float(head_pitch)]

    @staticmethod
    def _estimate_pupil_dilation(left_iris: np.ndarray, right_iris: np.ndarray) -> float:
        def iris_radius(iris: np.ndarray) -> float:
            center = iris.mean(axis=0)
            return float(np.mean(np.linalg.norm(iris - center, axis=1)))

        return float((iris_radius(left_iris) + iris_radius(right_iris)) / 2.0)

    @staticmethod
    def _draw_overlay(image_rgb: np.ndarray, results: Any, metrics: FrameMetrics) -> np.ndarray:
        if cv2 is None:
            return image_rgb
        output = image_rgb
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            height, width = output.shape[:2]
            for index in LEFT_EYE_INDICES + RIGHT_EYE_INDICES + LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
                landmark = face_landmarks.landmark[index]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

        if metrics.tracking_ready:
            gx = int(metrics.gaze_x * output.shape[1])
            gy = int(metrics.gaze_y * output.shape[0])
            cv2.drawMarker(output, (gx, gy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        return output

    @staticmethod
    def _to_qimage(image_rgb: np.ndarray) -> QImage:
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimage.copy()

    @staticmethod
    def _current_screen_size() -> tuple[int, int]:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return 1920, 1080
        geometry = screen.availableGeometry()
        return geometry.width(), geometry.height()

    def _is_running(self) -> bool:
        self._mutex.lock()
        value = self._running
        self._mutex.unlock()
        return value


class TrackerController(QWidget):
    """GUI-facing orchestration layer for tracking, calibration, and recording."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    recording_state_changed = pyqtSignal(bool)
    calibration_loaded = pyqtSignal(dict)
    camera_index_changed = pyqtSignal(int)
    camera_resolution_changed = pyqtSignal(tuple)
    camera_fps_changed = pyqtSignal(float)
    tracking_stride_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[CameraWorker] = None
        self._recording = RecordingWriter()
        self._calibration_model = CalibrationModel()
        self._calibration_payload: Optional[dict[str, Any]] = None
        self._calibration_window: Optional[CalibrationWindow] = None
        self._last_metrics: Optional[FrameMetrics] = None
        self._camera_index = DEFAULT_CAMERA_INDEX
        self._camera_resolution = tuple(DEFAULT_CAMERA_RESOLUTION)
        self._camera_fps = float(DEFAULT_CAMERA_FPS)
        self._tracking_stride = 1
        self._last_camera_scan: list[dict[str, Any]] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self.status_changed.emit("Tracking is already running.")
            return

        self._thread = QThread(self)
        self._worker = CameraWorker(
            camera_index=self._camera_index,
            camera_resolution=self._camera_resolution,
            camera_fps=self._camera_fps,
            tracking_stride=self._tracking_stride,
        )
        self._worker.update_calibration_model(self._calibration_model)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.metrics_ready.connect(self._on_metrics_ready)
        self._worker.status_changed.connect(self.status_changed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread_references)
        self._thread.start()
        self.status_changed.emit("Tracking requested.")

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self.status_changed.emit("Tracking stop requested.")


    def list_available_cameras(self, max_indices: int = 6) -> list[dict[str, Any]]:
        cameras: list[dict[str, Any]] = []
        if cv2 is None:
            fallback = [{"index": self._camera_index, "label": f"Camera {self._camera_index}"}]
            self._last_camera_scan = fallback
            return fallback

        for index in _candidate_camera_indices(max_indices):
            cap = self._probe_camera(index)
            try:
                if cap is None or not cap.isOpened():
                    continue
                ok, _ = cap.read()
                if ok:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    suffix = f" ({width}x{height})" if width > 0 and height > 0 else ""
                    cameras.append({"index": index, "label": f"Camera {index}{suffix}"})
            finally:
                if cap is not None:
                    cap.release()

        if not cameras:
            cameras = [{"index": self._camera_index, "label": f"Camera {self._camera_index}"}]
        self._last_camera_scan = cameras
        return cameras

    @staticmethod
    def _probe_camera(index: int) -> Any:
        if cv2 is None:
            return None
        if os.name == "posix" and hasattr(cv2, "CAP_V4L2"):
            return cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        return cv2.VideoCapture(int(index))

    def get_camera_index(self) -> int:
        return self._camera_index

    def set_camera_index(self, camera_index: int) -> None:
        camera_index = int(camera_index)
        if camera_index == self._camera_index:
            return

        was_running = self._thread is not None and self._thread.isRunning()
        if was_running:
            self.stop()

        self._camera_index = camera_index
        self.camera_index_changed.emit(camera_index)
        self.status_changed.emit(f"Selected camera index: {camera_index}")

        if was_running:
            self.start()

    def get_camera_resolution(self) -> tuple[int, int]:
        return self._camera_resolution

    def set_camera_resolution(self, resolution: tuple[int, int]) -> None:
        parsed = (int(resolution[0]), int(resolution[1]))
        if parsed == self._camera_resolution:
            return
        was_running = self._thread is not None and self._thread.isRunning()
        if was_running:
            self.stop()
        self._camera_resolution = parsed
        self.camera_resolution_changed.emit(parsed)
        self.status_changed.emit(f"Selected camera resolution: {parsed[0]}x{parsed[1]}")
        if was_running:
            self.start()

    def get_camera_fps(self) -> float:
        return self._camera_fps

    def set_camera_fps(self, fps: float) -> None:
        parsed = float(fps)
        if math.isclose(parsed, self._camera_fps, rel_tol=0.0, abs_tol=0.01):
            return
        was_running = self._thread is not None and self._thread.isRunning()
        if was_running:
            self.stop()
        self._camera_fps = parsed
        self.camera_fps_changed.emit(parsed)
        self.status_changed.emit(f"Selected target FPS: {parsed:.0f}")
        if was_running:
            self.start()

    def get_tracking_stride(self) -> int:
        return self._tracking_stride

    def set_tracking_stride(self, stride: int) -> None:
        parsed = max(1, min(3, int(stride)))
        if parsed == self._tracking_stride:
            return
        was_running = self._thread is not None and self._thread.isRunning()
        if was_running:
            self.stop()
        self._tracking_stride = parsed
        self.tracking_stride_changed.emit(parsed)
        self.status_changed.emit(f"MediaPipe inference frequency: every {parsed} frame(s)")
        if was_running:
            self.start()

    def begin_calibration(self) -> None:
        if self._calibration_window is not None:
            self.status_changed.emit("Calibration is already active.")
            return
        self._calibration_window = CalibrationWindow(self)
        self._calibration_window.status_message.connect(self.status_changed)
        self._calibration_window.calibration_finished.connect(self._finalize_calibration)
        self._calibration_window.start()

    def unregister_calibration_window(self, window: CalibrationWindow) -> None:
        if self._calibration_window is window:
            self._calibration_window = None

    def save_calibration(self, path: str) -> None:
        if self._calibration_payload is None:
            raise ValueError("No calibration model is available to save.")
        model_type = str(self._calibration_payload.get("model_type", ""))
        matrix = self._calibration_payload.get("transformation_matrix") or []
        coeff_rows = len(matrix) if isinstance(matrix, list) else 0
        is_normalized_model = "normalized" in model_type or coeff_rows == 28
        mean = self._calibration_payload.get("feature_mean")
        std = self._calibration_payload.get("feature_std")
        # Chronimy zapis przed utrwaleniem niekompletnego modelu z normalizacją.
        if is_normalized_model and (mean is None or std is None):
            raise ValueError(
                "Cannot save normalized calibration without feature_mean/feature_std."
            )
        # Zachowujemy komplet parametrów modelu, aby po wczytaniu predykcja była identyczna.
        payload = CalibrationData(
            timestamp=self._calibration_payload["timestamp"],
            camera_resolution=list(self._calibration_payload["camera_resolution"]),
            feature_names=list(self._calibration_payload["feature_names"]),
            model_type=self._calibration_payload["model_type"],
            transformation_matrix=list(self._calibration_payload["transformation_matrix"]),
            target_layout=list(self._calibration_payload["target_layout"]),
            validation_error_px=self._calibration_payload.get("validation_error_px"),
            feature_mean=list(mean) if mean is not None else None,
            feature_std=list(std) if std is not None else None,
        )
        CalibrationStorage.save(path, payload)
        self.status_changed.emit(f"Calibration saved to {path}")

    def load_calibration(self, path: str) -> dict[str, Any]:
        payload = CalibrationStorage.load(path)
        self._calibration_model.load_from_dict(payload)
        self._calibration_payload = payload
        if self._worker is not None:
            self._worker.update_calibration_model(self._calibration_model)
        self.calibration_loaded.emit(payload)
        self.status_changed.emit(f"Calibration loaded from {path}")
        return payload

    def start_recording(self, session_name: str, export_path: Optional[str] = None) -> None:
        path = self._recording.start(session_name, export_path)
        self.recording_state_changed.emit(True)
        self.status_changed.emit(f"Recording started: {path}")

    def stop_recording(self) -> None:
        path = self._recording.stop()
        self.recording_state_changed.emit(False)
        if path:
            self.status_changed.emit(f"Recording saved: {path}")
        else:
            self.status_changed.emit("Recording stopped.")

    def shutdown(self) -> None:
        if self._recording.is_recording:
            self.stop_recording()
        if self._calibration_window is not None:
            self._calibration_window.close()
        self.stop()

    def _on_metrics_ready(self, payload: dict[str, Any]) -> None:
        metrics = FrameMetrics(**payload)
        self._last_metrics = metrics
        self.metrics_ready.emit(payload)
        if self._recording.is_recording:
            self._recording.append(metrics)
        if self._calibration_window is not None:
            quality_ok = (
                not metrics.blink_detected
                and len(metrics.raw_feature_vector) == 6
                and abs(metrics.head_yaw) < 18.0
                and abs(metrics.head_pitch) < 18.0
            )
            self._calibration_window.push_feature_vector(metrics.raw_feature_vector, quality_ok)

    def _finalize_calibration(self, samples: list[CalibrationSample]) -> None:
        """Finalizuje kalibrację, aktualizuje model i zapisuje wynik do pliku YAML."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            screen_size = (1920, 1080)
        else:
            geometry = screen.availableGeometry()
            screen_size = (geometry.width(), geometry.height())

        payload = self._calibration_model.fit(samples, screen_size)
        self._calibration_payload = asdict(payload)
        # Każdą zakończoną kalibrację automatycznie archiwizujemy lokalnie,
        # aby użytkownik nie utracił modelu po przypadkowym zamknięciu aplikacji.
        autosave_path = self._build_autosave_calibration_path(payload.timestamp)
        CalibrationStorage.save(autosave_path, payload)
        if self._worker is not None:
            self._worker.update_calibration_model(self._calibration_model)
        self.calibration_loaded.emit(self._calibration_payload)
        self.status_changed.emit(
            "Calibration completed. "
            f"Mean validation error: {payload.validation_error_px:.1f}px. "
            f"Saved to: {autosave_path}"
        )

    def _clear_thread_references(self) -> None:
        self._thread = None
        self._worker = None

    @staticmethod
    def _build_autosave_calibration_path(timestamp: str) -> Path:
        """Buduje ścieżkę autozapisu kalibracji w katalogu data/calibrations."""
        safe_timestamp = timestamp.replace(":", "").replace(".", "").replace("T", "_")
        return CALIBRATIONS_DIR / f"calibration_{safe_timestamp}.yaml"


__all__ = [
    "CalibrationData",
    "CalibrationStorage",
    "TrackerController",
    "CameraWorker",
    "CalibrationModel",
    "BlinkDetector",
]
