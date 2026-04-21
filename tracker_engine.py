"""Runtime tracking backend for the eye-tracking research application.

This module keeps the GUI-facing contract small and stable:
- ``TrackerController`` is the facade used by ``main.py``.
- ``CameraThread`` handles webcam capture off the UI thread.
- ``BlinkDetector`` implements a practical EAR-based blink detector.
- Calibration, fullscreen presentation, and persistence are imported from the
  dedicated project modules instead of being duplicated here.

The implementation degrades safely when optional dependencies such as OpenCV or
MediaPipe are unavailable.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QWidget

from calibration import CalibrationData, CalibrationModel, CalibrationSample
from data_io import CalibrationStorage, RecordingWriter
from widgets.calibration_window import CalibrationWindow

try:  # pragma: no cover - optional runtime dependency
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:  # pragma: no cover - optional runtime dependency
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency
    mp = None


LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_BOX = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE_BOX = {"outer": 263, "inner": 362, "top": 386, "bottom": 374}
HEAD_POSE_REF = {"left": 33, "right": 263, "nose": 1, "mid": 168}


@dataclass(slots=True)
class BlinkEvent:
    """One completed blink event."""

    start_ts: float
    end_ts: float
    duration_ms: float
    min_ear: float


class BlinkDetector:
    """Practical blink detector based on eye aspect ratio with hysteresis."""

    def __init__(
        self,
        fps_hint: float = 30.0,
        close_threshold: float = 0.20,
        open_threshold: float = 0.24,
        min_closed_frames: int = 2,
        refractory_ms: float = 120.0,
    ) -> None:
        self.fps_hint = fps_hint
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_closed_frames = min_closed_frames
        self.refractory_ms = refractory_ms
        self._state = "OPEN"
        self._closed_frames = 0
        self._blink_start_ts: Optional[float] = None
        self._last_blink_end_ts = 0.0
        self._min_ear = 1.0
        self._events: deque[float] = deque()
        self._blink_active = False

    @property
    def blink_active(self) -> bool:
        return self._blink_active

    def blink_rate_bpm(self, now_ts: float) -> float:
        while self._events and (now_ts - self._events[0]) > 60.0:
            self._events.popleft()
        return float(len(self._events))

    def update(self, avg_ear: float, quality_ok: bool, now_ts: float) -> Optional[BlinkEvent]:
        self._blink_active = False

        if not quality_ok:
            self._state = "OPEN"
            self._closed_frames = 0
            self._blink_start_ts = None
            self._min_ear = 1.0
            return None

        if self._state == "OPEN":
            if avg_ear < self.close_threshold and (now_ts - self._last_blink_end_ts) * 1000.0 >= self.refractory_ms:
                self._closed_frames += 1
                self._min_ear = min(self._min_ear, avg_ear)
                if self._closed_frames >= self.min_closed_frames:
                    self._state = "CLOSED"
                    self._blink_start_ts = now_ts
                    self._blink_active = True
            else:
                self._closed_frames = 0
                self._min_ear = 1.0
            return None

        if self._state == "CLOSED":
            self._blink_active = True
            self._min_ear = min(self._min_ear, avg_ear)
            if avg_ear > self.open_threshold:
                start_ts = self._blink_start_ts or now_ts
                duration_ms = max(0.0, (now_ts - start_ts) * 1000.0)
                self._state = "OPEN"
                self._closed_frames = 0
                self._blink_start_ts = None
                self._last_blink_end_ts = now_ts
                self._events.append(now_ts)
                event = BlinkEvent(
                    start_ts=start_ts,
                    end_ts=now_ts,
                    duration_ms=duration_ms,
                    min_ear=self._min_ear,
                )
                self._min_ear = 1.0
                self._blink_active = False
                return event
            return None

        self._state = "OPEN"
        self._closed_frames = 0
        self._blink_start_ts = None
        self._min_ear = 1.0
        return None


class CameraThread(QThread):
    """Webcam processing thread with optional MediaPipe face mesh support."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    feature_vector_ready = pyqtSignal(list, bool)
    status_changed = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._screen_size = (1920, 1080)
        self._calibration_model = CalibrationModel()
        self._blink_detector = BlinkDetector()
        self._last_valid_features: Optional[list[float]] = None
        self._last_metrics: dict[str, Any] = {}

    def stop(self) -> None:
        self._running = False
        self.wait(1500)

    def set_screen_size(self, width: int, height: int) -> None:
        self._screen_size = (max(width, 1), max(height, 1))

    def set_calibration_payload(self, payload: dict[str, Any]) -> None:
        self._calibration_model.load_from_dict(payload)

    def run(self) -> None:  # pragma: no cover - runtime loop
        self._running = True

        if cv2 is None:
            self.status_changed.emit("OpenCV not available. Running without camera feed.")
            self._run_placeholder_loop()
            return

        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.status_changed.emit("Could not open camera. Running without live feed.")
            self._run_placeholder_loop()
            return

        self.status_changed.emit("Camera thread started.")

        face_mesh = None
        if mp is not None:
            try:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                face_mesh = None

        try:
            while self._running:
                ok, frame_bgr = capture.read()
                if not ok:
                    self.status_changed.emit("Camera frame read failed.")
                    time.sleep(0.05)
                    continue

                frame_bgr = cv2.flip(frame_bgr, 1)
                metrics, annotated_rgb = self._process_frame(frame_bgr, face_mesh)
                self._last_metrics = metrics
                self.metrics_ready.emit(metrics)
                self.frame_ready.emit(self._rgb_to_qimage(annotated_rgb))
                self.feature_vector_ready.emit(
                    list(metrics.get("feature_vector", [])),
                    bool(metrics.get("quality_ok", False)),
                )
                self.msleep(10)
        finally:
            capture.release()
            if face_mesh is not None:
                face_mesh.close()
            self.status_changed.emit("Camera thread stopped.")

    def _run_placeholder_loop(self) -> None:  # pragma: no cover - runtime loop
        while self._running:
            image = QImage(1280, 720, QImage.Format.Format_RGB888)
            image.fill(Qt.GlobalColor.black)
            self.frame_ready.emit(image)
            metrics = {
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "pupil_dilation": 0.0,
                "blink_rate": self._blink_detector.blink_rate_bpm(time.time()),
                "tracking_ready": False,
                "blink_detected": False,
                "quality_ok": False,
                "feature_vector": [],
                "head_pose_ok": False,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            }
            self.metrics_ready.emit(metrics)
            self.feature_vector_ready.emit([], False)
            self.msleep(100)

    def _process_frame(self, frame_bgr: np.ndarray, face_mesh: Any) -> tuple[dict[str, Any], np.ndarray]:
        now_ts = time.time()
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        metrics: dict[str, Any] = {
            "gaze_x": 0.0,
            "gaze_y": 0.0,
            "pupil_dilation": 0.0,
            "blink_rate": self._blink_detector.blink_rate_bpm(now_ts),
            "tracking_ready": False,
            "blink_detected": False,
            "quality_ok": False,
            "feature_vector": [],
            "head_pose_ok": False,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        }

        if face_mesh is None:
            cv2.putText(
                frame_rgb,
                "MediaPipe unavailable - camera preview only",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return metrics, frame_rgb

        results = face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            cv2.putText(
                frame_rgb,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return metrics, frame_rgb

        landmarks = results.multi_face_landmarks[0].landmark
        points = np.asarray([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float64)

        left_ear = self._compute_ear(points, LEFT_EYE_EAR)
        right_ear = self._compute_ear(points, RIGHT_EYE_EAR)
        avg_ear = (left_ear + right_ear) * 0.5
        head_yaw, head_pitch, head_pose_ok = self._estimate_head_pose(points, width, height)
        blink_event = self._blink_detector.update(avg_ear, head_pose_ok, now_ts)

        feature_vector, pupil_dilation, quality_ok = self._extract_feature_vector(points)
        tracking_ready = False
        gaze_x = 0.0
        gaze_y = 0.0
        if quality_ok:
            self._last_valid_features = feature_vector
            gaze_x, gaze_y, tracking_ready = self._calibration_model.map_to_screen(
                feature_vector,
                self._screen_size,
            )

        metrics.update(
            {
                "gaze_x": gaze_x,
                "gaze_y": gaze_y,
                "pupil_dilation": pupil_dilation,
                "blink_rate": self._blink_detector.blink_rate_bpm(now_ts),
                "tracking_ready": tracking_ready,
                "blink_detected": self._blink_detector.blink_active,
                "quality_ok": quality_ok and head_pose_ok and not self._blink_detector.blink_active,
                "feature_vector": feature_vector,
                "head_pose_ok": head_pose_ok,
                "head_yaw": head_yaw,
                "head_pitch": head_pitch,
                "left_ear": left_ear,
                "right_ear": right_ear,
                "avg_ear": avg_ear,
                "blink_duration_ms": blink_event.duration_ms if blink_event else 0.0,
            }
        )

        self._annotate_frame(
            frame_rgb=frame_rgb,
            points=points,
            left_ear=left_ear,
            right_ear=right_ear,
            quality_ok=bool(metrics["quality_ok"]),
            gaze_ready=tracking_ready,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
        return metrics, frame_rgb

    @staticmethod
    def _rgb_to_qimage(frame_rgb: np.ndarray) -> QImage:
        height, width, channels = frame_rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            frame_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return image.copy()

    @staticmethod
    def _compute_ear(points: np.ndarray, idx: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[i] for i in idx]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = max(np.linalg.norm(p1 - p4), 1e-6)
        return float(vertical / (2.0 * horizontal))

    @staticmethod
    def _mean_point(points: np.ndarray, indices: list[int]) -> np.ndarray:
        return np.mean(points[indices], axis=0)

    def _extract_feature_vector(self, points: np.ndarray) -> tuple[list[float], float, bool]:
        left_iris = self._mean_point(points, LEFT_IRIS)
        right_iris = self._mean_point(points, RIGHT_IRIS)

        left_width = max(np.linalg.norm(points[LEFT_EYE_BOX["outer"]] - points[LEFT_EYE_BOX["inner"]]), 1e-6)
        left_height = max(np.linalg.norm(points[LEFT_EYE_BOX["top"]] - points[LEFT_EYE_BOX["bottom"]]), 1e-6)
        right_width = max(np.linalg.norm(points[RIGHT_EYE_BOX["outer"]] - points[RIGHT_EYE_BOX["inner"]]), 1e-6)
        right_height = max(np.linalg.norm(points[RIGHT_EYE_BOX["top"]] - points[RIGHT_EYE_BOX["bottom"]]), 1e-6)

        left_outer = points[LEFT_EYE_BOX["outer"]]
        left_top = points[LEFT_EYE_BOX["top"]]
        right_outer = points[RIGHT_EYE_BOX["outer"]]
        right_top = points[RIGHT_EYE_BOX["top"]]

        left_iris_x = float((left_iris[0] - left_outer[0]) / left_width)
        left_iris_y = float((left_iris[1] - left_top[1]) / left_height)
        right_iris_x = float((right_iris[0] - right_outer[0]) / right_width)
        right_iris_y = float((right_iris[1] - right_top[1]) / right_height)

        head_yaw, head_pitch, head_pose_ok = self._estimate_head_pose(points, 1, 1)
        pupil_dilation = float((left_width / left_height + right_width / right_height) * 0.5)

        feature_vector = [
            left_iris_x,
            left_iris_y,
            right_iris_x,
            right_iris_y,
            head_yaw,
            head_pitch,
        ]

        finite_ok = all(math.isfinite(value) for value in feature_vector)
        eye_box_ok = left_width > 3.0 and right_width > 3.0 and left_height > 1.0 and right_height > 1.0
        iris_range_ok = all(-1.0 <= value <= 3.0 for value in feature_vector[:4])
        quality_ok = finite_ok and eye_box_ok and iris_range_ok and head_pose_ok
        return feature_vector, pupil_dilation, quality_ok

    @staticmethod
    def _estimate_head_pose(points: np.ndarray, width: int, height: int) -> tuple[float, float, bool]:
        left = points[HEAD_POSE_REF["left"]]
        right = points[HEAD_POSE_REF["right"]]
        nose = points[HEAD_POSE_REF["nose"]]
        mid = points[HEAD_POSE_REF["mid"]]

        eye_mid = (left + right) * 0.5
        baseline = max(np.linalg.norm(right - left), 1e-6)
        yaw = float((nose[0] - eye_mid[0]) / baseline)
        pitch = float((nose[1] - mid[1]) / baseline)
        head_pose_ok = abs(yaw) < 0.20 and abs(pitch) < 0.25
        return yaw, pitch, head_pose_ok

    @staticmethod
    def _annotate_frame(
        frame_rgb: np.ndarray,
        points: np.ndarray,
        left_ear: float,
        right_ear: float,
        quality_ok: bool,
        gaze_ready: bool,
        gaze_x: float,
        gaze_y: float,
    ) -> None:
        if cv2 is None:
            return

        for idx in LEFT_IRIS + RIGHT_IRIS:
            x, y = points[idx]
            cv2.circle(frame_rgb, (int(x), int(y)), 2, (0, 255, 255), -1)

        status_text = f"EAR L:{left_ear:.3f} R:{right_ear:.3f} | quality:{'ok' if quality_ok else 'bad'}"
        cv2.putText(frame_rgb, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if gaze_ready:
            text = f"Gaze {gaze_x:.3f}, {gaze_y:.3f}"
            cv2.putText(frame_rgb, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


class TrackerController(QObject):
    """GUI-facing backend facade used by ``main.py``."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    recording_state_changed = pyqtSignal(bool)
    calibration_loaded = pyqtSignal(dict)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._camera_thread: Optional[CameraThread] = None
        self._recording = RecordingWriter()
        self._calibration_model = CalibrationModel()
        self._calibration_window: Optional[CalibrationWindow] = None
        self._latest_features: list[float] = []
        self._latest_quality_ok = False
        self._latest_metrics: dict[str, Any] = {}

    def start(self) -> None:
        if self._camera_thread is not None and self._camera_thread.isRunning():
            self.status_changed.emit("Tracking already running.")
            return

        self._camera_thread = CameraThread(parent=self)
        self._camera_thread.frame_ready.connect(self.frame_ready)
        self._camera_thread.metrics_ready.connect(self._handle_metrics)
        self._camera_thread.feature_vector_ready.connect(self._handle_feature_vector)
        self._camera_thread.status_changed.connect(self.status_changed)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geometry = screen.availableGeometry()
            self._camera_thread.set_screen_size(geometry.width(), geometry.height())

        if self._calibration_model.coefficients is not None:
            self._camera_thread.set_calibration_payload(self._current_calibration_payload())

        self._camera_thread.start()
        self.status_changed.emit("Tracking started.")

    def stop(self) -> None:
        if self._camera_thread is None:
            self.status_changed.emit("Tracking already stopped.")
            return
        self._camera_thread.stop()
        self._camera_thread = None
        self.status_changed.emit("Tracking stopped.")

    def begin_calibration(self) -> None:
        parent_widget = self.parent() if isinstance(self.parent(), QWidget) else None
        if self._calibration_window is not None:
            self._calibration_window.close()
            self._calibration_window = None

        self._calibration_window = CalibrationWindow(parent_widget)
        self._calibration_window.status_message.connect(self.status_changed)
        self._calibration_window.calibration_finished.connect(self._handle_calibration_finished)
        self._calibration_window.destroyed.connect(self._handle_calibration_window_closed)
        self._calibration_window.start()

    def save_calibration(self, path: str) -> None:
        payload = self._current_calibration_payload(required=True)
        CalibrationStorage.save(path, payload)
        self.status_changed.emit(f"Calibration saved to {path}")
        self.calibration_loaded.emit(payload.to_dict())

    def load_calibration(self, path: str) -> dict[str, Any]:
        payload = CalibrationStorage.load(path)
        self._calibration_model.load_from_dict(payload)
        if self._camera_thread is not None:
            self._camera_thread.set_calibration_payload(payload)
        self.calibration_loaded.emit(payload)
        self.status_changed.emit(f"Calibration loaded from {path}")
        return payload

    def start_recording(self, session_name: str, export_path: Optional[str] = None) -> None:
        final_path = self._recording.start(session_name, export_path)
        self.recording_state_changed.emit(True)
        self.status_changed.emit(f"Recording started: {final_path}")

    def stop_recording(self) -> None:
        output = self._recording.stop()
        self.recording_state_changed.emit(False)
        if output:
            self.status_changed.emit(f"Recording saved: {output}")
        else:
            self.status_changed.emit("Recording stopped.")

    def shutdown(self) -> None:
        if self._recording.is_recording:
            self.stop_recording()
        self.stop()
        if self._calibration_window is not None:
            self._calibration_window.close()
            self._calibration_window = None

    def _handle_metrics(self, metrics: dict[str, Any]) -> None:
        self._latest_metrics = dict(metrics)
        if self._recording.is_recording:
            self._recording.append_dict(metrics)
        self.metrics_ready.emit(metrics)

    def _handle_feature_vector(self, features: list[float], quality_ok: bool) -> None:
        self._latest_features = list(features)
        self._latest_quality_ok = quality_ok
        if self._calibration_window is not None:
            self._calibration_window.push_feature_vector(features, quality_ok)

    def _handle_calibration_finished(self, samples: list[CalibrationSample]) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("No primary screen available for calibration fitting.")

        geometry = screen.availableGeometry()
        screen_size = (geometry.width(), geometry.height())
        payload = self._calibration_model.fit(samples, screen_size)
        if self._camera_thread is not None:
            self._camera_thread.set_calibration_payload(payload.to_dict())
        self.calibration_loaded.emit(payload.to_dict())
        self.status_changed.emit(
            f"Calibration complete. Mean validation error: {payload.validation_error_px:.1f} px"
        )

    def _handle_calibration_window_closed(self) -> None:
        self._calibration_window = None

    def _current_calibration_payload(self, required: bool = False) -> CalibrationData:
        if self._calibration_model.coefficients is None:
            if required:
                raise RuntimeError("No calibration available to save.")
            return CalibrationData(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                camera_resolution=[0, 0],
                feature_names=self._calibration_model.FEATURE_NAMES,
                model_type="polynomial_regression_order_2",
                transformation_matrix=[],
                target_layout=self._calibration_model.target_layout,
                validation_error_px=None,
            )

        screen = QApplication.primaryScreen()
        if screen is not None:
            geometry = screen.availableGeometry()
            resolution = [geometry.width(), geometry.height()]
        else:
            resolution = [0, 0]

        return CalibrationData(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            camera_resolution=resolution,
            feature_names=self._calibration_model.FEATURE_NAMES,
            model_type="polynomial_regression_order_2",
            transformation_matrix=self._calibration_model.coefficients.tolist(),
            target_layout=self._calibration_model.target_layout,
            validation_error_px=self._calibration_model.validation_error_px,
        )


__all__ = [
    "BlinkDetector",
    "CalibrationStorage",
    "CameraThread",
    "RecordingWriter",
    "TrackerController",
]
