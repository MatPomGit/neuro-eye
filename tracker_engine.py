"""Runtime tracking backend for the eye-tracking research application.

This module provides the GUI-facing tracking controller, a background camera
worker, a practical blink detector, and integration points for calibration and
recording. MediaPipe inference cadence is configurable so the application can
run gaze estimation on every frame or every Nth frame while still keeping the
video preview responsive.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImage
from PyQt6.QtWidgets import QWidget

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
    """EAR-based blink detector with hysteresis and refractory period."""

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
        self._closed_counter = 0
        self._blink_start_ts: Optional[float] = None
        self._min_ear = 1.0
        self._last_blink_end_ts = 0.0
        self._events: deque[BlinkEvent] = deque(maxlen=256)

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @classmethod
    def _ear(cls, eye_points: np.ndarray) -> float:
        if eye_points.shape[0] != 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = eye_points
        horizontal = max(cls._distance(p1, p4), 1e-6)
        vertical = cls._distance(p2, p6) + cls._distance(p3, p5)
        return float(vertical / (2.0 * horizontal))

    def update(
        self,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        head_pose_ok: bool,
    ) -> tuple[bool, float, float]:
        """Update detector state.

        Returns:
            blink_detected_now, blink_rate_per_min, current_avg_ear
        """

        now = time.perf_counter()
        avg_ear = (self._ear(left_eye) + self._ear(right_eye)) / 2.0
        blink_detected = False

        if not head_pose_ok:
            self._state = "OPEN"
            self._closed_counter = 0
            return False, self.current_blink_rate_per_min(), avg_ear

        refractory_open = (now - self._last_blink_end_ts) * 1000.0 >= self.refractory_ms

        if self._state == "OPEN":
            if avg_ear < self.close_threshold and refractory_open:
                self._closed_counter += 1
                self._min_ear = min(self._min_ear, avg_ear)
                if self._closed_counter >= self.min_closed_frames:
                    self._state = "CLOSED"
                    self._blink_start_ts = now
            else:
                self._closed_counter = 0
                self._min_ear = 1.0

        elif self._state == "CLOSED":
            self._min_ear = min(self._min_ear, avg_ear)
            if avg_ear > self.open_threshold:
                end_ts = now
                start_ts = self._blink_start_ts or end_ts
                duration_ms = (end_ts - start_ts) * 1000.0
                if 80.0 <= duration_ms <= 400.0:
                    self._events.append(
                        BlinkEvent(
                            start_ts=start_ts,
                            end_ts=end_ts,
                            duration_ms=duration_ms,
                            min_ear=self._min_ear,
                        )
                    )
                    blink_detected = True
                self._state = "OPEN"
                self._closed_counter = 0
                self._blink_start_ts = None
                self._last_blink_end_ts = end_ts
                self._min_ear = 1.0

        return blink_detected, self.current_blink_rate_per_min(), avg_ear

    def current_blink_rate_per_min(self) -> float:
        now = time.perf_counter()
        window_start = now - 60.0
        while self._events and self._events[0].end_ts < window_start:
            self._events.popleft()
        return float(len(self._events))


@dataclass(slots=True)
class FrameMetrics:
    """Per-frame metrics emitted from the worker to the GUI."""

    timestamp: str
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    pupil_dilation: float = 0.0
    blink_rate: float = 0.0
    tracking_ready: bool = False
    blink_detected: bool = False
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    avg_ear: float = 0.0
    tracking_stride: int = 1
    tracking_frame_index: int = 0
    raw_feature_vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class TrackingConfig:
    """Worker configuration that may be changed from the UI at runtime."""

    camera_index: int = 0
    tracking_stride: int = 1
    mirror_preview: bool = True
    max_head_angle_deg: float = 18.0
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass(slots=True)
class FaceState:
    """Cached state from the most recent successful MediaPipe inference."""

    landmarks_px: Optional[np.ndarray] = None
    left_eye_px: Optional[np.ndarray] = None
    right_eye_px: Optional[np.ndarray] = None
    left_iris_px: Optional[np.ndarray] = None
    right_iris_px: Optional[np.ndarray] = None
    feature_vector: list[float] = field(default_factory=list)
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    tracking_ready: bool = False
    pupil_dilation: float = 0.0
    blink_detected: bool = False
    blink_rate: float = 0.0
    avg_ear: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    quality_ok: bool = False
    inferenced_frame_index: int = 0


class CameraWorker(QObject):
    """Background webcam capture and MediaPipe processing worker."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, calibration_model: Optional[CalibrationModel] = None) -> None:
        super().__init__()
        self._config = TrackingConfig()
        self._calibration_model = calibration_model or CalibrationModel()
        self._running = False
        self._lock = threading.Lock()
        self._blink_detector = BlinkDetector()
        self._screen_size = self._current_screen_size()
        self._smoothed_gaze: Optional[np.ndarray] = None
        self._last_face_state = FaceState()

    def update_calibration_model(self, model: CalibrationModel) -> None:
        with self._lock:
            self._calibration_model = model

    def set_tracking_stride(self, stride: int) -> None:
        stride = max(1, min(int(stride), 3))
        with self._lock:
            self._config.tracking_stride = stride
        self.status_changed.emit(f"Tracking stride set to every {stride} frame(s).")

    def get_tracking_stride(self) -> int:
        with self._lock:
            return self._config.tracking_stride

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def run(self) -> None:
        self._running = True
        if cv2 is None:
            self.status_changed.emit("OpenCV is not installed. Preview unavailable.")
            self.finished.emit()
            return

        cap = cv2.VideoCapture(self._config.camera_index)
        if not cap.isOpened():
            self.status_changed.emit("Unable to open camera.")
            self.finished.emit()
            return

        if mp is None:
            self.status_changed.emit("MediaPipe unavailable. Running preview-only mode.")
            self._run_preview_only(cap)
            return

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        self.status_changed.emit(
            f"Camera started. MediaPipe tracking every {self.get_tracking_stride()} frame(s)."
        )

        frame_index = 0
        try:
            while self._is_running():
                ok, frame_bgr = cap.read()
                if not ok:
                    self.status_changed.emit("Camera frame read failed.")
                    break

                if self._config.mirror_preview:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_index += 1

                stride = self.get_tracking_stride()
                should_infer = (frame_index - 1) % stride == 0
                if should_infer:
                    frame_rgb.flags.writeable = False
                    results = face_mesh.process(frame_rgb)
                    frame_rgb.flags.writeable = True
                    self._last_face_state = self._analyze_results(frame_rgb, results, frame_index)

                face_state = self._last_face_state
                metrics = FrameMetrics(
                    timestamp=datetime.now().isoformat(timespec="milliseconds"),
                    gaze_x=face_state.gaze_x,
                    gaze_y=face_state.gaze_y,
                    pupil_dilation=face_state.pupil_dilation,
                    blink_rate=face_state.blink_rate,
                    tracking_ready=face_state.tracking_ready,
                    blink_detected=face_state.blink_detected,
                    head_yaw=face_state.head_yaw,
                    head_pitch=face_state.head_pitch,
                    avg_ear=face_state.avg_ear,
                    tracking_stride=stride,
                    tracking_frame_index=face_state.inferenced_frame_index,
                    raw_feature_vector=list(face_state.feature_vector),
                )

                frame_with_overlay = self._draw_overlay(frame_rgb.copy(), face_state)
                self.frame_ready.emit(self._to_qimage(frame_with_overlay))
                self.metrics_ready.emit(asdict(metrics))
        finally:
            face_mesh.close()
            cap.release()
            self.status_changed.emit("Camera worker stopped.")
            self.finished.emit()

    def _run_preview_only(self, cap: Any) -> None:
        try:
            while self._is_running():
                ok, frame_bgr = cap.read()
                if not ok:
                    self.status_changed.emit("Camera frame read failed.")
                    break
                if self._config.mirror_preview:
                    frame_bgr = cv2.flip(frame_bgr, 1)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(self._to_qimage(frame_rgb))
                self.metrics_ready.emit(
                    asdict(
                        FrameMetrics(
                            timestamp=datetime.now().isoformat(timespec="milliseconds"),
                            tracking_stride=self.get_tracking_stride(),
                        )
                    )
                )
        finally:
            cap.release()
            self.finished.emit()

    def _analyze_results(self, image_rgb: np.ndarray, results: Any, frame_index: int) -> FaceState:
        state = FaceState(inferenced_frame_index=frame_index)
        if not getattr(results, "multi_face_landmarks", None):
            self._smoothed_gaze = None
            return state

        points = self._landmarks_to_pixels(
            results.multi_face_landmarks[0], image_rgb.shape[1], image_rgb.shape[0]
        )
        left_eye = points[LEFT_EYE_EAR]
        right_eye = points[RIGHT_EYE_EAR]
        left_iris = points[LEFT_IRIS]
        right_iris = points[RIGHT_IRIS]

        head_yaw, head_pitch = self._estimate_head_pose(points)
        head_pose_ok = (
            abs(head_yaw) <= self._config.max_head_angle_deg
            and abs(head_pitch) <= self._config.max_head_angle_deg
        )
        blink_detected, blink_rate, avg_ear = self._blink_detector.update(
            left_eye,
            right_eye,
            head_pose_ok,
        )

        feature_vector = self._extract_feature_vector(points, head_yaw, head_pitch)
        gaze_x, gaze_y, tracking_ready = self._map_and_smooth_gaze(feature_vector)
        pupil_dilation = self._estimate_pupil_dilation(left_iris, right_iris)
        quality_ok = head_pose_ok and not blink_detected and len(feature_vector) == 6

        state.landmarks_px = points
        state.left_eye_px = left_eye
        state.right_eye_px = right_eye
        state.left_iris_px = left_iris
        state.right_iris_px = right_iris
        state.feature_vector = feature_vector
        state.gaze_x = gaze_x
        state.gaze_y = gaze_y
        state.tracking_ready = tracking_ready
        state.pupil_dilation = pupil_dilation
        state.blink_detected = blink_detected
        state.blink_rate = blink_rate
        state.avg_ear = avg_ear
        state.head_yaw = head_yaw
        state.head_pitch = head_pitch
        state.quality_ok = quality_ok
        return state

    def _map_and_smooth_gaze(self, feature_vector: list[float]) -> tuple[float, float, bool]:
        with self._lock:
            model = self._calibration_model
        gaze_x, gaze_y, tracking_ready = model.map_to_screen(feature_vector, self._screen_size)
        if not tracking_ready:
            self._smoothed_gaze = None
            return gaze_x, gaze_y, tracking_ready

        current = np.asarray([gaze_x, gaze_y], dtype=np.float64)
        if self._smoothed_gaze is None:
            self._smoothed_gaze = current
        else:
            alpha = 0.35
            self._smoothed_gaze = alpha * current + (1.0 - alpha) * self._smoothed_gaze
        return float(self._smoothed_gaze[0]), float(self._smoothed_gaze[1]), True

    @staticmethod
    def _landmarks_to_pixels(face_landmarks: Any, width: int, height: int) -> np.ndarray:
        coords = []
        for landmark in face_landmarks.landmark:
            coords.append([landmark.x * width, landmark.y * height])
        return np.asarray(coords, dtype=np.float64)

    @staticmethod
    def _estimate_head_pose(points: np.ndarray) -> tuple[float, float]:
        left = points[HEAD_POSE_REF["left"]]
        right = points[HEAD_POSE_REF["right"]]
        nose = points[HEAD_POSE_REF["nose"]]
        mid = points[HEAD_POSE_REF["mid"]]

        head_width = max(np.linalg.norm(right - left), 1e-6)
        center_x = (left[0] + right[0]) / 2.0
        yaw = ((nose[0] - center_x) / head_width) * 90.0

        eye_line_y = (left[1] + right[1]) / 2.0
        vertical_scale = max(abs(mid[1] - eye_line_y), 1e-6)
        pitch = ((nose[1] - mid[1]) / vertical_scale) * 35.0
        return float(yaw), float(pitch)

    @staticmethod
    def _extract_feature_vector(points: np.ndarray, head_yaw: float, head_pitch: float) -> list[float]:
        def iris_relative(iris_idx: list[int], box_idx: dict[str, int]) -> tuple[float, float]:
            iris = points[iris_idx]
            outer = points[box_idx["outer"]]
            inner = points[box_idx["inner"]]
            top = points[box_idx["top"]]
            bottom = points[box_idx["bottom"]]
            iris_center = iris.mean(axis=0)
            width = max(abs(inner[0] - outer[0]), 1e-6)
            height = max(abs(bottom[1] - top[1]), 1e-6)
            min_x = min(outer[0], inner[0])
            min_y = min(top[1], bottom[1])
            rel_x = float((iris_center[0] - min_x) / width)
            rel_y = float((iris_center[1] - min_y) / height)
            return rel_x, rel_y

        l_ix, l_iy = iris_relative(LEFT_IRIS, LEFT_EYE_BOX)
        r_ix, r_iy = iris_relative(RIGHT_IRIS, RIGHT_EYE_BOX)
        return [l_ix, l_iy, r_ix, r_iy, float(head_yaw), float(head_pitch)]

    @staticmethod
    def _estimate_pupil_dilation(left_iris: np.ndarray, right_iris: np.ndarray) -> float:
        def iris_radius(iris: np.ndarray) -> float:
            center = iris.mean(axis=0)
            return float(np.mean(np.linalg.norm(iris - center, axis=1)))

        return float((iris_radius(left_iris) + iris_radius(right_iris)) / 2.0)

    @staticmethod
    def _draw_overlay(image_rgb: np.ndarray, face_state: FaceState) -> np.ndarray:
        if cv2 is None or face_state.landmarks_px is None:
            return image_rgb

        output = image_rgb
        for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR + LEFT_IRIS + RIGHT_IRIS:
            x, y = face_state.landmarks_px[idx]
            cv2.circle(output, (int(x), int(y)), 2, (0, 255, 0), -1)

        if face_state.tracking_ready:
            gx = int(face_state.gaze_x * output.shape[1])
            gy = int(face_state.gaze_y * output.shape[0])
            cv2.drawMarker(
                output,
                (gx, gy),
                (255, 64, 64),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )

        return output

    @staticmethod
    def _to_qimage(image_rgb: np.ndarray) -> QImage:
        height, width, channels = image_rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return qimage.copy()

    @staticmethod
    def _current_screen_size() -> tuple[int, int]:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return 1920, 1080
        geometry = screen.availableGeometry()
        return geometry.width(), geometry.height()

    def _is_running(self) -> bool:
        with self._lock:
            return self._running


class TrackerController(QWidget):
    """GUI-facing orchestration layer for tracking, calibration, and recording."""

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    recording_state_changed = pyqtSignal(bool)
    calibration_loaded = pyqtSignal(dict)
    tracking_stride_changed = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[CameraWorker] = None
        self._thread: Optional[QThread] = None
        self._recording = RecordingWriter()
        self._calibration_model = CalibrationModel()
        self._calibration_payload: Optional[dict[str, Any]] = None
        self._calibration_window: Optional[CalibrationWindow] = None
        self._current_stride = 1

    def start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self.status_changed.emit("Tracking is already running.")
            return

        self._thread = QThread(self)
        self._worker = CameraWorker(calibration_model=self._calibration_model)
        self._worker.set_tracking_stride(self._current_stride)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.metrics_ready.connect(self._on_metrics_ready)
        self._worker.status_changed.connect(self.status_changed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()
        self.status_changed.emit("Tracking requested.")

    def stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self.status_changed.emit("Tracking stop requested.")

    def set_tracking_stride(self, stride: int) -> None:
        stride = max(1, min(int(stride), 3))
        self._current_stride = stride
        if self._worker is not None:
            self._worker.set_tracking_stride(stride)
        self.tracking_stride_changed.emit(stride)
        self.status_changed.emit(f"Tracking frequency changed: MediaPipe every {stride} frame(s).")

    def get_tracking_stride(self) -> int:
        return self._current_stride

    def begin_calibration(self) -> None:
        if self._calibration_window is not None:
            self.status_changed.emit("Calibration is already active.")
            return

        window = CalibrationWindow(parent=None)
        window.status_message.connect(self.status_changed)
        window.calibration_finished.connect(self._finalize_calibration)
        window.destroyed.connect(lambda *_: self._clear_calibration_window())
        self._calibration_window = window
        window.start()

    def save_calibration(self, path: str) -> None:
        if self._calibration_payload is None:
            raise ValueError("No calibration model is available to save.")
        payload = CalibrationData(
            timestamp=self._calibration_payload["timestamp"],
            camera_resolution=list(self._calibration_payload["camera_resolution"]),
            feature_names=list(self._calibration_payload["feature_names"]),
            model_type=self._calibration_payload["model_type"],
            transformation_matrix=list(self._calibration_payload["transformation_matrix"]),
            target_layout=list(self._calibration_payload["target_layout"]),
            validation_error_px=self._calibration_payload.get("validation_error_px"),
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
        self.status_changed.emit(f"Recording saved: {path}" if path else "Recording stopped.")

    def shutdown(self) -> None:
        if self._recording.is_recording:
            self.stop_recording()
        if self._calibration_window is not None:
            self._calibration_window.close()
        self.stop()

    def _on_metrics_ready(self, payload: dict[str, Any]) -> None:
        self.metrics_ready.emit(payload)
        if self._recording.is_recording:
            self._recording.append_dict(payload)
        if self._calibration_window is not None:
            raw_feature_vector = list(payload.get("raw_feature_vector", []))
            quality_ok = (
                not bool(payload.get("blink_detected", False))
                and len(raw_feature_vector) == 6
                and abs(float(payload.get("head_yaw", 0.0))) < 18.0
                and abs(float(payload.get("head_pitch", 0.0))) < 18.0
            )
            self._calibration_window.push_feature_vector(raw_feature_vector, quality_ok)

    def _finalize_calibration(self, samples: list[CalibrationSample]) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            screen_size = (1920, 1080)
        else:
            geometry = screen.availableGeometry()
            screen_size = (geometry.width(), geometry.height())

        payload = self._calibration_model.fit(samples, screen_size)
        self._calibration_payload = asdict(payload)
        if self._worker is not None:
            self._worker.update_calibration_model(self._calibration_model)
        self.calibration_loaded.emit(self._calibration_payload)
        self.status_changed.emit(
            f"Calibration completed. Mean validation error: {payload.validation_error_px:.1f}px"
        )
        self._clear_calibration_window()

    def _clear_calibration_window(self) -> None:
        self._calibration_window = None

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None


__all__ = [
    "CalibrationData",
    "CalibrationStorage",
    "TrackerController",
    "CameraWorker",
    "CalibrationModel",
    "BlinkDetector",
]
