"""Main application entry point for the modular eye-tracking research app.

This module defines the PyQt6 GUI shell requested in the project brief:
- Live Tracking tab with video preview and real-time metrics.
- Calibration tab with a 9-point calibration workflow entrypoint.
- Recording & Export tab for session naming and data export control.
- Disabled placeholder tabs for future research modules.

The implementation is intentionally modular and integrates with ``tracker_engine``
through a thin adapter layer. When the backend module is unavailable during early
project bootstrapping, the UI still starts with a safe fallback stub.
"""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSizePolicy,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


try:
    from tracker_engine import CalibrationStorage, TrackerController
except ImportError:
    CalibrationStorage = None
    TrackerController = None


APP_NAME = "Eye Tracking Research Suite"
APP_ORG = "PCDB"
DEFAULT_WINDOW_SIZE = (1440, 900)
CALIBRATION_FILENAME_FILTER = "YAML Files (*.yaml *.yml)"
EXPORT_FILENAME_FILTER = "CSV Files (*.csv);;JSON Files (*.json)"
DATA_DIR = Path("data")
SESSIONS_DIR = DATA_DIR / "sessions"
CALIBRATIONS_DIR = DATA_DIR / "calibrations"


@dataclass(slots=True)
class TrackingSnapshot:
    """Small view-model used by the GUI for current tracker outputs."""

    gaze_x: float = 0.0
    gaze_y: float = 0.0
    pupil_dilation: float = 0.0
    blink_rate: float = 0.0
    tracking_ready: bool = False
    blink_detected: bool = False


class StubTrackerController(QWidget):
    """Fallback backend used when ``tracker_engine`` is not importable.

    The class mirrors the public surface expected from the real backend well
    enough for GUI wiring and early manual testing.
    """

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    recording_state_changed = pyqtSignal(bool)
    calibration_loaded = pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._is_running = False
        self._is_recording = False
        self._status = "Tracker backend not connected (stub mode)."

    def start(self) -> None:
        self._is_running = True
        self.status_changed.emit("Tracking started in stub mode.")
        self.metrics_ready.emit(
            {
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "pupil_dilation": 0.0,
                "blink_rate": 0.0,
                "tracking_ready": False,
                "blink_detected": False,
            }
        )

    def stop(self) -> None:
        self._is_running = False
        self.status_changed.emit("Tracking stopped.")

    def start_recording(self, session_name: str, export_path: Optional[str] = None) -> None:
        self._is_recording = True
        self.recording_state_changed.emit(True)
        self.status_changed.emit(
            f"Recording requested for session '{session_name}' (stub mode)."
        )

    def stop_recording(self) -> None:
        self._is_recording = False
        self.recording_state_changed.emit(False)
        self.status_changed.emit("Recording stopped.")

    def save_calibration(self, path: str) -> None:
        calibration_path = Path(path)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_path.write_text(
            "timestamp: null\n"
            "camera_resolution: [0, 0]\n"
            "transformation_matrix: []\n",
            encoding="utf-8",
        )
        self.status_changed.emit(f"Calibration placeholder saved to {path}")

    def load_calibration(self, path: str) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera_resolution": [0, 0],
            "transformation_matrix": [],
        }
        self.calibration_loaded.emit(payload)
        self.status_changed.emit(f"Calibration placeholder loaded from {path}")
        return payload

    def begin_calibration(self) -> None:
        self.status_changed.emit("Calibration requested (stub mode).")

    def shutdown(self) -> None:
        self.stop()
        if self._is_recording:
            self.stop_recording()


def make_tracker_controller(parent: Optional[QWidget] = None) -> QWidget:
    """Create the real backend controller when available, otherwise a stub."""

    if TrackerController is not None:
        try:
            return TrackerController(parent=parent)
        except TypeError:
            return TrackerController()
    return StubTrackerController()


class VideoFrameLabel(QLabel):
    """Scaled video surface with a neutral placeholder appearance."""

    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap = QPixmap()
        self.setMinimumSize(960, 540)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("videoFrameLabel")
        self.set_placeholder_text("Camera preview will appear here")

    def set_placeholder_text(self, text: str) -> None:
        self._source_pixmap = QPixmap()
        self.clear()
        self.setText(text)
        self.setStyleSheet(
            "QLabel#videoFrameLabel {"
            "background-color: #111827;"
            "color: #D1D5DB;"
            "border: 1px solid #374151;"
            "font-size: 16px;"
            "}"
        )

    def set_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self.set_placeholder_text("No frame available")
            return
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return
        target_size = self.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_scaled_pixmap()


class TimeSeriesChart(QWidget):
    """Prosty wykres liniowy przebiegu wartości X/Y w czasie."""

    def __init__(
        self,
        title: str,
        x_label: str = "X",
        y_label: str = "Y",
        max_points: int = 180,
    ) -> None:
        super().__init__()
        # Bufor kołowy ogranicza zużycie pamięci przy długiej pracy aplikacji.
        self._title = title
        self._x_label = x_label
        self._y_label = y_label
        self._x_values: deque[float] = deque(maxlen=max_points)
        self._y_values: deque[float] = deque(maxlen=max_points)
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear_series(self) -> None:
        """Czyści historię wykresu, np. po zatrzymaniu śledzenia."""
        self._x_values.clear()
        self._y_values.clear()
        self.update()

    def append_sample(self, x_value: float, y_value: float) -> None:
        """Dodaje nową próbkę X/Y do wykresu czasowego."""
        self._x_values.append(float(x_value))
        self._y_values.append(float(y_value))
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        """Rysuje osie oraz dwie serie czasowe (X i Y)."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0B1220"))

        chart_rect = self.rect().adjusted(10, 26, -10, -18)
        if chart_rect.width() <= 0 or chart_rect.height() <= 0:
            return

        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRect(chart_rect)

        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(chart_rect.left(), chart_rect.top() - 8, self._title)

        if len(self._x_values) < 2:
            painter.setPen(QColor("#94A3B8"))
            painter.drawText(chart_rect.adjusted(8, 8, -8, -8), Qt.AlignmentFlag.AlignCenter, "Brak danych")
            return

        points_x = self._series_to_points(self._x_values, chart_rect)
        points_y = self._series_to_points(self._y_values, chart_rect)

        painter.setPen(QPen(QColor("#38BDF8"), 2))
        for i in range(1, len(points_x)):
            painter.drawLine(points_x[i - 1], points_x[i])
        painter.setPen(QPen(QColor("#F97316"), 2))
        for i in range(1, len(points_y)):
            painter.drawLine(points_y[i - 1], points_y[i])

        painter.setPen(QColor("#E2E8F0"))
        painter.drawText(chart_rect.left() + 8, chart_rect.bottom() - 4, self._x_label)
        painter.drawText(chart_rect.left() + 56, chart_rect.bottom() - 4, self._y_label)

    @staticmethod
    def _series_to_points(series: deque[float], rect: Any) -> list[QPointF]:
        """Mapuje wartości z zakresu [0, 1] do punktów w obszarze rysowania."""
        count = len(series)
        if count < 2:
            return []
        step = rect.width() / max(count - 1, 1)
        result: list[QPointF] = []
        for index, value in enumerate(series):
            clamped = max(0.0, min(1.0, float(value)))
            x_pos = rect.left() + index * step
            y_pos = rect.bottom() - clamped * rect.height()
            result.append(QPointF(x_pos, y_pos))
        return result


class LiveTrackingTab(QWidget):
    """Primary live tracking workspace with video and telemetry panel."""

    tracking_toggled = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = TrackingSnapshot()
        self.video_label = VideoFrameLabel()
        self.eye_position_chart = TimeSeriesChart(
            "Pozycja oczu na obrazie (znormalizowana)",
            x_label="X (cyjan)",
            y_label="Y (pomarańcz)",
        )
        self.gaze_focus_chart = TimeSeriesChart(
            "Punkt skupienia wzroku na ekranie (znormalizowany)",
            x_label="Gaze X (cyjan)",
            y_label="Gaze Y (pomarańcz)",
        )
        self.start_button = QPushButton("Start Tracking")
        self.stop_button = QPushButton("Stop")
        self.status_value = QLabel("Idle")
        self.gaze_x_value = QLabel("0.000")
        self.gaze_y_value = QLabel("0.000")
        self.pupil_value = QLabel("0.000")
        self.blink_rate_value = QLabel("0.00 blinks/min")
        self.blink_flag_value = QLabel("No")
        self.tracking_flag_value = QLabel("No")
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        controls = self._build_metrics_panel()

        self.start_button.clicked.connect(lambda: self.tracking_toggled.emit(True))
        self.stop_button.clicked.connect(lambda: self.tracking_toggled.emit(False))
        self.stop_button.setEnabled(False)

        left_panel = QVBoxLayout()
        left_panel.addWidget(self.video_label, stretch=1)
        left_panel.addWidget(self.eye_position_chart, stretch=0)
        left_panel.addWidget(self.gaze_focus_chart, stretch=0)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        left_panel.addLayout(button_row)

        root.addLayout(left_panel, stretch=5)
        root.addWidget(controls, stretch=2)

    def _build_metrics_panel(self) -> QWidget:
        panel = QGroupBox("Live Parameters")
        layout = QFormLayout(panel)
        # Długi status z backendu nie może rozszerzać całego okna aplikacji.
        self.status_value.setWordWrap(True)
        self.status_value.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addRow("Tracker Status", self.status_value)
        layout.addRow("Gaze X", self.gaze_x_value)
        layout.addRow("Gaze Y", self.gaze_y_value)
        layout.addRow("Pupil Dilation", self.pupil_value)
        layout.addRow("Blink Rate", self.blink_rate_value)
        layout.addRow("Blink Detected", self.blink_flag_value)
        layout.addRow("Tracking Ready", self.tracking_flag_value)
        return panel

    def set_running_state(self, is_running: bool) -> None:
        self.start_button.setEnabled(not is_running)
        self.stop_button.setEnabled(is_running)
        self.status_value.setText("Running" if is_running else "Idle")
        if not is_running:
            # Po zatrzymaniu sesji czyścimy wykresy, aby nie mieszać przebiegów.
            self.eye_position_chart.clear_series()
            self.gaze_focus_chart.clear_series()

    def update_status(self, status: str) -> None:
        self.status_value.setText(status)

    def update_frame(self, image: QImage) -> None:
        self.video_label.set_frame(image)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        self.snapshot = TrackingSnapshot(
            gaze_x=float(metrics.get("gaze_x", 0.0)),
            gaze_y=float(metrics.get("gaze_y", 0.0)),
            pupil_dilation=float(metrics.get("pupil_dilation", 0.0)),
            blink_rate=float(metrics.get("blink_rate", 0.0)),
            tracking_ready=bool(metrics.get("tracking_ready", False)),
            blink_detected=bool(metrics.get("blink_detected", False)),
        )
        self.gaze_x_value.setText(f"{self.snapshot.gaze_x:.3f}")
        self.gaze_y_value.setText(f"{self.snapshot.gaze_y:.3f}")
        self.pupil_value.setText(f"{self.snapshot.pupil_dilation:.3f}")
        self.blink_rate_value.setText(f"{self.snapshot.blink_rate:.2f} blinks/min")
        self.blink_flag_value.setText("Yes" if self.snapshot.blink_detected else "No")
        self.tracking_flag_value.setText("Yes" if self.snapshot.tracking_ready else "No")

        # Przybliżona pozycja oczu na obrazie z cech kalibracyjnych (średnia lewe/prawe oko).
        raw_features = metrics.get("raw_feature_vector", [])
        if isinstance(raw_features, list) and len(raw_features) >= 4:
            eye_x = (float(raw_features[0]) + float(raw_features[2])) / 2.0
            eye_y = (float(raw_features[1]) + float(raw_features[3])) / 2.0
            self.eye_position_chart.append_sample(eye_x, eye_y)

        # Wykres punktu skupienia wzroku dodajemy tylko dla poprawnej estymacji.
        if self.snapshot.tracking_ready:
            self.gaze_focus_chart.append_sample(self.snapshot.gaze_x, self.snapshot.gaze_y)


class CalibrationPreview(QWidget):
    """Simple preview widget for the 3x3 calibration target layout."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(240)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0F172A"))

        pen = QPen(QColor("#60A5FA"))
        pen.setWidth(2)
        painter.setPen(pen)

        radius = 10
        width = self.width()
        height = self.height()
        xs = [0.1, 0.5, 0.9]
        ys = [0.1, 0.5, 0.9]

        for y in ys:
            for x in xs:
                cx = int(width * x)
                cy = int(height * y)
                painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
                painter.drawLine(cx - 16, cy, cx + 16, cy)
                painter.drawLine(cx, cy - 16, cx, cy + 16)


class CalibrationTab(QWidget):
    """Calibration controls and persistence actions."""

    start_calibration_requested = pyqtSignal()
    save_calibration_requested = pyqtSignal(str)
    load_calibration_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.status_label = QLabel("Calibration not started.")
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.preview = CalibrationPreview()
        self.start_button = QPushButton("Start Calibration")
        self.save_button = QPushButton("Save Calibration")
        self.load_button = QPushButton("Load Calibration")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_box = QGroupBox("9-Point Calibration")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(
            QLabel(
                "Run a fullscreen 3x3 target routine and map eye features to screen "
                "coordinates using the backend calibration model."
            )
        )
        info_layout.addWidget(self.preview)
        info_layout.addWidget(self.status_label)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addStretch(1)
        info_layout.addLayout(button_row)

        metadata_box = QGroupBox("Calibration Metadata")
        metadata_layout = QVBoxLayout(metadata_box)
        metadata_layout.addWidget(self.metadata_text)

        layout.addWidget(info_box)
        layout.addWidget(metadata_box)

        self.start_button.clicked.connect(self.start_calibration_requested.emit)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)

    def _on_save_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            str(CALIBRATIONS_DIR / "calibration.yaml"),
            CALIBRATION_FILENAME_FILTER,
        )
        if path:
            self.save_calibration_requested.emit(path)

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            str(CALIBRATIONS_DIR),
            CALIBRATION_FILENAME_FILTER,
        )
        if path:
            self.load_calibration_requested.emit(path)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def update_metadata(self, payload: dict[str, Any]) -> None:
        timestamp = payload.get("timestamp", "n/a")
        resolution = payload.get("camera_resolution", "n/a")
        matrix = payload.get("transformation_matrix", [])
        self.metadata_text.setPlainText(
            f"timestamp: {timestamp}\n"
            f"camera_resolution: {resolution}\n"
            f"transformation_matrix: {matrix}\n"
        )


class RecordingTab(QWidget):
    """Session recording controls and export information."""

    recording_toggled = pyqtSignal(bool, str, str)

    def __init__(self) -> None:
        super().__init__()
        self.session_name_edit = QLineEdit(datetime.now().strftime("session_%Y%m%d_%H%M%S"))
        self.export_path_edit = QLineEdit()
        self.record_button = QPushButton("Record")
        self.stop_button = QPushButton("Stop Recording")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        config_box = QGroupBox("Session Export")
        form = QGridLayout(config_box)
        browse_button = QPushButton("Browse")
        form.addWidget(QLabel("Session Name"), 0, 0)
        form.addWidget(self.session_name_edit, 0, 1, 1, 2)
        form.addWidget(QLabel("Export Path"), 1, 0)
        form.addWidget(self.export_path_edit, 1, 1)
        form.addWidget(browse_button, 1, 2)
        form.addWidget(self.record_button, 2, 1)
        form.addWidget(self.stop_button, 2, 2)

        log_box = QGroupBox("Recording Log")
        log_layout = QVBoxLayout(log_box)
        log_layout.addWidget(self.log_output)

        layout.addWidget(config_box)
        layout.addWidget(log_box)

        self.stop_button.setEnabled(False)
        browse_button.clicked.connect(self._choose_export_path)
        self.record_button.clicked.connect(self._emit_record_start)
        self.stop_button.clicked.connect(self._emit_record_stop)

    def _choose_export_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Export Path",
            str(SESSIONS_DIR / "gaze_recording.csv"),
            EXPORT_FILENAME_FILTER,
        )
        if path:
            self.export_path_edit.setText(path)

    def _emit_record_start(self) -> None:
        self.recording_toggled.emit(
            True,
            self.session_name_edit.text().strip(),
            self.export_path_edit.text().strip(),
        )

    def _emit_record_stop(self) -> None:
        self.recording_toggled.emit(
            False,
            self.session_name_edit.text().strip(),
            self.export_path_edit.text().strip(),
        )

    def set_recording_state(self, is_recording: bool) -> None:
        self.record_button.setEnabled(not is_recording)
        self.stop_button.setEnabled(is_recording)

    def append_log(self, text: str) -> None:
        self.log_output.append(text)


class PerspectiveViewport3D(QWidget):
    """Wizualizacja pseudo-3D z perspektywą zależną od pozycji głowy."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(420)
        self._yaw = 0.0
        self._pitch = 0.0
        self._distance_scale = 1.0
        self._tracking_ready = False
        self._distance_baseline: Optional[float] = None

    def update_from_metrics(self, metrics: dict[str, Any]) -> None:
        """Aktualizuje parametry kamery na podstawie telemetrii śledzenia."""
        self._tracking_ready = bool(metrics.get("tracking_ready", False))
        if not self._tracking_ready:
            self.update()
            return

        yaw = float(metrics.get("head_yaw", 0.0))
        pitch = float(metrics.get("head_pitch", 0.0))
        pupil = float(metrics.get("pupil_dilation", 0.0))

        # Wygładzanie redukuje mikrodrgania sygnału z landmarków.
        self._yaw = 0.84 * self._yaw + 0.16 * yaw
        self._pitch = 0.84 * self._pitch + 0.16 * pitch

        # Używamy promienia tęczówki jako prostego przybliżenia dystansu.
        # Większa wartość zwykle oznacza mniejszą odległość od kamery.
        if pupil > 1e-6:
            if self._distance_baseline is None:
                self._distance_baseline = pupil
            else:
                self._distance_baseline = 0.98 * self._distance_baseline + 0.02 * pupil
            ratio = pupil / max(self._distance_baseline, 1e-6)
            self._distance_scale = float(max(0.65, min(1.45, ratio)))

        self.update()

    @staticmethod
    def _project_point(
        point: tuple[float, float, float],
        eye: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Rzutuje punkt 3D na płaszczyznę ekranu (z=0) metodą pinhole camera."""
        px, py, pz = point
        ex, ey, ez = eye
        denom = ez - pz
        if abs(denom) < 1e-6:
            denom = 1e-6
        t = ez / denom
        sx = ex + (px - ex) * t
        sy = ey + (py - ey) * t
        return sx, sy

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#020617"))

        frame = self.rect().adjusted(36, 36, -36, -36)
        painter.setPen(QPen(QColor("#1E293B"), 2))
        painter.setBrush(QColor("#0F172A"))
        painter.drawRoundedRect(frame, 12, 12)

        # Przekształcenie pozy głowy na pozycję "wirtualnego oka" widza.
        # Dzięki temu obiekt wydaje się znajdować za ekranem.
        yaw_norm = max(-1.0, min(1.0, self._yaw / 20.0))
        pitch_norm = max(-1.0, min(1.0, self._pitch / 20.0))
        cam_x = yaw_norm * 0.55
        cam_y = -pitch_norm * 0.45
        cam_z = 2.0 / max(self._distance_scale, 0.35)
        eye = (cam_x, cam_y, cam_z)

        # Sześcian osadzony "za ekranem", czyli przy ujemnych wartościach Z.
        size = 0.62
        z_center = -1.8
        cube_vertices = [
            (-size, -size, z_center - size),
            (size, -size, z_center - size),
            (size, size, z_center - size),
            (-size, size, z_center - size),
            (-size, -size, z_center + size),
            (size, -size, z_center + size),
            (size, size, z_center + size),
            (-size, size, z_center + size),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        projected = [self._project_point(point, eye) for point in cube_vertices]
        scale = min(frame.width(), frame.height()) * 0.36
        cx = frame.center().x()
        cy = frame.center().y()
        points_2d = [QPointF(cx + p[0] * scale, cy + p[1] * scale) for p in projected]

        glow_alpha = 210 if self._tracking_ready else 110
        painter.setPen(QPen(QColor(56, 189, 248, glow_alpha), 2))
        for start, end in edges:
            painter.drawLine(points_2d[start], points_2d[end])

        # Delikatna "mgła głębi" wzmacnia złudzenie odległości obiektu.
        painter.setPen(QPen(QColor(125, 211, 252, 55), 1))
        for ring in range(5):
            inset = int(18 + ring * 12)
            r = frame.adjusted(inset, inset, -inset, -inset)
            if r.width() > 0 and r.height() > 0:
                painter.drawRoundedRect(r, 8, 8)

        info_text = (
            "Rusz głową na boki i w pionie, aby zmienić perspektywę.\n"
            "Przybliż/oddal głowę od kamery, aby zmienić odczucie głębi."
        )
        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(
            frame.adjusted(10, 10, -10, -10),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            info_text,
        )


class Depth3DTab(QWidget):
    """Zakładka prezentująca efekt paralaksy i głębi 3D za ekranem."""

    def __init__(self) -> None:
        super().__init__()
        self.viewport = PerspectiveViewport3D()
        self.status_label = QLabel("Uruchom tracking, aby aktywować perspektywę opartą o ruch głowy.")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel("Głębia 3D")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        subtitle = QLabel(
            "Wizualizacja obiektu 3D osadzonego za ekranem, aktualizowana na żywo "
            "na podstawie pozy i odległości głowy od kamery."
        )
        subtitle.setWordWrap(True)
        self.status_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.viewport, stretch=1)
        layout.addWidget(self.status_label)

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        self.viewport.update_from_metrics(metrics)
        tracking_ready = bool(metrics.get("tracking_ready", False))
        if tracking_ready:
            yaw = float(metrics.get("head_yaw", 0.0))
            pitch = float(metrics.get("head_pitch", 0.0))
            self.status_label.setText(
                f"Tracking aktywny. Head yaw: {yaw:.1f}°, head pitch: {pitch:.1f}°."
            )
        else:
            self.status_label.setText(
                "Brak stabilnego trackingu. Spójrz w kamerę i utrzymaj twarz w kadrze."
            )


class MainWindow(QMainWindow):
    """Top-level application shell with tabbed navigation."""

    def __init__(self) -> None:
        super().__init__()
        self.tracker = make_tracker_controller(self)
        self.tabs = QTabWidget()
        self.live_tab = LiveTrackingTab()
        self.calibration_tab = CalibrationTab()
        self.recording_tab = RecordingTab()
        self.depth_3d_tab = Depth3DTab()
        self._setup_window()
        self._build_tabs()
        self._build_menu()
        self._connect_signals()

    def _setup_window(self) -> None:
        self.setWindowTitle(APP_NAME)
        self.resize(*DEFAULT_WINDOW_SIZE)
        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def _build_tabs(self) -> None:
        self.tabs.addTab(self.live_tab, "Live Tracking")
        self.tabs.addTab(self.calibration_tab, "Calibration")
        self.tabs.addTab(self.recording_tab, "Recording & Export")
        self.tabs.addTab(self.depth_3d_tab, "Głębia 3D")

        for title in (
            "Heatmap Generator",
            "Area of Interest (AOI) Editor",
            "External Hardware Sync (EEG/HRM)",
        ):
            placeholder = self._make_disabled_placeholder(title)
            index = self.tabs.addTab(placeholder, title)
            self.tabs.setTabEnabled(index, False)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tracking_menu = self.menuBar().addMenu("Tracking")
        start_action = QAction("Start", self)
        stop_action = QAction("Stop", self)
        start_action.triggered.connect(lambda: self._handle_tracking_toggle(True))
        stop_action.triggered.connect(lambda: self._handle_tracking_toggle(False))
        tracking_menu.addAction(start_action)
        tracking_menu.addAction(stop_action)

    def _make_disabled_placeholder(self, title: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"{title} is reserved for future expansion.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        layout.addWidget(label)
        layout.addStretch(1)
        return widget

    def _connect_signals(self) -> None:
        self.live_tab.tracking_toggled.connect(self._handle_tracking_toggle)
        self.calibration_tab.start_calibration_requested.connect(self._handle_calibration_start)
        self.calibration_tab.save_calibration_requested.connect(self._handle_calibration_save)
        self.calibration_tab.load_calibration_requested.connect(self._handle_calibration_load)
        self.recording_tab.recording_toggled.connect(self._handle_recording_toggle)

        self.tracker.frame_ready.connect(self.live_tab.update_frame)
        self.tracker.metrics_ready.connect(self.live_tab.update_metrics)
        self.tracker.metrics_ready.connect(self.depth_3d_tab.update_metrics)
        self.tracker.status_changed.connect(self._broadcast_status)
        self.tracker.recording_state_changed.connect(self.recording_tab.set_recording_state)
        self.tracker.calibration_loaded.connect(self.calibration_tab.update_metadata)

    def _broadcast_status(self, status: str) -> None:
        self.statusBar().showMessage(status)
        self.live_tab.update_status(status)
        self.recording_tab.append_log(status)
        self.calibration_tab.set_status(status)

    def _handle_tracking_toggle(self, should_start: bool) -> None:
        try:
            if should_start:
                self.tracker.start()
            else:
                self.tracker.stop()
            self.live_tab.set_running_state(should_start)
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error("Tracking Error", str(exc))

    def _handle_calibration_start(self) -> None:
        try:
            self.tracker.begin_calibration()
            self.calibration_tab.set_status(
                "Fullscreen calibration requested. Follow on-screen targets."
            )
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error("Calibration Error", str(exc))

    def _handle_calibration_save(self, path: str) -> None:
        try:
            self.tracker.save_calibration(path)
            self.calibration_tab.set_status(f"Calibration saved to: {path}")
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error("Save Calibration Error", str(exc))

    def _handle_calibration_load(self, path: str) -> None:
        try:
            payload = self.tracker.load_calibration(path)
            if isinstance(payload, dict):
                self.calibration_tab.update_metadata(payload)
            self.calibration_tab.set_status(f"Calibration loaded from: {path}")
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error("Load Calibration Error", str(exc))

    def _handle_recording_toggle(
        self,
        should_record: bool,
        session_name: str,
        export_path: str,
    ) -> None:
        if not session_name:
            self._show_error("Invalid Session", "Session name cannot be empty.")
            return

        try:
            if should_record:
                self.tracker.start_recording(session_name, export_path or None)
            else:
                self.tracker.stop_recording()
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._show_error("Recording Error", str(exc))

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(f"{title}: {message}")

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        shutdown = getattr(self.tracker, "shutdown", None)
        if callable(shutdown):
            shutdown()
        super().closeEvent(event)


def configure_application() -> QApplication:
    """Create and configure the QApplication instance."""

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setStyle("Fusion")
    return app


def main() -> int:
    """Run the desktop application."""

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    app = configure_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
