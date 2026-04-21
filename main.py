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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
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
        Path(path).write_text(
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


class LiveTrackingTab(QWidget):
    """Primary live tracking workspace with video and telemetry panel."""

    tracking_toggled = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = TrackingSnapshot()
        self.video_label = VideoFrameLabel()
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
            "calibration.yaml",
            CALIBRATION_FILENAME_FILTER,
        )
        if path:
            self.save_calibration_requested.emit(path)

    def _on_load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            "",
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
            "gaze_recording.csv",
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


class MainWindow(QMainWindow):
    """Top-level application shell with tabbed navigation."""

    def __init__(self) -> None:
        super().__init__()
        self.tracker = make_tracker_controller(self)
        self.tabs = QTabWidget()
        self.live_tab = LiveTrackingTab()
        self.calibration_tab = CalibrationTab()
        self.recording_tab = RecordingTab()
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

    app = configure_application()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
