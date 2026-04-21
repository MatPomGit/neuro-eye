"""Fullscreen calibration target presenter widget."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from calibration import CalibrationModel, CalibrationSample


class CalibrationWindow(QWidget):
    """Fullscreen presenter implementing a 9-point calibration routine.

    Valid feature vectors are pushed in externally by the tracking backend.
    The widget only manages presentation timing and accepted sample aggregation.
    """

    calibration_finished = pyqtSignal(list)
    status_message = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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

    def push_feature_vector(self, features: list[float], quality_ok: bool) -> None:
        if self.collecting and quality_ok:
            self._point_samples.append(list(features))

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.status_message.emit("Calibration cancelled.")
            self.close()
            return
        super().keyPressEvent(event)

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

            progress_width = max(0.0, (self.current_index + 1) / len(self.points) * (self.width() - 40.0))
            progress_rect = QRectF(20.0, self.height() - 40.0, progress_width, 12.0)
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

    def _finalize_current_point(self) -> None:
        self.collecting = False
        if not self._point_samples:
            self.status_message.emit(f"Point {self.current_index + 1}/9 had no valid samples. Retrying.")
            self._settle_timer.start(500)
            return

        features = np.median(np.asarray(self._point_samples, dtype=np.float64), axis=0).tolist()
        tx, ty = self.points[self.current_index]
        self.collected_samples.append(
            CalibrationSample(
                target_x=tx,
                target_y=ty,
                features=features,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )
        self._advance_to_next_point()
