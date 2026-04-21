"""
Main entry point aplikacji Neuro-Eye.

Kod UI (PyQt6) trzyma logikę widoków i sterowanie sesją.
Silnik śledzenia oraz kalibracja są delegowane do tracker_engine.py.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tracker_engine import CalibrationModel, GazeSample, TrackerThread


class MainWindow(QMainWindow):
    """Główne okno aplikacji z tabami badawczymi."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Neuro-Eye | Eye Tracking Research App")
        self.resize(1300, 800)

        # Katalogi robocze na dane.
        self.base_dir = Path.cwd()
        self.calib_dir = self.base_dir / "data" / "calibrations"
        self.session_dir = self.base_dir / "data" / "sessions"
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Model kalibracji współdzielony między UI i trackerem.
        self.calibration_model = CalibrationModel()

        # Bufor sesji nagrywania.
        self.recording_enabled = False
        self.recorded_samples: list[GazeSample] = []

        # Wątek trackera (kamera + MediaPipe).
        self.tracker_thread: Optional[TrackerThread] = None

        # Budowa interfejsu.
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_tab = self._build_live_tracking_tab()
        self.calib_tab = self._build_calibration_tab()
        self.record_tab = self._build_recording_tab()

        self.tabs.addTab(self.live_tab, "Live Tracking")
        self.tabs.addTab(self.calib_tab, "Calibration")
        self.tabs.addTab(self.record_tab, "Recording & Export")

        # Placeholdery rozwojowe (wyłączone).
        self.tabs.addTab(QWidget(), "Heatmap Generator")
        self.tabs.setTabEnabled(3, False)

        self.tabs.addTab(QWidget(), "Area of Interest (AOI) Editor")
        self.tabs.setTabEnabled(4, False)

        self.tabs.addTab(QWidget(), "External Hardware Sync (EEG/HRM)")
        self.tabs.setTabEnabled(5, False)

        self._start_tracker()

    def _build_live_tracking_tab(self) -> QWidget:
        """Tworzy zakładkę podglądu na żywo i panel parametrów."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Lewa część: obraz z kamery.
        self.video_label = QLabel("No camera feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(900, 650)
        self.video_label.setStyleSheet("background-color: #111; color: #ddd;")
        layout.addWidget(self.video_label, stretch=3)

        # Prawa część: parametry.
        panel = QGroupBox("Real-time Parameters")
        panel_layout = QFormLayout(panel)

        self.lbl_gaze_x = QLabel("-")
        self.lbl_gaze_y = QLabel("-")
        self.lbl_pupil = QLabel("-")
        self.lbl_blink = QLabel("-")

        panel_layout.addRow("Gaze X:", self.lbl_gaze_x)
        panel_layout.addRow("Gaze Y:", self.lbl_gaze_y)
        panel_layout.addRow("Pupil Dilation:", self.lbl_pupil)
        panel_layout.addRow("Blink Rate (/min):", self.lbl_blink)

        layout.addWidget(panel, stretch=1)
        return tab

    def _build_calibration_tab(self) -> QWidget:
        """Tworzy zakładkę kalibracji 9-punktowej."""
        tab = QWidget()
        root = QVBoxLayout(tab)

        info = QLabel(
            "9-point calibration:\n"
            "Kliknij Start Calibration i patrz kolejno na punkty.\n"
            "Wersja bazowa: próbki dla punktów są zbierane z aktualnego gaze vector."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # Przyciski sterujące kalibracją.
        buttons_layout = QHBoxLayout()
        self.btn_start_calib = QPushButton("Start Calibration")
        self.btn_save_calib = QPushButton("Save Calibration")
        self.btn_load_calib = QPushButton("Load Calibration")

        self.btn_start_calib.clicked.connect(self._run_calibration_template)
        self.btn_save_calib.clicked.connect(self._save_calibration_yaml)
        self.btn_load_calib.clicked.connect(self._load_calibration_yaml)

        buttons_layout.addWidget(self.btn_start_calib)
        buttons_layout.addWidget(self.btn_save_calib)
        buttons_layout.addWidget(self.btn_load_calib)
        root.addLayout(buttons_layout)

        # Podgląd statusu.
        self.lbl_calib_status = QLabel("Calibration status: not calibrated")
        root.addWidget(self.lbl_calib_status)

        return tab

    def _build_recording_tab(self) -> QWidget:
        """Tworzy zakładkę nagrywania i eksportu danych."""
        tab = QWidget()
        root = QVBoxLayout(tab)

        form_box = QGroupBox("Session Settings")
        form_layout = QGridLayout(form_box)

        self.session_name_input = QLineEdit()
        self.session_name_input.setPlaceholderText("e.g. participant_001_taskA")
        form_layout.addWidget(QLabel("Session Name:"), 0, 0)
        form_layout.addWidget(self.session_name_input, 0, 1)

        self.btn_record = QPushButton("Start Record")
        self.btn_record.clicked.connect(self._toggle_recording)
        form_layout.addWidget(self.btn_record, 1, 0, 1, 2)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_json.clicked.connect(self._export_json)
        form_layout.addWidget(self.btn_export_csv, 2, 0)
        form_layout.addWidget(self.btn_export_json, 2, 1)

        root.addWidget(form_box)

        self.lbl_recording_status = QLabel("Recording: OFF")
        root.addWidget(self.lbl_recording_status)

        return tab

    def _start_tracker(self) -> None:
        """Uruchamia wątek śledzenia kamery."""
        self.tracker_thread = TrackerThread(camera_index=0, calibration_model=self.calibration_model)
        self.tracker_thread.frame_ready.connect(self._on_frame_ready)
        self.tracker_thread.metrics_ready.connect(self._on_metrics_ready)
        self.tracker_thread.start()

    def _on_frame_ready(self, image: QImage) -> None:
        """Odbiera klatkę z trackera i aktualizuje QLabel."""
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def _on_metrics_ready(self, sample: GazeSample) -> None:
        """Aktualizuje metryki UI i ewentualnie zapisuje próbkę do sesji."""
        self.lbl_gaze_x.setText(f"{sample.gaze_x:.3f}")
        self.lbl_gaze_y.setText(f"{sample.gaze_y:.3f}")
        self.lbl_pupil.setText(f"{sample.pupil_dilation:.3f}")
        self.lbl_blink.setText(f"{sample.blink_rate:.1f}")

        if self.recording_enabled:
            self.recorded_samples.append(sample)

    def _run_calibration_template(self) -> None:
        """
        Szablon kalibracji 9-punktowej.

        W produkcji: prezentuj punkt na ekranie -> zbieraj N próbek landmark vector
        -> uśrednij -> dodaj do datasetu -> fit transform.
        """
        if not self.tracker_thread:
            QMessageBox.warning(self, "Calibration", "Tracker is not running.")
            return

        # 9 punktów w układzie znormalizowanym (0..1, 0..1).
        points_screen = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
        ]

        eye_vectors = []
        for _pt in points_screen:
            # Uproszczenie MVP: pobieramy ostatni vector trackera.
            vec = self.tracker_thread.get_latest_eye_vector()
            if vec is None:
                QMessageBox.warning(self, "Calibration", "No eye vector available.")
                return
            eye_vectors.append(vec)

        self.calibration_model.fit(
            eye_vectors=eye_vectors,
            screen_points=points_screen,
            camera_resolution=self.tracker_thread.camera_resolution,
        )
        self.lbl_calib_status.setText("Calibration status: calibrated")
        QMessageBox.information(self, "Calibration", "9-point calibration completed (template).")

    def _save_calibration_yaml(self) -> None:
        """Zapisuje kalibrację do pliku YAML."""
        if not self.calibration_model.is_fitted:
            QMessageBox.warning(self, "Calibration", "Model is not calibrated yet.")
            return

        default_name = f"calibration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            str(self.calib_dir / default_name),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return

        self.calibration_model.save_yaml(path)
        QMessageBox.information(self, "Calibration", f"Saved to:\n{path}")

    def _load_calibration_yaml(self) -> None:
        """Wczytuje kalibrację z YAML."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            str(self.calib_dir),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return

        self.calibration_model.load_yaml(path)
        self.lbl_calib_status.setText("Calibration status: loaded")
        QMessageBox.information(self, "Calibration", f"Loaded from:\n{path}")

    def _toggle_recording(self) -> None:
        """Włącza/wyłącza nagrywanie próbek gaze."""
        self.recording_enabled = not self.recording_enabled
        if self.recording_enabled:
            self.recorded_samples.clear()
            self.btn_record.setText("Stop Record")
            self.lbl_recording_status.setText("Recording: ON")
        else:
            self.btn_record.setText("Start Record")
            self.lbl_recording_status.setText(f"Recording: OFF (samples={len(self.recorded_samples)})")

    def _session_basename(self) -> str:
        """Zwraca bezpieczną nazwę sesji."""
        raw = self.session_name_input.text().strip()
        if not raw:
            raw = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_"))

    def _export_csv(self) -> None:
        """Eksportuje nagrane dane do CSV."""
        if not self.recorded_samples:
            QMessageBox.warning(self, "Export CSV", "No data recorded.")
            return

        out_path = self.session_dir / f"{self._session_basename()}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "gaze_x", "gaze_y", "pupil_dilation", "blink_rate", "eye_vector"]
            )
            for s in self.recorded_samples:
                writer.writerow(
                    [s.timestamp, s.gaze_x, s.gaze_y, s.pupil_dilation, s.blink_rate, list(s.eye_vector)]
                )

        QMessageBox.information(self, "Export CSV", f"Saved:\n{out_path}")

    def _export_json(self) -> None:
        """Eksportuje nagrane dane do JSON."""
        if not self.recorded_samples:
            QMessageBox.warning(self, "Export JSON", "No data recorded.")
            return

        out_path = self.session_dir / f"{self._session_basename()}.json"
        payload = [
            {
                "timestamp": s.timestamp,
                "gaze_x": s.gaze_x,
                "gaze_y": s.gaze_y,
                "pupil_dilation": s.pupil_dilation,
                "blink_rate": s.blink_rate,
                "eye_vector": list(s.eye_vector),
            }
            for s in self.recorded_samples
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Export JSON", f"Saved:\n{out_path}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        """Sprzątanie wątku trackera przy zamknięciu aplikacji."""
        if self.tracker_thread:
            self.tracker_thread.stop()
            self.tracker_thread.wait(2000)
        super().closeEvent(event)


def main() -> None:
    """Uruchamia aplikację desktopową."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Tworzymy okno główne.
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
