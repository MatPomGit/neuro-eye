"""
Silnik śledzenia wzroku (kamera + MediaPipe) oraz model kalibracji.

Zawiera:
- QThread do nieblokującego pobierania klatek.
- Placeholder estymacji gaze/pupil/blink.
- Model kalibracji z zapisem/odczytem YAML.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


@dataclass
class GazeSample:
    """Pojedyncza próbka danych gaze do UI i zapisu sesji."""
    timestamp: str
    gaze_x: float
    gaze_y: float
    pupil_dilation: float
    blink_rate: float
    eye_vector: np.ndarray


class CalibrationModel:
    """
    Prosty model kalibracji: mapowanie wektora oka -> (x, y) ekranu.

    Tu użyta jest liniowa regresja wielomianowa rzędu 1:
      [x, y] = [1, v0, v1, ..., vn] @ W
    """

    def __init__(self) -> None:
        self.transform_matrix: Optional[np.ndarray] = None
        self.camera_resolution: tuple[int, int] = (0, 0)
        self.timestamp_utc: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        """Informuje, czy model został dopasowany."""
        return self.transform_matrix is not None

    def fit(
        self,
        eye_vectors: list[np.ndarray],
        screen_points: list[tuple[float, float]],
        camera_resolution: tuple[int, int],
    ) -> None:
        """Dopasowuje transformację liniową metodą najmniejszych kwadratów."""
        x = np.asarray(eye_vectors, dtype=np.float64)
        y = np.asarray(screen_points, dtype=np.float64)

        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("Invalid calibration dataset dimensions.")

        # Dodajemy bias (kolumnę jedynek) dla wyrazu wolnego.
        x_aug = np.hstack([np.ones((x.shape[0], 1)), x])

        # Rozwiązanie LS: x_aug @ W ~= y
        w, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        self.transform_matrix = w  # shape: (n_features + 1, 2)
        self.camera_resolution = camera_resolution
        self.timestamp_utc = datetime.utcnow().isoformat()

    def map_eye_vector(self, eye_vector: np.ndarray) -> tuple[float, float]:
        """Mapuje wektor oka na współrzędne gaze w układzie znormalizowanym."""
        if self.transform_matrix is None:
            # Fallback bez kalibracji.
            return 0.5, 0.5

        v = np.asarray(eye_vector, dtype=np.float64).reshape(1, -1)
        v_aug = np.hstack([np.ones((1, 1)), v])
        pred = v_aug @ self.transform_matrix
        gx, gy = float(pred[0, 0]), float(pred[0, 1])
        return gx, gy

    def save_yaml(self, path: str) -> None:
        """Zapisuje model kalibracji do YAML."""
        if self.transform_matrix is None:
            raise RuntimeError("Cannot save: model is not fitted.")

        payload = {
            "timestamp_utc": self.timestamp_utc,
            "camera_resolution": {
                "width": int(self.camera_resolution[0]),
                "height": int(self.camera_resolution[1]),
            },
            "transformation_matrix": self.transform_matrix.tolist(),
            "model_type": "linear_polynomial_regression_order_1",
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    def load_yaml(self, path: str) -> None:
        """Wczytuje model kalibracji z YAML."""
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)

        tm = np.asarray(payload["transformation_matrix"], dtype=np.float64)
        width = int(payload["camera_resolution"]["width"])
        height = int(payload["camera_resolution"]["height"])

        self.transform_matrix = tm
        self.camera_resolution = (width, height)
        self.timestamp_utc = payload.get("timestamp_utc")


class TrackerThread(QThread):
    """
    Wątek przechwytujący obraz z kamery i liczący parametry wzroku.

    Sygnały:
    - frame_ready(QImage): klatka do wyświetlenia w UI.
    - metrics_ready(GazeSample): bieżące metryki.
    """

    frame_ready = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(GazeSample)

    def __init__(self, camera_index: int, calibration_model: CalibrationModel) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.calibration_model = calibration_model
        self.running = False
        self.latest_eye_vector: Optional[np.ndarray] = None
        self.camera_resolution: tuple[int, int] = (640, 480)

        # Blink state (placeholder logiki).
        self.blink_counter = 0
        self.start_time = time.time()

        # MediaPipe FaceMesh.
        self.face_mesh = None
        if mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Ważne: iris landmarks.
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def stop(self) -> None:
        """Prosi pętlę run o zatrzymanie."""
        self.running = False

    def get_latest_eye_vector(self) -> Optional[np.ndarray]:
        """Zwraca ostatni dostępny wektor oka (do kalibracji)."""
        return self.latest_eye_vector.copy() if self.latest_eye_vector is not None else None

    def run(self) -> None:
        """Pętla główna trackera uruchamiana w osobnym wątku."""
        self.running = True
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.camera_resolution = (width, height)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            eye_vector = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            pupil_dilation = 0.0

            if self.face_mesh is not None:
                result = self.face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    h, w = rgb.shape[:2]

                    # Indeksy orientacyjne iris/eye (MediaPipe FaceMesh).
                    # Komentarz: w eksperymencie możesz doprecyzować zestaw indeksów.
                    left_iris_idx = [468, 469, 470, 471, 472]
                    right_iris_idx = [473, 474, 475, 476, 477]

                    left_pts = np.array(
                        [(landmarks[i].x * w, landmarks[i].y * h) for i in left_iris_idx],
                        dtype=np.float64,
                    )
                    right_pts = np.array(
                        [(landmarks[i].x * w, landmarks[i].y * h) for i in right_iris_idx],
                        dtype=np.float64,
                    )

                    left_center = left_pts.mean(axis=0)
                    right_center = right_pts.mean(axis=0)

                    # Wektor oka (placeholder cech do modelu kalibracji).
                    eye_vector = np.array(
                        [
                            left_center[0] / w,
                            left_center[1] / h,
                            right_center[0] / w,
                            right_center[1] / h,
                        ],
                        dtype=np.float64,
                    )

                    # Prosty estymator średnicy źrenicy: promień z punktów iris.
                    left_radius = np.mean(np.linalg.norm(left_pts - left_center, axis=1))
                    right_radius = np.mean(np.linalg.norm(right_pts - right_center, axis=1))
                    pupil_dilation = float((left_radius + right_radius) / 2.0)

                    # Overlay punktów.
                    for p in np.vstack([left_pts, right_pts]):
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)

            self.latest_eye_vector = eye_vector

            gaze_x, gaze_y = self.calibration_model.map_eye_vector(eye_vector)

            # Blink rate placeholder: utrzymane API, docelowo podmienić na realny detector.
            elapsed_min = max((time.time() - self.start_time) / 60.0, 1e-6)
            blink_rate = float(self.blink_counter / elapsed_min)

            sample = GazeSample(
                timestamp=datetime.utcnow().isoformat(),
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                pupil_dilation=pupil_dilation,
                blink_rate=blink_rate,
                eye_vector=eye_vector,
            )
            self.metrics_ready.emit(sample)

            qimg = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.strides[0],
                QImage.Format.Format_BGR888,
            ).copy()
            self.frame_ready.emit(qimg)

        cap.release()
        if self.face_mesh is not None:
            self.face_mesh.close()
