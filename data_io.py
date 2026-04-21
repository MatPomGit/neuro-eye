"""Data persistence helpers for calibration and gaze recording."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Protocol

import yaml

from calibration import CalibrationData


class SupportsAsDict(Protocol):
    """Protocol for serializable row objects used by the recording writer."""

    def __dict__(self) -> dict[str, Any]:
        ...


class CalibrationStorage:
    """Read and write calibration data as YAML files."""

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
            return yaml.safe_load(handle) or {}


class RecordingWriter:
    """Buffered writer for gaze telemetry exported to CSV or JSON."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._session_name = ""
        self._export_path: Optional[Path] = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, session_name: str, export_path: str | None = None) -> str:
        if not session_name.strip():
            raise ValueError("session_name must not be empty")
        self._session_name = session_name.strip()
        self._export_path = Path(export_path) if export_path else Path.cwd() / f"{self._session_name}.csv"
        self._rows = []
        self._is_recording = True
        return str(self._export_path)

    def append_dict(self, payload: dict[str, Any]) -> None:
        if not self._is_recording:
            return
        row = dict(payload)
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
            fieldnames = list(self._rows[0].keys()) if self._rows else ["session_name"]
            with export_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._rows)

        self._rows = []
        self._session_name = ""
        self._export_path = None
        self._is_recording = False
        return str(export_path)
