Eye Tracking Research Suite - local run notes

1. Create a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the application:
   python main.py

Files prepared so far:
- main.py: PyQt6 application shell and tabs
- tracker_engine.py: backend tracking, blink detection, calibration, recording
- calibration.py: isolated calibration model and sample structures
- data_io.py: YAML and CSV/JSON persistence helpers
- widgets/calibration_window.py: fullscreen 9-point target presenter

Suggested next refactor:
- move CalibrationModel, CalibrationStorage, RecordingWriter, and CalibrationWindow imports out of tracker_engine.py and import them from the dedicated modules.
