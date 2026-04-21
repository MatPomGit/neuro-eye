# Eye Tracking Research App

## Important MediaPipe note

This project uses the classic **Face Mesh / Iris landmarks** API. It requires a MediaPipe build that exposes the **solutions** interface.

If `python -c "import mediapipe as mp; print(dir(mp))"` does not show `solutions`, your installed package is not compatible with this code path. In that case the application now falls back to **preview-only mode** instead of crashing, but gaze/blink tracking will be unavailable.

The recommended environment is the pinned Conda setup below.

## Conda setup

```bash
conda env remove -n eye-tracking-research -y
conda env config vars set QT_QPA_PLATFORM=xcb
conda env create -f environment.yml
conda activate eye-tracking-research
python main.py
```

## Verify MediaPipe

```bash
python -c "import mediapipe as mp; print(mp.__version__); import mediapipe.solutions.face_mesh as fm; print(fm.FaceMesh)"
```

That command should print a valid `FaceMesh` class. If it fails, reinstall the environment from `environment.yml`.

## Linux camera notes

- Camera enumeration prefers `/dev/video*` on Linux to reduce noisy OpenCV warnings.
- Some drivers ignore requested FPS or resolution presets. The camera may negotiate a nearby mode.
- On Wayland, Qt may still print harmless plugin warnings.
