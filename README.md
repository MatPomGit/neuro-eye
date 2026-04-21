# Eye Tracking Research Suite

Modularna aplikacja badawcza do eye trackingu w Pythonie, zbudowana na `PyQt6`, `OpenCV` i `MediaPipe Face Mesh / Iris`.

Projekt zawiera:
- interfejs tabbed UI w `PyQt6`,
- podgląd kamery na żywo,
- estymację landmarków oczu i tęczówek,
- detektor mrugnięć oparty o `EAR` z histerezą,
- 9-punktową kalibrację fullscreen,
- zapis kalibracji do YAML,
- recording metryk do CSV lub JSON,
- regulację częstości inferencji MediaPipe: co 1, 2 lub 3 klatki.

## Struktura projektu

```text
main.py
tracker_engine.py
calibration.py
data_io.py
requirements.txt
environment.yml
README.md
widgets/
  calibration_window.py
```

## Wymagania systemowe

Zalecane:
- Python 3.11
- Conda lub Miniconda
- kamera internetowa
- system Windows, Linux albo macOS

## Uruchamianie przez conda

Ta ścieżka jest zalecana, bo minimalizuje problemy z zależnościami GUI i CV.

### 1. Utwórz środowisko

W katalogu projektu uruchom:

```bash
conda env create -f environment.yml
```

Jeżeli środowisko już istnieje i chcesz je odświeżyć:

```bash
conda env update -f environment.yml --prune
```

### 2. Aktywuj środowisko

```bash
conda activate eye-tracking-research
```

### 3. Uruchom aplikację

```bash
python main.py
```

## Alternatywa: szybka instalacja w istniejącym środowisku conda

Jeżeli wolisz stworzyć środowisko ręcznie:

```bash
conda create -n eye-tracking-research python=3.11 -y
conda activate eye-tracking-research
pip install -r requirements.txt
python main.py
```

## Uwagi o zależnościach

Projekt używa:
- `PyQt6` do GUI,
- `opencv-python` do kamery i przetwarzania obrazu,
- `mediapipe` do detekcji twarzy, oczu i tęczówek,
- `numpy` do obliczeń,
- `pyyaml` do kalibracji w YAML.

W niektórych środowiskach `mediapipe` może mieć bardziej restrykcyjne wymagania niż reszta pakietów. Dlatego domyślna instrukcja używa `conda` jako nośnika środowiska, a same pakiety instalowane są przez `pip` wewnątrz tego środowiska.

## Jak działa regulacja częstości śledzenia

W zakładce `Live Tracking` dostępny jest suwak `MediaPipe Frequency`.

Zakres:
- `1` = inferencja co klatkę,
- `2` = inferencja co 2 klatki,
- `3` = inferencja co 3 klatki.

Interpretacja praktyczna:
- `1` daje najwyższą responsywność i największe obciążenie CPU,
- `2` zwykle daje najlepszy kompromis,
- `3` zmniejsza koszt obliczeń, ale zwiększa wiek ostatniej inferencji i może pogorszyć responsywność blink/gaze.

Podgląd kamery nadal odświeża się płynnie, a pomiędzy inferencjami backend używa ostatniego poprawnego stanu twarzy.

## Główne moduły

### `main.py`
Warstwa GUI:
- główne okno,
- zakładki aplikacji,
- obsługa przycisków i statusów,
- suwak regulacji stride dla MediaPipe,
- integracja z backendem `TrackerController`.

### `tracker_engine.py`
Backend runtime:
- obsługa kamery w `QThread`,
- pipeline MediaPipe,
- blink detector,
- zbieranie feature vector dla kalibracji,
- integracja z recordingiem i fullscreen calibration.

### `calibration.py`
Logika kalibracji:
- próbki kalibracyjne,
- model mapowania gaze feature -> współrzędne ekranu,
- polynomial regression 2. rzędu.

### `data_io.py`
Warstwa I/O:
- zapis/odczyt YAML,
- eksport CSV/JSON.

### `widgets/calibration_window.py`
Fullscreen UI do 9-punktowej kalibracji.

## Obecne ograniczenia

To jest już sensowna baza badawcza, ale nadal nie finalny produkt produkcyjny. Aktualne ograniczenia:
- dokładność gaze zależy od jakości kamery i oświetlenia,
- blink detection jest praktyczny, ale niekliniczny,
- brak zaawansowanej kompensacji head pose 3D,
- brak wyboru kamery z poziomu UI,
- brak raportu jakości kalibracji punkt po punkcie w interfejsie.

## Typowe problemy

### Kamera się nie otwiera
Sprawdź:
- czy żadna inna aplikacja nie używa kamery,
- czy system nadał uprawnienia do kamery,
- czy `camera_index=0` odpowiada właściwemu urządzeniu.

### MediaPipe nie działa
Najczęściej pomaga:
- użycie czystego środowiska `conda`,
- ponowna instalacja z `environment.yml`,
- upewnienie się, że wersja Pythona zgadza się z plikiem środowiska.

### GUI uruchamia się, ale bez trackingu
Aplikacja ma bezpieczny fallback. Jeżeli backend nie może uruchomić kamery albo MediaPipe, UI nadal wstaje, ale pracuje w ograniczonym trybie.

## Dalszy rozwój

Najbardziej sensowne kolejne kroki:
- wybór kamery z UI,
- walidacja jakości kalibracji w interfejsie,
- lepsza kompensacja ruchu głowy,
- bardziej stabilny pupil diameter proxy,
- heatmap i AOI jako kolejne moduły badawcze.
