# FaceCursor — Hands-Free Computer Control

[![CI](https://github.com/prawnsgupta/Cursor-Control-System-Using-Computer-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/prawnsgupta/Cursor-Control-System-Using-Computer-Vision/actions/workflows/ci.yml)

Control your computer's cursor, clicks, scrolling, volume, and screen brightness using only your face and hands. FaceCursor combines **MediaPipe** (Face Mesh + Hands) with a custom **PyTorch GNN + PINN** model for smooth, jitter-free cursor motion — running in real time on CPU with a model that is just **~9K parameters (40 KB)**.

## Features

- **Face-controlled cursor** — move the cursor by moving your head; a deep-learning model translates nose trajectories into smooth cursor motion.
- **Mouth clicks** — open your mouth slightly to click. Turning your head right while doing so triggers a right-click; otherwise a left-click. Cooldown-guarded so one gesture equals one click.
- **Head scrolling** — toggle scroll mode and nod up/down; scroll velocity decays smoothly (momentum-style).
- **Hand gestures** — right-hand pinch adjusts system volume, left-hand pinch adjusts screen brightness.

## Architecture

```
Webcam frame (30 FPS)
      │  MediaPipe Face Mesh (478 landmarks, refined w/ iris)
      ▼
FaceGNN — 2× GCNConv (3→64→128) over the FACEMESH_TESSELATION graph,
      │   nose-node readout → raw (dx, dy) displacement
      ▼
MotionPINN — physics-informed smoother (2→32→2, Tanh-bounded) enforcing
      │   kinematic continuity; suppresses high-frequency facial twitches
      ▼
Control pipeline — EMA bias correction → non-linear gain → edge boost
      │   → exponential low-pass → dead-zone → step clamp
      ▼
pyautogui cursor updates
```

- **Why a GNN?** Facial landmarks are not independent — neighbouring mesh vertices move together. Graph convolutions over the actual face-mesh topology exploit that structure with a fraction of the parameters a flat MLP would need. That's how the model stays at ~9K params / 40 KB and runs real-time on CPU.
- **Why physics-informed?** The PINN stage penalises physically implausible motion, filtering sensor noise without the lag of a plain moving-average filter.
- **Bias self-correction:** a slow EMA of the model's own output is subtracted each frame, so any drift the model develops cancels out instead of pulling the cursor.

## Install

```bash
pip install -r requirements.txt
```

> **Version pin that matters:** `mediapipe < 0.10.30` — newer releases remove the legacy `mp.solutions` API (FaceMesh/Hands) this project uses, and the 0.10.2x line needs `numpy < 2`. `requirements.txt` encodes the verified combination.

## Run

```bash
python run.py
```

Focus the OpenCV window to use the hotkeys:

| Key | Action |
| --- | --- |
| `E` | Toggle **Cursor Mode** |
| `S` | Toggle **Scroll Mode** |
| `V` | Toggle **Volume Mode** (right-hand pinch) |
| `B` | Toggle **Brightness Mode** (left-hand pinch) |
| `D` | **Disable all** modes |
| `Q` | **Quit** |

## Tuning

All control constants live at the top of `run.py`:

| Constant | Default | Effect |
| --- | --- | --- |
| `RAW_GAIN` | 500 | base head-motion → pixels scaling |
| `MODEL_GAIN` | 0.6 | how much the GNN+PINN correction contributes |
| `LOWPASS_ALPHA` | 0.35 | smoothing strength (lower = smoother, laggier) |
| `DEADZONE` | 1.2 px | ignores micro-jitter below this displacement |
| `NOSE_EPS` | 0.0008 | minimum nose movement to process a frame |
| `MOUTH_OPEN_THRESH` | 0.028 | mouth aperture that registers a click |
| `SCROLL_GAIN` / `SCROLL_ALPHA` | 1600 / 0.25 | scroll speed and momentum decay |

## Train on your own head movements

The shipped `models/face_cursor.pth` works out of the box, but 60 seconds of your own data makes it feel native:

1. **Record** (well-lit room, trace your full head range + screen corners, ≥30–60 s, `Q` to save):
   ```bash
   python collect_data.py
   ```
2. **Train** (8 epochs on the recorded nose-velocity targets, best checkpoint auto-saved):
   ```bash
   python train.py
   ```

Data collection and training are sequential — `train.py` reads `data/samples.npz`, which is written when `collect_data.py` exits.

## Tests

Headless model tests (no webcam or GUI needed — CI runs these on every push):

```bash
python -m pytest tests/ -v
```

Covered: forward shapes for GNN/PINN/combined, strict pretrained-checkpoint compatibility (guards architecture drift), eval determinism, parameter-budget ceiling, and training-data structure.

## Repository structure

```
run.py               # real-time controller (cursor / clicks / scroll / volume / brightness)
collect_data.py      # webcam data recorder → data/samples.npz
train.py             # trains FaceGNN+MotionPINN on recorded head velocities
models/
  gnn.py             # FaceGNN — 2-layer GCNConv over the face mesh
  pinn.py            # MotionPINN — Tanh-bounded motion smoother
  combined.py        # FaceCursorModel = FaceGNN → MotionPINN
  face_cursor.pth    # pretrained weights (~9K params, 40 KB)
data/samples.npz     # sample training recording (970 frames)
tests/test_models.py # headless unit tests
```
