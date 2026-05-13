# Face Cursor Project Final With Hands

This project allows you to control your computer's cursor, scroll, volume, and screen brightness using only your facial movements and hand gestures. It relies on advanced computer vision powered by **MediaPipe** (Face Mesh & Hands) alongside a custom PyTorch model (featuring Graph Neural Networks and Physics-Informed Neural Networks) for cursor stabilization and velocity prediction.

## Features

- **Face-Controlled Cursor**: Move the cursor by moving your head. The movement leverages a deep learning model to translate your nose movements into smooth cursor actions.
- **Mouth Clicks**: Open your mouth slightly to trigger a mouse click. Moving your head to the right while doing so triggers a right-click, otherwise, it registers as a left-click.
- **Head Scrolling**: Toggle scroll mode and simply move your head up and down to scroll pages vertically.
- **Hand Gestures for Volume & Brightness**:
  - **Right Hand Pinch**: Pinch your thumb and index finger together to control system volume.
  - **Left Hand Pinch**: Pinch your thumb and index finger together to control your screen's brightness.

## The Architecture: GNN & PINN

The machine learning backend (`models/combined.py`) uses a state-of-the-art hybrid architecture to ensure the cursor perfectly mimics the dynamics of human head movements without stutter or jitter:

- **Graph Neural Network (GNN)**: The `FaceGNN` architecture ingests the 468 3D facial landmarks from MediaPipe. Instead of treating them as a flat array, the GNN uses the physical connections between facial points (`FACEMESH_TESSELATION`) to perform Graph Convolutions. It understands the structural geometry of the face to extract highly accurate, spatially-aware movement targets based predominantly on the nose node.
- **Physics-Informed Neural Network (PINN)**: The output of the GNN is piped into `MotionPINN`. This layer acts as a biomechanical constraint and smoother. Rather than raw chaotic predictions, the PINN enforces smooth, physical kinematic continuity using non-linear transformations (`Tanh`), effectively filtering out unnatural high-frequency facial twitches to produce stable cursor velocity vectors.

## Prerequisites

Make sure you have Python 3 installed. Run the following command to install the necessary dependencies:

```bash
pip install opencv-python mediapipe torch torch-geometric pyautogui screen-brightness-control pycaw numpy
```

## How to Run

To launch the real-time controller interface, run:

```bash
python run.py
```

It will initialize your webcam. Ensure your face and hands are clearly visible in the camera. Focus the OpenCV window if you want to use the hotkeys below. 

### Hotkeys & Controls

| Key | Action |
| --- | --- |
| `E` | Toggle **Cursor Mode** on/off |
| `S` | Toggle **Scroll Mode** on/off |
| `V` | Toggle **Volume Mode** (Use right hand pinch) |
| `B` | Toggle **Brightness Mode** (Use left hand pinch) |
| `D` | **Disable All Modes** |
| `Q` | **Quit** the application |

## Custom Calibration & Training (Optional)

If you find that the default model doesn't suit your distinct head movements, you can easily train the system yourself!

> [!IMPORTANT]
> **Data Collection and Training Must be Sequential, Not Simultaneous.**
> Do not run data collection and training at the same time in multiple terminals. `train.py` requires the complete data file (`data/samples.npz`) which is only generated *after* `collect_data.py` finishes and is safely closed.

**How to step-by-step get the best training results:**

1. **Collect your own data**:
   ```bash
   python collect_data.py
   ```
   **Tips for the best result:** 
   - Ensure your room is well-lit so facial landmarks track flawlessly.
   - Look directly into the camera and move your head continuously and *smoothly* across your entire natural vertical and horizontal range. 
   - Be sure to trace the edges/corners of the screen to teach the model its spatial limits.
   - Collect data for at least 30-60 seconds.
   - Press `Q` when you are done to save the data to `data/samples.npz`.

2. **Train your model**:
   Once step 1 is complete and the terminal closes, start training:
   ```bash
   python train.py
   ```
   This will train the GNN+PINN pipeline on your newly recorded head velocities, calculating the lowest training loss over 8 epochs. The custom weights will be automatically saved to `models/face_cursor.pth` for immediate use next time you run `run.py`!
