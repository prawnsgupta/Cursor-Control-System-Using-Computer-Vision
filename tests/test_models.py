"""Unit tests for the FaceCursor model stack.

These tests deliberately avoid importing mediapipe / cv2 / pyautogui so they
run headless (CI, no webcam). Graph edges are synthesized instead of using
FACEMESH_TESSELATION.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.combined import FaceCursorModel
from models.gnn import FaceGNN
from models.pinn import MotionPINN

N_LANDMARKS = 478  # refined face mesh: 468 + 10 iris landmarks
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "models", "face_cursor.pth")
SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples.npz")


def synthetic_graph(n=N_LANDMARKS):
    """A chain graph over n nodes — enough structure for GCNConv message passing."""
    x = torch.randn(n, 3)
    src = torch.arange(0, n - 1)
    dst = torch.arange(1, n)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    return x, edge_index


class TestForwardShapes:
    def test_gnn_outputs_2d_displacement(self):
        x, edges = synthetic_graph()
        out = FaceGNN()(x, edges)
        assert out.shape == (2,)

    def test_pinn_refines_displacement(self):
        out = MotionPINN()(torch.tensor([0.5, -0.3]))
        assert out.shape == (2,)
        assert torch.isfinite(out).all()

    def test_combined_pipeline(self):
        x, edges = synthetic_graph()
        out = FaceCursorModel()(x, edges)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()

    def test_pinn_finite_on_extreme_inputs(self):
        out = MotionPINN()(torch.tensor([1e6, -1e6]))
        assert torch.isfinite(out).all()


class TestCheckpointCompatibility:
    @pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="no checkpoint in repo")
    def test_pretrained_weights_load_strict(self):
        """Guards against architecture drift: the shipped .pth must always
        match the code's layer names and shapes exactly."""
        model = FaceCursorModel()
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state, strict=True)

    @pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="no checkpoint in repo")
    def test_pretrained_model_is_lightweight(self):
        state = torch.load(CHECKPOINT, map_location="cpu")
        n_params = sum(v.numel() for v in state.values())
        assert n_params < 50_000, "model is meant to stay CPU-real-time tiny"

    @pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="no checkpoint in repo")
    def test_eval_forward_is_deterministic(self):
        model = FaceCursorModel()
        model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
        model.eval()
        x, edges = synthetic_graph()
        with torch.no_grad():
            a = model(x, edges)
            b = model(x, edges)
        assert torch.equal(a, b)


class TestTrainingData:
    @pytest.mark.skipif(not os.path.exists(SAMPLES), reason="no dataset in repo")
    def test_samples_have_expected_structure(self):
        raw = np.load(SAMPLES, allow_pickle=True)["data"]
        assert len(raw) > 0
        landmarks, nose = raw[0]
        assert np.asarray(landmarks).shape == (N_LANDMARKS, 3)
        assert len(nose) == 2
