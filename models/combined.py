import torch
import torch.nn as nn
from models.gnn import FaceGNN
from models.pinn import MotionPINN


class FaceCursorModel(nn.Module):
    """Combined model that chains FaceGNN → MotionPINN.
    First, the GNN extracts spatial features from the face mesh graph
    and predicts a raw cursor displacement. Then the PINN refines that
    prediction into a smooth, stable correction signal."""

    def __init__(self):
        super().__init__()
        self.gnn = FaceGNN()
        self.pinn = MotionPINN()

    def forward(self, x, edge_index):
        pos = self.gnn(x, edge_index)
        smooth_pos = self.pinn(pos)
        return smooth_pos