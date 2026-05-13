import torch
import torch.nn as nn


class MotionPINN(nn.Module):
    """Physics-Informed Neural Network for motion smoothing.
    Takes the GNN's raw (dx, dy) prediction and refines it into a
    smoother, more physically plausible cursor correction. The Tanh
    activation naturally bounds intermediate values, suppressing
    jitter while preserving intentional movements."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, pos):
        return self.net(pos)