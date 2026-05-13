import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class FaceGNN(nn.Module):
    """Two-layer Graph Convolutional Network over the face mesh.
    Processes all 468 landmarks as graph nodes, then extracts the
    nose node's features to predict cursor displacement (dx, dy)."""

    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc = nn.Linear(128, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # Extract the nose tip node (index 1) after graph convolutions
        nose_feat = x[1]
        return self.fc(nose_feat)