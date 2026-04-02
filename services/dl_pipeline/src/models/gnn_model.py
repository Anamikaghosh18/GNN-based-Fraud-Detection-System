import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # Layer 1: Input to Hidden
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        # Layer 2: Hidden to Hidden (This was 64 in Colab, not 1!)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        # Final Linear layer for prediction
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x) # Prediction layer
        return x

def build_hetero_model(metadata, hidden_channels, out_channels):
    model = GNNModel(hidden_channels, out_channels)
    model = to_hetero(model, metadata, aggr='sum')
    return model