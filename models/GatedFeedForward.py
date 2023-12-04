
import torch
from modules import GatedLinearBlock

class GatedFeedForward(torch.nn.Module):

    def __init__(self, parameters):

        super(GatedFeedForward, self).__init__()
        self.layers = torch.nn.ModuleList([])

        for i in range(0, parameters["n_layers"]):
            self.layers.append(
                GatedLinearBlock(in_dim=parameters["in_dim"][i], 
                                 out_dim=parameters["out_dim"][i])
            )

    def forward(self, X, dropout=None):
        for layer in self.layers:
            X = layer(X, dropout=dropout)
        return X
