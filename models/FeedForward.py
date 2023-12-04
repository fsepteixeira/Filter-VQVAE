
import torch
from modules import LinearBlock

class FeedForward(torch.nn.Module):

    def __init__(self, parameters):

        super(FeedForward, self).__init__()
        self.layers = torch.nn.ModuleList([])

        for i in range(0, parameters["n_layers"]):
            self.layers.append(
                LinearBlock(in_dim=parameters["in_dim"][i], 
                            out_dim=parameters["out_dim"][i],
                            activation=parameters["activation"])
            )

    def forward(self, X, dropout=None):
        for layer in self.layers:
            X = layer(X, dropout)
        return X
