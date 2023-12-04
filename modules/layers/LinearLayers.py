
import torch

class LinearBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim, activation):
        super(LinearBlock, self).__init__()

        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.bn     = torch.nn.BatchNorm1d(out_dim)

        self.activation = activation()

    def forward(self, X, dropout_p=None):
        output = self.bn(self.activation(self.linear(X)))

        if dropout_p is not None:
            output = torch.nn.functional.dropout(output, p=dropout_p, training=self.training)

        return output

class GatedLinearBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GatedLinearBlock, self).__init__()

        self.linear_i = torch.nn.Linear(in_dim, out_dim)
        self.linear_g = torch.nn.Linear(in_dim, out_dim)
        self.bn     = torch.nn.BatchNorm1d(out_dim)

    def forward(self, X, dropout=None):
        output = self.bn(self.linear_i(X) * torch.sigmoid(self.linear_g(X)))

        if dropout is not None:
            output = torch.nn.functional.dropout(output, p=dropout, training=self.training)

        return output
