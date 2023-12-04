
import torch

class EmbeddingProjection(torch.nn.Module):

    def __init__(self, parameters):
        super(EmbeddingProjection, self).__init__()

        self.n_classes      = parameters["n_classes"]
        self.projection_dim = parameters["projection_dim"]

        self.weight_proj = torch.nn.Linear(parameters["n_classes"], parameters["projection_dim"])

        torch.nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        torch.nn.init.zeros_(self.weight_proj.bias)

    def forward(self, Y):
        return self.weight_proj(Y)

class CodeProjection(torch.nn.Module):

    def __init__(self, parameters):
        super(CodeProjection, self).__init__()

        self.in_dim         = parameters["in_dim"]
        self.projection_dim = parameters["projection_dim"]

        self.weight_proj = torch.nn.Linear(parameters["in_dim"], parameters["projection_dim"])

        torch.nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        torch.nn.init.zeros_(self.weight_proj.bias)

    def forward(self, Y):
        return self.weight_proj(Y)
