import torch

class GradientReversal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

class GradientReversalLayer(torch.nn.Module):

    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    @staticmethod
    def forward(input):
        return GradientReversal.apply(input)
