import torch

class STTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input)
        return torch.topk(input, k)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        with torch.enable_grad():
            output = torch.sigmoid(input)
            return torch.autograd.grad(output, input, grad_output)

class StraightThroughTopK(torch.nn.Module):

    def __init__(self, k):
        super(StraightThroughTopK, self).__init__()
        self.k = k

    def forward(self, input):
        return STTopK.apply(input, self.k)
