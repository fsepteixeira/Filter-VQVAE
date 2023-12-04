import torch

class STHeaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.heaviside(input, torch.zeros_like(input).to(input.device))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        with torch.enable_grad():
            output = torch.sigmoid(input)
            return torch.autograd.grad(output, input, grad_output)

        # if grad_output is None:
        #    return None
        # else:
        #    return torch.sigmoid(input) * (1-torch.sigmoid(input)) * grad_output

class StraightThroughHeaviside(torch.nn.Module):

    def __init__(self):
        super(StraightThroughHeaviside, self).__init__()

    def forward(self, input):
        return STHeaviside.apply(input)
