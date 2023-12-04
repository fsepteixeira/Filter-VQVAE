import torch

class DropoutScheduler(torch.nn.Module):
    def __init__(self, dropout_p=0.0):
        super(DropoutScheduler, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, idx):
        return self.dropout_p

class HybridLDLLoss(torch.nn.Module):
    def __init__(self, parameters):
        super(HybridLDLLoss, self).__init__()

        self.min_k = parameters["min_k"]
        self.max_k = parameters["max_k"]

        self.lambda_1 = parameters["lambda_1"]  # Weight for KL div
        self.lambda_2 = parameters["lambda_2"]  # Weight for L1 loss
        self.lambda_3 = parameters["lambda_3"]  # Weight for Var loss

        self.sigma = parameters["sigma"]

        self.Ks = torch.ones(self.max_k-self.min_k).cumsum(0).reshape(1, -1)

    def forward(self, x, y):

        x = torch.nn.functional.softmax(x, dim=1)

        loss = 0
        if abs(self.lambda_1) > 0:
            loss = loss + self.lambda_1 * self._ldl_loss(x, y)

        if abs(self.lambda_2) > 0:
            loss = loss + self.lambda_2 * self._l1_loss(x, y)

        if abs(self.lambda_3) > 0:
            loss = loss + self.lambda_3 * self._var_loss(x)

        return loss

    def _label_to_pdf(self, y):
        norm_cst = torch.sqrt(torch.FloatTensor([2*torch.pi])) * self.sigma

        # To ensure y has the correct shape
        y = y.reshape(-1, 1)

        # Compute exponent
        exp = - (y - self.Ks.to(y.device).T)**2 / (2 * self.sigma**2)

        return torch.exp(exp) / norm_cst

    def _ldl_loss(self, x, y):
        return torch.nn.functional.kl_div(x, self._label_to_pdf(y))

    def _l1_loss(self, x, y):
        return torch.nn.functional.l1_loss(torch.matmul(x, self.Ks.to(x.device).T), y)

    def _var_loss(self, x):
        Ks = self.Ks.to(x.device)
        return torch.matmul(x, (torch.matmul(x, Ks.T) - Ks)**2).mean()

