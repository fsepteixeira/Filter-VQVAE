
import torch
import torch.nn.functional as F

class ReconstructionLoss(torch.nn.Module):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, X, Y):
        return 1-F.cosine_similarity(X, Y).mean()

class CodebookDiversityLoss(torch.nn.Module):

    def __init__(self):
        super(CodebookDiversityLoss, self).__init__()

    @staticmethod
    def forward(self, r):
        return (r["entries_dim"] - r["prob_perplexity"]) / r["entries_dim"]

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
        exp = - (y - self.Ks.to(y.device))**2 / (2 * self.sigma**2)

        return torch.exp(exp) / norm_cst.to(y.device)

    def _ldl_loss(self, x, y):
        return torch.nn.functional.kl_div(x, self._label_to_pdf(y), reduction="batchmean")

    def _l1_loss(self, x, y):
        return torch.nn.functional.l1_loss(torch.matmul(x, self.Ks.to(x.device).T).squeeze(dim=1), y)

    def _var_loss(self, x):
        Ks = self.Ks.to(x.device)
        return torch.mul(x, (torch.matmul(x, Ks.T) - Ks)**2).sum(dim=1).mean()

