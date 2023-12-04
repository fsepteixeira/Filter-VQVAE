
import torch
import torch.nn.functional as F

class SoftTopKFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):

        with torch.no_grad():
            if epsilon > 1e-2:
                Gamma = SoftTopKFunc.sinkhorn_forward_naive(C, mu, nu, epsilon, max_iter)
                if bool(torch.any(Gamma != Gamma)):
                    print('NaN appeard in Gamma, re-computing...')
                    Gamma = SoftTopKFunc.sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)
            else:
                Gamma = SoftTopKFunc.sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter)

            ctx.save_for_backward(mu, nu, Gamma)
            ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):

        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors

        # mu    - [1,  n,   1]
        # nu    - [1,  1, k+1]
        # Gamma - [bs, n, k+1]

        with torch.no_grad():
            grad_C = SoftTopKFunc.sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon)

        return grad_C, None, None, None, None

    @staticmethod
    def sinkhorn_forward_naive(C, mu, nu, epsilon, max_iter):
        assert(len(C.shape) == 3)

        bs, n, k_ = C.size()

        # "Random" initialization of the sinkhorn algorithm
        v = torch.ones([bs, 1, k_])/(k_) 

        # Put C in Sinkhorn "form"
        G = torch.exp(-C/epsilon)

        # Ensure variable is in the same device as input
        v = v.to(G.device)

        for i in range(max_iter):
            u = mu / (G*v).sum(-1, keepdim=True)
            v = nu / (G*u).sum(-2, keepdim=True)

        # Gamma corresponds to the optimal transport solution
        Gamma = u*G*v

        return Gamma

    @staticmethod
    def sinkhorn_forward_stablized(C, mu, nu, epsilon, max_iter):

        # Compute everything in the log domain
        assert(len(C.shape) == 3)
        bs, n, k_ = C.size()

        k = k_ - 1

        f = torch.zeros([bs, n, 1]).to(C.device)
        g = torch.zeros([bs, 1, k+1]).to(C.device)

        epsilon_log_mu = epsilon*torch.log(mu).to(C.device)
        epsilon_log_nu = epsilon*torch.log(nu).to(C.device)

        min_epsilon_row = lambda Z, eps: -eps*torch.logsumexp((-Z)/eps, -1, keepdim=True)
        min_epsilon_col = lambda Z, eps: -eps*torch.logsumexp((-Z)/eps, -2, keepdim=True)

        for i in range(max_iter):
            f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
            g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

        Gamma = torch.exp((-C + f + g) / epsilon)

        return Gamma

    @staticmethod
    def sinkhorn_backward(grad_output_Gamma, Gamma, mu, nu, epsilon):

        nu_ = nu[:, :, :-1]
        Gamma_ = Gamma[:, :, :-1]

        bs, n, k_ = Gamma.size()

        inv_mu = 1./(mu.view([1, -1]))              # [1, n]
        Kappa = torch.diag_embed(nu_.squeeze(-2))          \
                    -torch.matmul(Gamma_.transpose(-1, -2) \
                    * inv_mu.unsqueeze(-2), Gamma_) # [bs, k, k]
        inv_Kappa = torch.inverse(Kappa) # [bs, k, k]

        Gamma_mu = inv_mu.unsqueeze(-1) * Gamma_
        L  = Gamma_mu.matmul(inv_Kappa) # [bs, n, k]
        G1 = grad_output_Gamma * Gamma  # [bs, n, k+1]

        g1   = G1.sum(-1)
        G21  = (g1 * inv_mu).unsqueeze(-1) * Gamma # [bs, n, k+1]
        g1_L = g1.unsqueeze(-2).matmul(L)          # [bs, 1,   k]
        G22  = g1_L.matmul(Gamma_mu.transpose(-1, -2)).transpose(-1, -2) * Gamma # [bs, n, k+1]
        G23  = -F.pad(g1_L, pad=(0, 1), mode='constant', value=0) * Gamma        # [bs, n, k+1]
        G2   = G21 + G22 + G23                                                   # [bs, n, k+1]

        del g1, G21, G22, G23, Gamma_mu

        g2 = G1.sum(-2).unsqueeze(-1) # [bs, k+1, 1]
        g2 = g2[:, :-1, :]            # [bs, k,   1]

        G31 = -L.matmul(g2) * Gamma   # [bs, n, k+1]
        G32 = F.pad(inv_Kappa.matmul(g2).transpose(-1, -2), 
                    pad=(0, 1), mode='constant', value=0) * Gamma # [bs, n, k+1]
        G3 = G31 + G32                                            # [bs, n, k+1]

        grad_C = (-G1 + G2 + G3) / epsilon # [bs, n, k+1]

        return grad_C

class SoftTopK(torch.nn.Module):

    def __init__(self, k, epsilon=0.1, max_iter=200, largest=True, use_cuda=False):
        super(SoftTopK, self).__init__()

        self.k = k
        self.epsilon = epsilon
        self.max_iter = max_iter

        # Sorted version of top-k
        self.anchors = torch.FloatTensor([k - i for i in range(k + 1)]).view([1, 1, k + 1]) 

        self.largest = largest
        self.use_cuda = use_cuda

        # if use_cuda and torch.cuda.is_available():
        #    self.anchors = self.anchors.cuda()

    def forward(self, X):

        """
        X should correspond to a 2D matrix of BatchSize by N,
        where N is the dimension over which to perform top-k.
        """

        assert(len(X.shape) == 2)

        self.anchors = self.anchors.to(X.device)

        bs, n = X.size()
        X = X.unsqueeze(dim=-1)

        if self.largest:
            # For stability purposes
            X_   = X.clone().detach()
            max_ = torch.max(X_).detach()

            X_[X_ == float('-inf')] = float('inf')

            min_ = torch.min(X_).detach()
            min_value = min_ - (max_ - min_)

            mask = X == float('-inf')
            X = X.masked_fill(mask, min_value)
            ##
        else:

            # We want the largest elements to become the smallest and vice-versa
            X = 1 / (X + 1e-6)  # Screws up the zeros in the diagonal 
                                # -> Must replace them with very large values as well

            # For stability purposes
            X_   = X.clone().detach()

            max_ = torch.max(X_).detach()

            X_[X_ == float('-inf')] = float('inf')

            min_ = torch.min(X_).detach()
            min_value = min_ - (max_ - min_)

            mask = X == float('-inf')
            X = X.masked_fill(mask, min_value)
            ##

        C = (X - self.anchors)**2
        C = C / (C.max().detach())  # Normalize

        self.anchors.cpu()

        mu = torch.ones([1, n, 1], requires_grad=False)/n
        nu = [1./n for _ in range(self.k)]
        nu.append((n-self.k)/n)
        nu = torch.FloatTensor(nu).view([1, 1, self.k+1])

        mu = mu.to(C.device)
        nu = nu.to(C.device)

        Gamma = SoftTopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter)
        A = Gamma[:, :, :self.k] * n

        return A, None

