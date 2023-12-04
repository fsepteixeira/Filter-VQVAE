import torch
import torch.nn.functional as F

from math import floor

def cosine_distance_2d(x, y):
    return 1-F.cosine_similarity(x, y, dim=1)

def pairwise_cosine_dists_(x):
    x_norm = F.normalize(x)
    dist = 1 - torch.mm(x_norm, x_norm.T)
    dist = dist.fill_diagonal_(0)
    return dist

def pairwise_euclidean_dists_(x):
    non_diag_dists = F.pdist(x, p=2)
    dists = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)

    dists[idx[0], idx[1]] = non_diag_dists
    dists[idx[1], idx[0]] = non_diag_dists

    return dists

def pairwise_dists_(x, dist_fn, fill_diagonal=False, diag_value=10e6):

    # Get number of elements in the vector
    N = x.shape[0]

    # Compute number of necessary rotations
    rot = int(floor(N/2))

    # Is N odd or even?
    rem = N % 2

    # Indices vector
    idx = torch.ones(N).long().to(x.device)
    idx[0] = 0
    idx = torch.cumsum(idx, 0)

     # Initial x_rot corresponds to x
    x_rot = x
    idx_rot = idx

    # Init dists matrix || could probably be made more efficient in terms of memory
    # (e.g. torch.triu)
    if fill_diagonal:
        dists = diag_value * torch.ones(N, N).to(x.device)
    else:
        dists = torch.zeros(N, N).to(x.device)

    # This cycle can be parallelized
    for i in range(0, rot):

        # Rotate matrix
        x_rot = torch.roll(x_rot, 1, dims=0)
        idx_rot = torch.roll(idx_rot, 1, dims=0)    # Could be done offline,
                                                    # but computational cost
                                                    # is most likely negligible

        # If N is even, the last rotation only
        # requires computing half of the distances.
        # Discard the remaining elements.
        if i == (rot-1) and rem == 0:
            x     = x[0:int(N/2)]
            x_rot = x_rot[0:int(N/2)]

            idx     = idx[0:int(N/2)]
            idx_rot = idx_rot[0:int(N/2)]

        # Compute dists
        dists_ = dist_fn(x, x_rot)

        # Save to matrix
        dists[idx, idx_rot] = dists_
        dists[idx_rot, idx] = dists_    # Matrix is symmetric,
                                        # Main diagonal is filled with zeros dist(x,x) = 0
    return dists

class PairwiseMAEDists(torch.nn.Module):

    def __init__(self, range_):
        super(PairwiseMAEDists, self).__init__()
        assert range_ > 0
        self.range_ = range_

    def forward(self, x):
        non_diag_dists = F.pdist(x, p=1) / float(self.range_)
        dists = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
        idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)

        dists[idx[0], idx[1]] = non_diag_dists
        dists[idx[1], idx[0]] = non_diag_dists

        return dists

class PairwiseInfDists(torch.nn.Module):

    def __init__(self):
        super(PairwiseInfDists, self).__init__()

    def forward(self, x):
        non_diag_dists = F.pdist(x, p=float("inf"))
        dists = torch.zeros((x.shape[0], x.shape[0]), device=x.device)
        idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)

        dists[idx[0], idx[1]] = non_diag_dists
        dists[idx[1], idx[0]] = non_diag_dists

        return dists
