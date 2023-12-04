import torch
import torch.nn as nn

import numpy as np
from scipy.special import gamma

from .Functions.StraighThroughHeaviside import StraightThroughHeaviside as STHeaviside
from .utils.distances import cosine_distance_2d, pairwise_dists_, pairwise_cosine_dists_, pairwise_euclidean_dists_, PairwiseMAEDists, PairwiseInfDists

class DiffClusterMISTcc_bias(nn.Module):
    """
    Implementation of Nearest Neighbors approach to the computation of the MI (KSG) between
    continuous datasets.
    Based on the method described by Kraskov, Stogbauer and Grassberger (KSG), Algorithm 2,
     in "Estimating Mutual Information"
     https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
    """

    def __init__(self, parameters):
        super(DiffClusterMISTcc_bias, self).__init__()

        self.k = parameters["k"]
        assert self.k > 0, "k must be strictly positive."

        self.k_digamma = torch.digamma(torch.tensor(self.k).float())

        self.distance_x = pairwise_euclidean_dists_
        self.distance_y = pairwise_euclidean_dists_
        self.distance_xy = pairwise_euclidean_dists_

        self.st_heaviside = STHeaviside()

    def forward(self, X, y):
        return self._mutual_information(X, y)

    def _mutual_information(self, x, y):
        
        # Ensure X is bi-dimensional
        if x.ndim <= 1:
            x = x.reshape(-1, 1)

        # Ensure y is bi-dimensional
        if y.ndim <= 1:
            y = y.reshape(-1, 1)

        # x and y are assumed to be NxF, with N the number of samples, and F the number of "features"        
        dx = x.shape[1]
        dy = y.shape[1]

        cx = (torch.pi**(dx/2.0)) / gamma(dx/2.0 + 1)
        cy = (torch.pi**(dy/2.0)) / gamma(dy/2.0 + 1)
        cxy = (torch.pi**((dx+dy)/2.0)) / gamma((dx+dy)/2.0 + 1)

        c_log = np.log((cx * cy) / cxy)

        # Total number of samples
        N = x.shape[0]
        assert (x.shape[0] == y.shape[0])

        x = x.float()
        y = y.float()

        # Compute N dependent term
        N_log = torch.log(torch.tensor(N).float())

        # Non-data-dependent terms
        mutual_information = N_log + c_log + self.k_digamma

        # Create joint variable
        xy = torch.cat((x, y), dim=1)

        # Compute pairwise distances between all vectors in matrix x
        dists_matrix_xy = self.distance_xy(xy)  # Symmetric matrix N x N

        # Compute pairwise distances between all vectors in matrix x
        dists_matrix_x = self.distance_x(x)  # Symmetric matrix N x N

        # Compute pairwise distances between all vectors in matrix y
        dists_matrix_y = self.distance_y(y)  # Symmetric matrix N x N

        # Get k neighbors
        _, anchor_dists_xy_idx = torch.topk(dists_matrix_xy,
                                            self.k + 1, dim=1, 
                                            largest=False)  # k+1 because of diagonal 0s

        # Get k neighbors
        #_, anchor_dists_y_idx = torch.topk(dists_matrix_y,
        #                                    self.k + 1, dim=1, 
        #                                    largest=False)  # k+1 because of diagonal 0s

        # Select dx and dy corresponding to the kth neighbor -- Algorithm 2 of KSG
        anchor_dists_x = dists_matrix_x.gather(0, anchor_dists_xy_idx[:, -1].unsqueeze(dim=1))
        anchor_dists_y = dists_matrix_y.gather(0, anchor_dists_xy_idx[:, -1].unsqueeze(dim=1))

        # Count number of samples with distance less or equal to anchor in both spaces
        gtz_x = self.st_heaviside(anchor_dists_x - dists_matrix_x)
        gtz_y = self.st_heaviside(anchor_dists_y - dists_matrix_y)

        nx_i = torch.sum(gtz_x, dim=1) 
        ny_i = torch.sum(gtz_y, dim=1)
        
        # Compute digamma of count
        nx_i_log = torch.log(nx_i.float() + 1e-7)
        ny_i_log = torch.log(ny_i.float() + 1e-7)

        n_avg_log = (nx_i_log + ny_i_log).mean()

        # Subtract sample-dependent term from sample statistics dependent term 
        mutual_information = mutual_information.to(x.device)
        mutual_information = mutual_information - n_avg_log

        # To have value in bits
        mutual_information = mutual_information / torch.log(torch.tensor(2.0))

        # The mutual information can't be less than zero
        return torch.nn.functional.relu(mutual_information)

