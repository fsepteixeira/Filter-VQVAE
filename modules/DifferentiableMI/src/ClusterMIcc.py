
import torch
import torch.nn as nn

from .utils.distances import cosine_distance_2d, pairwise_dists_, pairwise_cosine_dists_, pairwise_euclidean_dists_, PairwiseMAEDists, PairwiseInfDists

class ClusterMIcc(nn.Module):
    """
    Implementation of Nearest Neighbors approach to the computation of the MI (KSG) between
    continuous datasets.
    Based on the method described by Kraskov, Stogbauer and Grassberger (KSG), Algorithm 2,
     in "Estimating Mutual Information"
     https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
    """

    def __init__(self, parameters):
        super(ClusterMIcc, self).__init__()

        self.k = parameters["k"]
        assert self.k > 0, "k must be strictly positive."

        self.k_digamma = torch.digamma(torch.tensor(self.k).float()) - (1/self.k)

        if parameters["dist_metric_x"] == 'cosine':
            self.distance_x = pairwise_cosine_dists_
        elif parameters["dist_metric_x"] == "euclidean":
            self.distance_x = pairwise_euclidean_dists_
        elif parameters["dist_metric_x"] == "inf_dist":
            self.distance_x = PairwiseInfDists()
        else:
            raise NotImplementError()

        if parameters["dist_metric_y"] == 'cosine':
            self.distance_y = pairwise_cosine_dists_
        elif parameters["dist_metric_y"] == "euclidean":
            self.distance_y = pairwise_euclidean_dists_
        elif parameters["dist_metric_y"] == "mae_dist":
            self.distance_y = PairwiseMAEDists(parameters["range_"])
        elif parameters["dist_metric_y"] == "inf_dist":
            self.distance_y = PairwiseInfDists()
        else:
            raise NotImplementError()

    def forward(self, X, y):
        return self._mutual_information(X, y)

    def _mutual_information(self, x, y):
        # x and y are assumed to be NxF, with N the number of samples, and F the number of "features"        

        # Ensure X is bi-dimensional
        if x.ndim <= 1:
            x = x.reshape(-1, 1)

        # Ensure y is bi-dimensional
        if y.ndim <= 1:
            y = y.reshape(-1, 1)

        # Total number of samples
        N = x.shape[0]
        assert (x.shape[0] == y.shape[0])

        x = x.float()
        y = y.float()

        # Compute N dependent term
        N_digamma = torch.digamma(torch.tensor(N).float())

        # Non-data-dependent terms
        mutual_information = N_digamma + self.k_digamma

        # Compute pairwise distances between all vectors in matrix x
        dists_matrix_x = self.distance_x(x)  # Symmetric matrix N x N

        # Compute pairwise distances between all vectors in matrix y
        dists_matrix_y = self.distance_y(y)  # Symmetric matrix N x N

        # Pairwise distances for xy correspond to maximum{d_x, d_y}
        dists_matrix_xy = torch.maximum(dists_matrix_x, dists_matrix_y) # Symmetric matrix N x N

        # Get k neighbors
        _, anchor_dists_xy_idx = torch.topk(dists_matrix_xy, 
                                            self.k + 1, dim=1, 
                                            largest=False)  # k+1 because of diagonal 0s

        # Select dx and dy corresponding to the kth neighbor -- Algorithm 2 of KSG
        anchor_dists_x = dists_matrix_x[anchor_dists_xy_idx[:, -1]]
        anchor_dists_y = dists_matrix_y[anchor_dists_xy_idx[:, -1]]

        # Count number of samples with distance less or equal to anchor in both spaces
        nx_i = torch.sum(torch.le(dists_matrix_x, anchor_dists_x), dim=1)
        ny_i = torch.sum(torch.le(dists_matrix_y, anchor_dists_y), dim=1)

        # Compute digamma of count
        nx_i_digamma  = torch.digamma(nx_i.float() + 1)
        ny_i_digamma  = torch.digamma(ny_i.float() + 1)
        n_avg_digamma = (nx_i_digamma + ny_i_digamma).mean()

        # Subtract sample-dependent term from sample statistics dependent term 
        mutual_information = mutual_information.to(x.device)
        mutual_information -= n_avg_digamma

        # To have value in bits
        mutual_information = mutual_information / torch.log(torch.tensor(2.0))

        # The mutual information can't be less than zero
        return torch.nn.functional.relu(mutual_information)

