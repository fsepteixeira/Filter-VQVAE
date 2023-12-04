import torch
import torch.nn as nn
import warnings

from .Functions.SoftTopK import SoftTopK
from .Functions.StraighThroughHeaviside import StraightThroughHeaviside as STHeaviside
from .utils.distances import cosine_distance_2d, pairwise_dists_, pairwise_cosine_dists_, pairwise_euclidean_dists_

class DiffClusterMIST(nn.Module):

    """

    Implementation of Nearest Neighbors approach to the computation of the MI between
    discrete and continuous datasets.

    Based on the method described in:
    https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable

    """

    def __init__(self, parameters):
        super(DiffClusterMIST, self).__init__()

        assert parameters["n_classes"] >= 2, "Number of classes. Needs to be larger than or equal to two - " \
                               + str(parameters["n_classes"]) + " given."

        self.k = parameters["k"]
        self.k_digamma = torch.digamma(torch.tensor(self.k).float())
        self.n_classes = parameters["n_classes"]

        if parameters["dist_metric"] == 'cosine':
            self.distance = pairwise_cosine_dists_
        elif parameters["dist_metric"] == "euclidean":
            self.distance = pairwise_euclidean_dists_
        else:
            raise NotImplementError()

        self.st_heaviside = STHeaviside()

    def forward(self, X, y):
        return self._mutual_information(X, y)

    def _mutual_information(self, X, y):
        device = X.device

        # Total number of samples
        N = X.shape[0]
        N_digamma = torch.digamma(torch.tensor(N).float())

        # Number of samples per class
        N_x = torch.FloatTensor([torch.sum(y == i) for i in range(0, self.n_classes)])

        # Compute average <digamma(N_x)>
        N_x_w        = N_x / N
        N_x_digammas = torch.digamma(N_x)
        avg_N_x      = torch.sum(N_x_w * N_x_digammas)

        # Compute pairwise distances between all vectors in matrix X (assumed to be 2d)
        dists_matrix = self.distance(X)  # Symmetric matrix N x N

        classes = list(set(y.tolist()))
        classes.sort()

        mutual_information = N_digamma - avg_N_x + self.k_digamma
        anchor_dists_all = torch.zeros_like(dists_matrix)
        for i in classes:
            # Get same class anchor distance for each sample
            y_same_class = (y == i)
            dists_same_class = torch.index_select(dists_matrix,     0, y_same_class.nonzero().squeeze())
            dists_same_class = torch.index_select(dists_same_class, 1, y_same_class.nonzero().squeeze())

            # Apply top-k operator 
            try:
                anchor_dists, _ = torch.topk(dists_same_class, self.k+1, dim=1, largest=False)  # bs x N x k
            except:
                warnings.warn("Not enough neighbors for MI calculation, returning 0 and skipping...")
                return 0

            anchor_dists = anchor_dists[:, -1]  # N x 1 - take kth element (last one)
            anchor_dists_all[y_same_class] = anchor_dists.unsqueeze(dim=-1)

        # Count number of samples with distance less than anchor except the original value
        gtz = self.st_heaviside(anchor_dists_all - dists_matrix)
        m_i = torch.sum(gtz, dim=1) - 1

        # Compute digamma of soft count
        m_i_digamma = torch.digamma(m_i.float() + 1e-7)
        avg_m_i     = torch.mean(m_i_digamma)

        # Subtract sample-dependent term from sample statistics dependent term 
        mutual_information = mutual_information.to(avg_m_i.device)
        mutual_information -= avg_m_i

        # To have value in bits
        mutual_information = mutual_information / torch.log(torch.tensor(2.0))

        # The mutual information can't be less than zero
        return torch.nn.functional.relu(mutual_information)

