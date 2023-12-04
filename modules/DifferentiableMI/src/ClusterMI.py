
import torch
from .utils.distances import cosine_distance_2d, pairwise_dists_

class ClusterMI(torch.nn.Module):

    """

    Implementation of Nearest Neighbors approach to the computation of the MI between
    discrete and continuous datasets.

    Based on the method described in:
    https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0087357&type=printable

    """

    def __init__(self, parameters): #n_classes=2, k=3, dist_metric=cosine_distance_2d):
        super(ClusterMI, self).__init__()

        self.k = parameters["k"]
        self.k_digamma = torch.digamma(torch.tensor(parameters["k"]).float())
        self.n_classes = parameters["n_classes"]
        assert self.n_classes >= 2, "Number of classes. Needs to be larger than or equal to two - " \
                               + str(self.n_classes) + " given."

        self.distance  = parameters["dist_metric"]
        self.high_cst  = 10e6

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
        N_x_w = N_x / N
        N_x_digammas = torch.digamma(N_x)
        avg_N_x = torch.sum(N_x_w * N_x_digammas)

        # Compute pairwise distances between all vectors in matrix X (assumed to be 2d)
        dists_matrix = pairwise_dists_(X, self.distance)  # fill_diagonal=True, diag_value=10e6)

        # Broadcast y
        y_mat = y.repeat(N, 1).T    # N x N matrix

        # Get same class anchor distance for each sample
        y_same_class = y_mat == y

        # Distances that don't matter (i.e. from another class or in the diagonal) should be very high
        dists_same_class = torch.where(y_same_class.bool(), dists_matrix,
                                       self.high_cst * torch.ones_like(dists_matrix).to(device))

        anchor_dists, anchor_idx = torch.topk(dists_same_class, self.k+1, dim=1, largest=False)

        # Last dist will be the kth distance
        anchor_dists = anchor_dists[:, -1]

        # Count number of samples with distance less than anchor - weird behavior
        # All that are bellow anchor dist including the original value (d=0.0)
        m_i = torch.sum(torch.le(dists_matrix, anchor_dists.unsqueeze(dim=1)), dim=1) - 1
        m_i_digamma = torch.digamma(m_i.float())
        avg_m_i = torch.mean(m_i_digamma)

        # Final sum
        mutual_information = N_digamma - avg_N_x + self.k_digamma - avg_m_i

        # To have value in bits
        return max((mutual_information / torch.log(torch.tensor(2.0))).item(), 0)

