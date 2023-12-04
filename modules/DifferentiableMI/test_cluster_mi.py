import torch
import numpy as np

from src.ClusterMI     import ClusterMI
from src.DiffClusterMI import DiffClusterMI

from sklearn.feature_selection import mutual_info_regression

def sample_uniform_distributions(means, stds):
    return np.random.normal(means, stds, len(means))

def main():
    # mean = np.array([0.0, 0.0]) #, 0.0])
    # std  = np.array([1.0, 1.0]) #, 1.1])

    # mean = np.array([100.0, -100.1, 20.0])
    # std  = np.array([0.1, 0.8, 2.5])
    # amplitude = np.array([0.9, 1.0, 0.1])

    mean = np.array([0.4, 0.5, 0.8])
    std  = np.array([0.2, 0.3, 0.25])
    amplitude = np.array([0.2, 1.0, 0.5])

    amplitude /= np.sum(amplitude)
    cumsum_amplitude = np.cumsum(amplitude)

    k = 5
    n_classes = 3
    n_samples = int(1024)
    distributions = np.argmax(np.random.rand(n_samples) <= np.expand_dims(cumsum_amplitude, axis=1), axis=0)

    means = mean[distributions]
    stds  = std[distributions]
    samples = sample_uniform_distributions(means, stds)

    # MI Scikit-learn
    mi = mutual_info_regression(np.expand_dims(samples, axis=-1), distributions)[0]
    print("MI SkLearn:", (mi / np.log(2)).item())

    # ClusterMI
    clusterMI = ClusterMI(n_classes, k, lambda x, y: (x - y) ** 2)
    MInfo = clusterMI.forward(torch.from_numpy(samples).float().to("cpu"),
                              torch.from_numpy(distributions).long().to("cpu"))
    print("MI:", MInfo)

    # Differential ClusterMI
    diff_clusterMI = DiffClusterMI(n_classes, k, epsilon=0.001, 
                                   max_iter=450, dist_metric=lambda x, y: (x - y) ** 2)

    samples = torch.from_numpy(samples).float().to("cpu")
    samples.requires_grad = True

    MInfo = diff_clusterMI.forward(samples,
                                   torch.from_numpy(distributions).long().to("cpu"))
    print("DiffMI:", MInfo)

if __name__ == "__main__":
    main()
