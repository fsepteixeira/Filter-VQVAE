import torch
import torch.utils.data
import random, math

from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

###
# Adapted from https://github.com/galatolofederico/pytorch-balanced-batch
###

T_co = TypeVar('T_co', covariant=True)

class BalancedBatchSampler(Sampler):

    def __init__(self, dataset:Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, oversample: bool = False) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.labels = dataset.labels
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if oversample:
            self.num_samples_per_class = 0
        else:
            self.num_samples_per_class = float('inf')

        # Save index per sample per class
        self.n_classes = 0
        self.index_per_class = dict()
        for idx in range(0, len(self.dataset)):
            label = self.labels[idx]

            if label not in self.index_per_class:
                self.n_classes += 1
                self.index_per_class[label] = list()

            self.index_per_class[label].append(idx)

            if oversample:
                self.num_samples_per_class = len(self.index_per_class[label]) if len(self.index_per_class[label]) > self.num_samples_per_class else self.num_samples_per_class
            else:
                self.num_samples_per_class = len(self.index_per_class[label]) if len(self.index_per_class[label]) < self.num_samples_per_class else self.num_samples_per_class
        
        # Oversample the classes with fewer elements than the max
        if oversample:
            for label in self.index_per_class:
                if not shuffle:
                    random.seed(seed) # For reproducibility
                while len(self.index_per_class[label]) < self.num_samples_per_class:
                    self.index_per_class[label].append(random.choice(self.index_per_class[label]))


        else: # undersample
            for label in self.index_per_class:
                if len(self.index_per_class[label]) > self.num_samples_per_class:
                    self.index_per_class[label] = self.index_per_class[label][:self.num_samples_per_class]

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.num_samples_per_class % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = self.n_classes * math.ceil(
                (self.num_samples_per_class - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = self.n_classes * math.ceil(self.num_samples_per_class / self.num_replicas)  # type: ignore[arg-type]

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = []
            for label in range(self.n_classes):
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch + label)
                indices.append(torch.randperm(len(self.index_per_class[label]), generator=g).tolist())
        else:
            indices = [list(range(len(self.index_per_class[label]))) for label in range(self.n_classes)]  # type: ignore[arg-type]

        for n in range(self.n_classes):
            if not self.drop_last:
                # add extra samples to make it evenly divisible
                padding_size = int(self.num_samples*self.num_replicas/self.n_classes) - len(indices[n])
                if padding_size <= len(indices[n]):
                    indices[n] += indices[n][:padding_size]
                else:
                    indices[n] += (indices[n] * math.ceil(padding_size / len(indices[n])))[:padding_size]
            else:
                # remove tail of data to make it evenly divisible.
                indices[n] = indices[n][:self.total_size]

        total_len = torch.sum(torch.LongTensor([len(indices[n]) for n in range(self.n_classes)])).item()
        assert total_len == self.total_size

        # subsample
        indices = [indices[label][self.rank:int(self.num_samples*self.num_replicas/self.n_classes):self.num_replicas] 
                                                                                        for label in range(self.n_classes)]

        # Cat and interleave
        indices = [torch.LongTensor(self.index_per_class[label])[indices[label]] for label in range(self.n_classes)]
        indices = list(torch.stack([torch.LongTensor(i).unsqueeze(dim=1) for i in indices], dim=2).view(-1).squeeze().tolist())

        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

