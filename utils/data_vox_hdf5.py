import random

import h5py
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader

class HDF5VoxCelebDataset(Dataset):

    def __init__(self, data_csv, label, data_folder, filename):
        super(HDF5VoxCelebDataset, self).__init__()

        data_csv          = pd.read_csv(data_csv)
        self.paths        = data_csv["path"].values
        self.labels       = data_csv[label].values
        self.lengths      = data_csv["lengths"].values
        self.data_folder  = data_folder
        self.filename     = filename

    def __getitem__(self, index):

        fields = self.paths[index].split("/")
        spkid, uttid, fileid = fields[0], fields[1], fields[2]

        path  = self.data_folder + "/" + spkid + ".h5"
        label = self.labels[index]

        with h5py.File(path, "r") as hf:
            data = hf[spkid][uttid][fileid][0]

        return torch.FloatTensor(data).squeeze(dim=1).squeeze(dim=1), label

    def __len__(self):
        return len(self.paths)
