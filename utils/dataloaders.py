import h5py
import os
import random

import torch
import torchaudio

import numpy as np
import pandas as pd

from speechbrain.dataio.preprocess import AudioNormalizer
from torch.utils.data import Dataset, DataLoader

from .balanced_sampler import BalancedBatchSampler

class PadBatchWithLengths(torch.nn.Module):
    """
    Class to pad batches of sequences - Also sorts by sequence length from shortest to largest
    : X - List of uneven sequence matrices
    : y - Corresponding labels to sort
    : (Optional) lengths - Numpy array containing the length of each sequence in X
    Returns:
    : X - Padded with zeros, sorted by sequence length
    : y - Sorted in the same way as X
    """

    def __init__(self):
        super(PadBatchWithLengths, self).__init__()

    def __call__(self, data):
        # Get data arrays
        data_tuple = zip(*data)

        if len(data) == 0:
            raise ValueError("DataLoader did not return any samples... Aborting.")

        if len(data[0]) == 3:
            X, ids, y = data_tuple
            lengths = False
        elif len(data[0]) == 4:
            lengths = True
            X, ids, y, lens = data_tuple

        else:
            raise ValueError("Dataloader returned too many parameters. Dataloader should return (X,y) or (X,y,l) \
                              where l are the lengths of each element in the batch.")

        if not lengths:
            lens = [len(x) for x in X]

        lens = torch.FloatTensor(lens).requires_grad_(False)
        lens = lens / lens.max()
        y    = torch.LongTensor(y).requires_grad_(False)

        if len(X) == 1:
            return X[0].unsqueeze(dim=0).requires_grad_(False), ids, y, lens

        # Pad sequences
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True).requires_grad_(False)

        return X, ids, y, lens

class VoxCelebDataset(Dataset):

    def __init__(self, data_csv, label, random_chunk=False, chunk_length=3, sample_rate=16000, data_folder=None, max_duration=None):
        super(VoxCelebDataset, self).__init__()

        data_csv = pd.read_csv(data_csv)
        paths    = data_csv["path"].values

        self.paths   = data_csv["path"].values
        self.ids     = ["-".join(p.split(".")[0].split("/")[-3:]) for p in paths]

        if "lengths" in data_csv.keys():
            self.lengths = data_csv["lengths"].values
        self.labels  = data_csv[label].values

        self.random_chunk = random_chunk
        self.chunk_length = chunk_length
        self.sample_rate  = sample_rate
        self.max_duration = max_duration

        self.normalizer   = AudioNormalizer()
        self.data_folder  = "" if data_folder is None else data_folder + "/"

    def __getitem__(self, index):

        paths  = self.data_folder + self.paths[index]
        ids    = self.ids[index]
        labels = self.labels[index]

        if self.random_chunk:
            duration_sample = int(self.lengths[index] * self.sample_rate)

            start = random.randint(0, duration_sample - int(self.chunk_length * self.sample_rate) - 1)
            stop  = start + int(self.chunk_length * self.sample_rate)

            num_frames = stop - start
            wav, sr = torchaudio.load(paths, num_frames=num_frames, frame_offset=start, channels_first=False)

        else:
            if self.max_duration is not None:
                if self.lengths[index] > self.max_duration:
                    stop = int(self.max_duration*self.sample_rate)
                else:
                    stop = int(self.lengths[index] * self.sample_rate)
                wav, sr = torchaudio.load(paths, num_frames=stop, channels_first=False)
            else:
                wav, sr = torchaudio.load(paths, channels_first=False)

        wav = self.normalizer(wav, sr)
        return wav.squeeze(), ids, labels

    def __len__(self):
        return len(self.paths)

def create_dataloaders_voxceleb(parameters):

    batch_size  = parameters["batch_size"]
    num_workers = parameters["num_workers"]

    train_dataset = VoxCelebDataset(parameters["train_csv"],    parameters["label"], 
                                    parameters["random_chunk"], parameters["chunk_length"], 
                                    parameters["sample_rate"], 
                                    parameters["data_folder"] if "data_folder" in parameters.keys() else None)

    dev_dataset = VoxCelebDataset(parameters["dev_csv"],      parameters["label"],
                                  parameters["random_chunk"], parameters["chunk_length"], 
                                  parameters["sample_rate"], 
                                  parameters["data_folder"] if "data_folder" in parameters.keys() else None)

    eval_dataset  = VoxCelebDataset(parameters["eval_csv"],  parameters["label"], False,
                                    data_folder=parameters["data_folder"]
                                                  if "data_folder" in parameters.keys() else None)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                                  shuffle=True,  collate_fn=PadBatchWithLengths())
    dev_dataloader   = DataLoader(dev_dataset,   batch_size=batch_size, num_workers=num_workers, 
                                  shuffle=False, collate_fn=PadBatchWithLengths())
    eval_dataloader  = DataLoader(eval_dataset,  batch_size=batch_size, num_workers=num_workers, 
                                  shuffle=False, collate_fn=PadBatchWithLengths())

    return {"train": train_dataloader, "dev": dev_dataloader, "eval": eval_dataloader}

class HDF5VoxCelebDataset(Dataset):

    def __init__(self, data_csv, label, data_folder=None, return_full_id=False, return_trg_label=False):
        super(HDF5VoxCelebDataset, self).__init__()

        data_csv = pd.read_csv(data_csv, dtype={"spkid": str})
        paths    = data_csv["path"].values

        self.paths   = data_csv["path"].values
        self.ids     = ["-".join(p.split(".")[0].split("/")[-3:]) for p in paths]
        self.labels  = data_csv[label].values

        self.data_folder  = data_folder 
        self.return_full_id = return_full_id
        self.return_trg_label = return_trg_label

    def __getitem__(self, index):

        path  = self.paths[index]
        key   = self.ids[index]
        label = self.labels[index]

        id_      = key.split("-")[0]
        filename = "-".join(key.split("-")[1:-1])
        path     = os.path.join(self.data_folder, id_, filename + ".h5")
        with h5py.File(path, "r") as hf:
            ds = hf[key]
            X = torch.from_numpy(np.array(ds)).squeeze(dim=0)

            if self.return_trg_label:
                y = ds.attrs["label_trg"]
            else:
                y = label #ds.attrs["label"]

        if self.return_full_id:
            return X, y, key
        else:
            return X, y, id_

    def __len__(self):
        return len(self.paths)

def create_dataloaders_classifier(parameters):

    batch_size  = parameters["batch_size"]
    num_workers = parameters["num_workers"]

    train_dataset = HDF5VoxCelebDataset(parameters["train_csv"],
                                        parameters["label"], 
                                        parameters["data_folder_train"],
                                        parameters["return_full_id"],
                                        parameters["return_trg_label"])

    dev_dataset   = HDF5VoxCelebDataset(parameters["dev_csv"],
                                        parameters["label"],
                                        parameters["data_folder_dev"],
                                        parameters["return_full_id"],
                                        parameters["return_trg_label"])

    eval_dataset  = HDF5VoxCelebDataset(parameters["eval_csv"],
                                        parameters["label"],
                                        parameters["data_folder_eval"],
                                        parameters["return_full_id"],
                                        parameters["return_trg_label"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_train"])
    dev_dataloader   = DataLoader(dev_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_dev"])
    eval_dataloader  = DataLoader(eval_dataset,  batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_eval"])

    return {"train": train_dataloader, "dev": dev_dataloader, "eval": eval_dataloader}

def create_dataloaders_classifier_tmp(parameters):

    batch_size  = parameters["batch_size"]
    num_workers = parameters["num_workers"]

    train_dataset = HDF5VoxCelebDataset(parameters["train_csv"],
                                        parameters["label"], 
                                        parameters["data_folder_train"],
                                        parameters["return_full_id"]["train"],
                                        parameters["return_trg_label"])

    dev_dataset   = HDF5VoxCelebDataset(parameters["dev_csv"],
                                        parameters["label"],
                                        parameters["data_folder_dev"],
                                        parameters["return_full_id"]["dev"],
                                        parameters["return_trg_label"])

    eval_dataset  = HDF5VoxCelebDataset(parameters["eval_csv"],
                                        parameters["label"],
                                        parameters["data_folder_eval"],
                                        parameters["return_full_id"]["eval"],
                                        parameters["return_trg_label"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_train"])
    dev_dataloader   = DataLoader(dev_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_dev"])
    eval_dataloader  = DataLoader(eval_dataset,  batch_size=batch_size, num_workers=num_workers, shuffle=parameters["shuffle_eval"])

    return {"train": train_dataloader, "dev": dev_dataloader, "eval": eval_dataloader}

def create_dataloaders_vqvae(parameters):

    batch_size  = parameters["batch_size"]
    num_workers = parameters["num_workers"]

    train_dataset = HDF5VoxCelebDataset(parameters["train_csv"],
                                        parameters["label"], 
                                        parameters["data_folder_train"])

    dev_dataset   = HDF5VoxCelebDataset(parameters["dev_csv"],
                                        parameters["label"],
                                        parameters["data_folder_dev"])

    eval_dataset  = HDF5VoxCelebDataset(parameters["eval_csv"],
                                        parameters["label"],
                                        parameters["data_folder_eval"])

    if not parameters["balanced_train"]:
        train_dataloader = DataLoader(train_dataset,           batch_size=batch_size,
                                      num_workers=num_workers, shuffle=parameters["shuffle_train"])
    else:
        train_dataset_oversample = HDF5VoxCelebDataset(parameters["train_csv"],
                                                       parameters["label_categorical"] if "label_categorical" in parameters.keys() else parameters["label"], 
                                                       parameters["data_folder_train"])

        train_dataloader = DataLoader(train_dataset,
                                      sampler=BalancedBatchSampler(
                                      train_dataset_oversample,
                                      shuffle=parameters["shuffle_train"],
                                      oversample=parameters["oversample_train"]),
                                      num_workers=num_workers)

    if not parameters["balanced_dev"]:
        dev_dataloader   = DataLoader(dev_dataset,             batch_size=batch_size,
                                      num_workers=num_workers, shuffle=parameters["shuffle_dev"])
    else:
        dev_dataset_oversample = HDF5VoxCelebDataset(parameters["dev_csv"],
                                                     parameters["label_categorical"] if "label_categorical" in parameters.keys() else parameters["label"], 
                                                     parameters["data_folder_dev"])
        dev_dataloader = DataLoader(dev_dataset,
                                    sampler=BalancedBatchSampler(
                                    dev_dataset_oversample,
                                    shuffle=parameters["shuffle_dev"], 
                                    oversample=parameters["oversample_dev"]),
                                    num_workers=num_workers)

    if not parameters["balanced_eval"]:
        eval_dataloader  = DataLoader(eval_dataset,            batch_size=batch_size,
                                      num_workers=num_workers, shuffle=parameters["shuffle_eval"])
    else:
        eval_dataset_oversample = HDF5VoxCelebDataset(parameters["eval_csv"],
                                                     parameters["label_categorical"] if "label_categorical" in parameters.keys() else parameters["label"], 
                                                     parameters["data_folder_eval"])
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=BalancedBatchSampler(
                                     eval_dataset,
                                     shuffle=parameters["shuffle_eval"], 
                                     oversample=parameters["oversample_eval"]),
                                     num_workers=num_workers)


    return {"train": train_dataloader, "dev": dev_dataloader, "eval": eval_dataloader}

class HDF5VoxCelebVeriDataset(Dataset):

    def __init__(self, data_csv, data_folder=None):
        super(HDF5VoxCelebVeriDataset, self).__init__()

        data_csv = pd.read_csv(data_csv, sep=" ", header=None)
        data_csv.columns = ["label", "enroll", "test"]

        self.paths_enroll = data_csv["enroll"].values
        self.paths_test   = data_csv["test"].values

        self.ids_enroll = ["-".join(p.split(".")[0].split("/")[-3:]) for p in self.paths_enroll]
        self.ids_test   = ["-".join(p.split(".")[0].split("/")[-3:]) for p in self.paths_test]

        self.labels  = data_csv["label"].values
        self.data_folder  = data_folder 

    def __getitem__(self, index):

        path_enroll = self.paths_enroll[index]
        path_test   = self.paths_test[index]

        key_enroll  = self.ids_enroll[index]
        key_test    = self.ids_test[index]

        label = self.labels[index]

        id_enroll = key_enroll.split("-")[0]
        id_test   = key_test.split("-")[0]

        filename_enroll = "-".join(key_enroll.split("-")[1:-1])
        filename_test   = "-".join(key_test.split("-")[1:-1])

        path_enroll = os.path.join(self.data_folder, id_enroll, filename_enroll + ".h5")
        path_test   = os.path.join(self.data_folder, id_test,   filename_test   + ".h5")

        with h5py.File(path_enroll, "r") as hf:
            ds = hf[key_enroll]
            X_enroll = torch.from_numpy(np.array(ds)).squeeze(dim=0)

        with h5py.File(path_test, "r") as hf:
            ds = hf[key_test]
            X_test = torch.from_numpy(np.array(ds)).squeeze(dim=0)

        return X_enroll, X_test, label, key_enroll, key_test

    def __len__(self):
        return len(self.labels)
