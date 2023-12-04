
import os
import pandas as pd
import warnings
import h5py
import math
import joblib
import speechbrain
import torch
import torch.optim
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data         import Dataset, DataLoader
from speechbrain.utils.metric_stats  import ErrorRateStats, EER, minDCF

from utils import HDF5VoxCelebVeriDataset

class VerificationTrainer(pl.LightningModule):

    def __init__(self, parameters):

        super(VerificationTrainer, self).__init__()

        self.parameters_data = parameters
        self.num_workers     = parameters["num_workers"]

        self.save_verification_scores = parameters["save_verification_scores"]
        self.save_score_path          = parameters["save_score_path"] if self.save_verification_scores else ""

        self.verification_dataset = HDF5VoxCelebVeriDataset(parameters["data_csv"],
                                                            parameters["data_folder"])

        # Save hyper-parameters
        self.save_hyperparameters()

    def on_test_epoch_start(self):
        self.scores = {0: [], 1: []}
        self.verification_info = []

    def test_step(self, batch, batch_idx):
        return self.verify(batch)

    def on_test_epoch_end(self, outputs=None):
        eer,     eer_th     = EER(torch.tensor(self.scores[1]),    torch.tensor(self.scores[0]))
        min_dcf, min_dcf_th = minDCF(torch.tensor(self.scores[1]), torch.tensor(self.scores[0]))
        self.log("verification_EER",       eer)
        self.log("verification_EER_th",    eer_th)
        self.log("verification_minDCF",    min_dcf)
        self.log("verification_minDCF_th", min_dcf_th)
        
        if self.save_verification_scores:
            scores_ = pd.DataFrame(data=self.verification_info)
            path = "/".join(self.save_score_path.split("/")[:-1])
            if not os.path.exists(path):
                os.makedirs(path)
            scores_.to_csv(self.save_score_path, sep=" ", index=False)
                

    def verify(self, batch):
        # Get batch
        enroll, test, label, id_0, id_1 = batch

        with torch.no_grad():
            # Compute original score, for sanity check
            score = F.cosine_similarity(enroll, test)

        # Save score
        self.scores[label.item()].append(score.item())

        self.verification_info.append({"score": score.item(),
                                       "label": label.item(), "key_0": id_0, "key_1": id_1})

    def verification_dataloader(self):
        return DataLoader(self.verification_dataset,
                          batch_size=1, num_workers=1)

