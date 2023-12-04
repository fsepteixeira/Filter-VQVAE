
import pandas as pd
import speechbrain
import torch
import torch.optim
import pytorch_lightning as pl

from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from torch.utils.data import Dataset, DataLoader

from models  import SpeakerIDClassifier
from utils import BalancedBatchSampler, HDF5VoxCelebDataset

from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.pretrained    import EncoderClassifier

class SpeakerIDTrainer(pl.LightningModule):

    def __init__(self, parameters_spk,
                       parameters_training,
                       parameters_data, 
                       parameters_loss):

        super(SpeakerIDTrainer, self).__init__()


        # Init data parameters
        self.batch_size      = parameters_data["batch_size"]
        self.batch_size_val  = parameters_data["batch_size_dev"]
        self.batch_size_test = parameters_data["batch_size_eval"]

        self.num_workers    = parameters_data["num_workers"]

        self.train_dataset = HDF5VoxCelebDataset(parameters_data["train_csv"],
                                                 parameters_data["label"], 
                                                 parameters_data["data_folder_train"],
                                                 parameters_data["return_full_id"])

        self.val_dataset = HDF5VoxCelebDataset(parameters_data["dev_csv"],
                                                 parameters_data["label"], 
                                                 parameters_data["data_folder_dev"],
                                                 parameters_data["return_full_id"])

        self.eval_dataset = HDF5VoxCelebDataset(parameters_data["eval_csv"],
                                                 parameters_data["label"], 
                                                 parameters_data["data_folder_eval"],
                                                 parameters_data["return_full_id"])
        # Parameters used for training
        self.n_spk_classes = parameters_spk["n_speakers"]
        self.id_classifier = SpeakerIDClassifier(parameters_spk)

        self.label_encoder = CategoricalEncoder()
        speaker_ids = pd.read_csv(parameters_data["train_csv"], dtype={"spkid": str})["spkid"]
        self.label_encoder.update_from_iterable(speaker_ids)
        self.label_encoder.add_unk()

        assert len(self.label_encoder) == self.n_spk_classes + 1, str("Specified " + str(self.n_spk_classes) + " speakers. Found " + str(len(self.label_encoder)-1) + " instead.")

        self.parameters_data     = parameters_data
        self.parameters_training = parameters_training

        self.aam_loss            = parameters_loss["aam"]

        # Save hyper-parameters
        self.save_hyperparameters()

    def classify_id(self, emb, ids, stage=None):
        preds = self.id_classifier(emb)
        if stage == "train":
            labels = self.label_encoder.encode_sequence_torch(ids)
        else:
            labels = self.label_encoder.encode_sequence_torch(ids, allow_unk=True)
        return preds, labels

    def training_step(self, batch, batch_idx):
        # Get data from batch
        X, y, ids = batch
        batch_size = X.shape[0]

        # Compute AdditiveAngularMargin loss as in the training of the x-vector network
        spk, y_spk = self.classify_id(X, ids, "train")
        aam_loss = self.aam_loss(spk, y_spk.to(X.device).unsqueeze(dim=1))

        # Total loss
        loss = aam_loss

        # Log everything
        self.log("train_aam_loss", aam_loss, batch_size=batch_size)

        # For logging purposes only
        with torch.no_grad():
            spk_acc = accuracy(spk.argmax(dim=-1).detach().cpu(), y_spk,
                               "multiclass", num_classes=self.n_spk_classes)
            self.log("train_spk_acc", spk_acc, batch_size=batch_size)

        return loss

    def evaluate(self, batch, stage=None, random=False):

        # Get batch
        X, y, ids = batch
        batch_size = X.shape[0]

        with torch.no_grad():

            spk, y_spk = self.classify_id(X, ids)
            spk_acc = accuracy(spk.argmax(dim=-1).cpu(), y_spk, 
                               "multiclass", num_classes=self.n_spk_classes)

        return spk, y_spk 

    def on_validation_epoch_start(self):
        self.preds = []
        self.labels = []

    def on_test_epoch_start(self):
        self.preds = []
        self.labels = []

    def validation_step(self, batch, batch_idx):
        preds, labels = self.evaluate(batch, stage="valid")
        self.preds.append(preds.cpu())
        self.labels.append(labels.cpu())

    def test_step(self, batch, batch_idx):
        preds, labels = self.evaluate(batch, stage="test")
        self.preds.append(preds.cpu())
        self.labels.append(labels.cpu())

    def on_test_epoch_end(self, outputs=None):
        preds = torch.cat(self.preds)
        labels = torch.cat(self.labels)
        spk_acc = accuracy(preds.argmax(dim=1), labels, task="multiclass", num_classes=self.n_spk_classes)
        self.log("test_spk_acc", spk_acc)
        self.preds = []
        self.labels= []

    def on_validation_epoch_end(self, outputs=None):
        preds = torch.cat(self.preds)
        labels = torch.cat(self.labels)
        spk_acc = accuracy(preds.argmax(dim=1), labels, task="multiclass", num_classes=self.n_spk_classes)
        self.log("val_spk_acc", spk_acc)
        self.labels = []
        self.preds = []

    def configure_optimizers(self):
        trainable_parameters = list(self.id_classifier.parameters())

        optimizer = torch.optim.Adam(trainable_parameters,
                                     lr=self.hparams.parameters_training["lr"],
                                     weight_decay=self.hparams.parameters_training["weight_decay"])

        scheduler_dict = {"scheduler": OneCycleLR(optimizer,
                                                max_lr=self.hparams.parameters_training["max_lr"],
                                                epochs=self.hparams.parameters_training["epochs"],
                                                steps_per_epoch=len(self.train_dataloader())),
                          "interval": "step"}

        return {"optimizer":optimizer, "lr_scheduler": scheduler_dict}

    def train_dataloader(self):
        if not self.hparams.parameters_data["balanced_train"]:
            return DataLoader(self.train_dataset, 
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=self.parameters_data["shuffle_train"]) 
        else:
            return DataLoader(self.train_dataset, 
                              sampler=BalancedBatchSampler(
                                  self.train_dataset,
                                  shuffle=self.parameters_data["shuffle_train"],
                                  oversample=self.parameters_data["oversample_train"]),
                              batch_size=self.batch_size,
                              num_workers=self.num_workers)

    def val_dataloader(self):
        if not self.hparams.parameters_data["balanced_val"]:
            return DataLoader(self.val_dataset,
                              batch_size=self.batch_size_val,
                              num_workers=self.num_workers,
                              shuffle=self.parameters_data["shuffle_dev"]) 
        else:
            return DataLoader(self.val_dataset, 
                              sampler=BalancedBatchSampler(
                                  self.val_dataset,
                                  shuffle=self.parameters_data["shuffle_dev"],
                                  oversample=self.parameters_data["oversample_dev"]),
                              batch_size=self.batch_size_val,
                              num_workers=self.num_workers)

    def test_dataloader(self):
        if not self.hparams.parameters_data["balanced_test"]:
            return DataLoader(self.eval_dataset,
                              batch_size=self.batch_size_test,
                              num_workers=self.num_workers,
                              shuffle=self.parameters_data["shuffle_eval"])
        else:
            return DataLoader(self.eval_dataset, 
                              sampler=BalancedBatchSampler(
                                  self.test_dataset,
                                  shuffle=self.parameters_data["shuffle_eval"],
                                  oversample=self.parameters_data["oversample_eval"]),
                              batch_size=self.batch_size_test,
                              num_workers=self.num_workers)

