
import os
import h5py
import tqdm
import numpy as np

import torch
import torch.optim
import torch.nn.functional as F
import pytorch_lightning as pl

import pandas as pd

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, recall, precision, f1_score, auroc, confusion_matrix, average_precision, \
                                    mean_squared_error, mean_absolute_error, pearson_corrcoef, concordance_corrcoef

from sklearn.feature_selection import mutual_info_classif

from models import EmbeddingClassifier

class ClassifierTrainer(pl.LightningModule):

    def __init__(self, classifier, ptraining, loss_fn, regression=False, n_classes=2, steps=5000, range_=[0, 100]):

        super(ClassifierTrainer, self).__init__()
        self.classifier    = EmbeddingClassifier(classifier)
        self.ptraining     = ptraining
        self.steps         = steps
        self.loss_fn       = loss_fn
        self.regression    = regression

        if not self.regression:
            assert n_classes >= 2
        self.n_classes     = n_classes
        self.range_        = range_

        self.save_hyperparameters()

    def forward(self, X, dropout_p=None):
        return self.classifier(X, dropout_p)

    def training_step(self, batch, batch_idx):

        X, y, _ = batch

        dropout_p = self.hparams.ptraining["dropout_schedule"](batch_idx) # Currently not in use
        logits    = self.classifier(X, dropout_p)
        
        if y.ndim > 1:
            y = y.squeeze()

        if y.is_floating_point:
            y = y.long()
        
        if type(self.loss_fn) == type(torch.nn.MSELoss()):
            y = y.float()

        loss = self.loss_fn(logits.squeeze(), y)
        self.log("train_loss", loss.detach().item(), batch_size=len(y))

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels, ids = self.evaluate(batch, "val")
        return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), "ids": ids}

    def test_step(self, batch, batch_idx):
        loss, logits, labels, ids = self.evaluate(batch, "test")
        return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), "ids": ids}

    def evaluate(self, batch, stage=None):

        # Get batch
        X, y, ids = batch

        # Get predictions
        logits = self.classifier(X)

        # Get metrics
        if y.ndim > 1:
            y = y.squeeze()

        if y.is_floating_point:
            y = y.long()
        elif isinstance(self.loss_fn, torch.nn.MSELoss):
            y = y.float()
        
        loss = self.loss_fn(logits.squeeze(), y)

        # Log metrics
        if stage != "test":
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=len(y))

        return loss, logits, y, ids

    def validation_epoch_end(self, outputs):
        preds  = torch.cat([output["logits"] for output in outputs], dim=0)
        labels = torch.cat([output["labels"] for output in outputs], dim=0)

        if not self.regression:
            acc = accuracy(preds,  labels, task="multiclass", num_classes=self.n_classes)
            uar = recall(preds,    labels, task="multiclass", average="macro", num_classes=self.n_classes)
            prc = precision(preds, labels, task="multiclass", average="macro", num_classes=self.n_classes)
            f1  = f1_score(preds,  labels, task="multiclass", average="macro", num_classes=self.n_classes)

            auroc_ = auroc(preds,            labels, task="multiclass", average="macro", num_classes=self.n_classes)
            cfm    = confusion_matrix(preds, labels, task="multiclass", num_classes=self.n_classes)

            print("Recall per class:",    recall(preds,    labels, task="multiclass", average=None, num_classes=self.n_classes))
            print("Precision per class:", precision(preds, labels, task="multiclass", average=None, num_classes=self.n_classes))
            print("F1 per class:",        f1_score(preds,  labels, task="multiclass", average=None, num_classes=self.n_classes))

            self.log(f"valid_acc", acc)
            self.log(f"valid_auc", auroc_)
            self.log(f"valid_f1",  f1)
            self.log(f"valid_uap", prc)
            self.log(f"valid_uar", uar)

        else:
            if self.n_classes > 1:
                k = torch.linspace(start=self.range_[0], end=self.range_[1], steps=self.n_classes, 
                                   dtype=torch.int, requires_grad=False).reshape(-1, 1)
                assert preds.shape[1] == self.n_classes
                preds_ = torch.matmul(torch.nn.functional.softmax(preds, dim=1), k.float())
            else:
                preds_ = preds.reshape(-1, 1)

            rmse = mean_squared_error(preds_.squeeze(dim=1), labels, squared=False)
            mae  = mean_absolute_error(preds_.squeeze(dim=1), labels)

            print("RMSE:", rmse)
            print("MAE:", mae)

            self.log(f"valid_rmse", rmse)
            self.log(f"valid_mae", mae)


    def test_epoch_end(self, outputs):
        ids    = np.concatenate([list(output["ids"]) for output in outputs])
        ids_   = np.array([id_.split("-")[0] for id_ in ids])

        if ids_[0] == "all":    # Hack for TIMIT
            ids_   = np.array([id_.split("-")[1] for id_ in ids])

        preds  = torch.cat([output["logits"] for output in outputs], dim=0)
        labels = torch.cat([output["labels"] for output in outputs], dim=0)

        if not self.regression:
            acc = accuracy(preds,  labels, "multiclass", num_classes=self.n_classes)
            uar = recall(preds,    labels, task="multiclass", average="macro", num_classes=self.n_classes)
            prc = precision(preds, labels, task="multiclass", average="macro", num_classes=self.n_classes)
            f1  = f1_score(preds,  labels, task="multiclass", average="macro", num_classes=self.n_classes)

            auroc_ = auroc(preds,             labels, task="multiclass", average="macro", num_classes=self.n_classes)
            auprc_ = average_precision(preds, labels, task="multiclass", average="macro", num_classes=self.n_classes)
            cfm    = confusion_matrix(preds,  labels, task="multiclass", num_classes=self.n_classes)

            print("Recall per class:",    recall(preds,    labels, task="multiclass", average=None, num_classes=self.n_classes))
            print("Precision per class:", precision(preds, labels, task="multiclass", average=None, num_classes=self.n_classes))
            print("F1 per class:",        f1_score(preds,  labels, task="multiclass", average=None, num_classes=self.n_classes))

            self.log(f"test_acc",   acc)
            self.log(f"test_auc",   auroc_)
            self.log(f"test_auprc", auprc_)
            self.log(f"test_f1",    f1)
            self.log(f"test_uap",   prc)
            self.log(f"test_uar",   uar)

        else:
            if self.n_classes > 1:
                k = torch.linspace(start=self.range_[0], end=self.range_[1], steps=self.n_classes, 
                                   dtype=torch.int, requires_grad=False).reshape(-1, 1)
                assert preds.shape[1] == self.n_classes
                preds_ = torch.matmul(torch.nn.functional.softmax(preds, dim=1), k.float())
            else:
                preds_ = preds.reshape(-1, 1)
            rmse = mean_squared_error(preds_.squeeze(dim=1), labels, squared=False)
            mae  = mean_absolute_error(preds_.squeeze(dim=1), labels)
            pcc = pearson_corrcoef(preds_.squeeze(dim=1), labels.float())
            ccc = concordance_corrcoef(preds_.squeeze(dim=1), labels.float())

            print("RMSE:", rmse)
            print("MAE:", mae)
            print("PCC:", pcc)
            self.log(f"test_rmse", rmse)
            self.log(f"test_mae", mae)
            self.log(f"test_pcc", pcc)
            self.log(f"test_ccc", ccc)


        if self.ptraining["save_mvn"]:
            if self.n_classes > 1:
                self.mvn = MultivariateNormal(preds.mean(dim=0), torch.cov(preds.T))
            else:
                self.mvn = Normal(preds.mean(dim=0), torch.std(preds.T))
            torch.save(self.mvn, self.ptraining["mvn_save_path"])

        if "save_predictions" in self.ptraining.keys():
            scores = pd.DataFrame(data={0:ids_, 1:ids_, 2:torch.nn.functional.softmax(preds, dim=1)[:, 1].numpy()})
            keys   = pd.DataFrame(data={0:ids_, 1:ids_, 2:labels.numpy()})

            os.makedirs("/".join(self.ptraining["save_predictions"]["score_save_path"].split("/")[:-1]), exist_ok=True)
            os.makedirs("/".join(self.ptraining["save_predictions"]["key_save_path"].split("/")[:-1]) , exist_ok=True)

            scores.to_csv(self.ptraining["save_predictions"]["score_save_path"], header=False, index=False, sep=" ")
            keys.to_csv(self.ptraining["save_predictions"]["key_save_path"], header=False, index=False, sep=" ")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.ptraining["lr"],
                                     weight_decay=self.hparams.ptraining["weight_decay"])
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.ptraining["max_lr"],
                epochs=self.hparams.ptraining["epochs"],
                steps_per_epoch=int(self.steps/self.hparams.ptraining["n_gpus"]),
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

