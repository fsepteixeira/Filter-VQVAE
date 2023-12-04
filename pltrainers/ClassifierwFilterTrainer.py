
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

from zebra import PriorLogOddsPlots, zebra_framework, cllr
from models import EmbeddingClassifier, FilterVQVAE

def create_load_vqvae(parameters):
    filter_ = FilterVQVAE(parameters)
    state_dict  = torch.load(parameters["checkpoint"], map_location="cpu")["state_dict"]
    relevant_state_dict = {}
    for k in state_dict.keys():
        if "filter_vqvae" in k:
            relevant_state_dict[".".join(k.split(".")[1:])] = state_dict[k]
    filter_.load_state_dict(relevant_state_dict)
    filter_.eval()
    return filter_

def create_load_classifier(parameters):
    classifier = EmbeddingClassifier(parameters)
    state_dict  = torch.load(parameters["checkpoint"], map_location="cpu")["state_dict"]
    relevant_state_dict = {}
    for k in state_dict.keys():
        if "classifier" in k:
            relevant_state_dict[".".join(k.split(".")[1:])] = state_dict[k]
    classifier.load_state_dict(relevant_state_dict)
    return classifier

class ClassifierwFilterTrainer(pl.LightningModule):

    def __init__(self, classifier, filter_, ptraining, loss_fn, regression=False, n_classes=2, steps=5000, range_=[0, 100]):

        super(ClassifierwFilterTrainer, self).__init__()
        if classifier["checkpoint"] != None:
            self.classifier    = create_load_classifier(classifier)
        else:
            self.classifier = EmbeddingClassifier(classifier)

        self.filter_data_train = ptraining["filter_data_train"]
        self.filter_data_valid = ptraining["filter_data_valid"]
        if self.filter_data_train or self.filter_data_valid:
            self.filter        = create_load_vqvae(filter_)

        self.ptraining     = ptraining
        self.steps         = steps
        self.loss_fn       = loss_fn
        self.regression    = regression
        if "evaluate_target" in ptraining.keys():
            self.evaluate_target = ptraining["evaluate_target"]
        else:
            self.evaluate_target = False
        if not self.regression:
            assert n_classes >= 2

        self.n_classes     = n_classes
        self.range_        = range_

        if self.filter_data_train or self.filter_data_valid:
            self.random_mode = ptraining["random_mode"]
            self.random_logits_mode = ptraining["random_logits_mode"]
            if self.random_logits_mode == "mvn":
                self.mvn = torch.load(ptraining["mvn_load_path"])
                
        self.save_hyperparameters(ignore=["loss_fn"])

    def forward(self, X, dropout_p=None):
        if self.filter_data_train or self.filter_data_valid:
            with torch.no_grad():
                X = self.apply_filter(X)
        return self.classifier(X, dropout_p)

    def apply_filter(self, batch):
        X, y, ids = batch
        
        if self.random_mode == "remb":
            logits = self.generate_random_logits(X.shape[0]).to(X.device)
        elif self.random_mode == "mean":
            logits = self.mvn.loc.unsqueeze(dim=0).repeat(X.shape[0], 1).to(X.device)
        elif self.random_mode == "zeros":
            logits = torch.zeros(X.shape[0], self.n_classes).to(X.device)
        elif self.random_mode == "remb_same":
            logits = self.get_spk_embs(ids).to(X.device)
        else:
            raise NotImplementedError("Random mode not implemented.")
        
        self.filter.eval()
        Y, _, _, _ = self.filter(X, embedding=logits)
        if self.random_mode == "remb" or self.random_mode == "remb_same":
            return Y, logits.argmax(dim=1)
        else:
            return Y

    def generate_random_logits(self, n_samples):
        if self.random_logits_mode == "mvn":
            return self.mvn.rsample([n_samples])
        elif self.random_logits_mode == "int_unif":
            return (torch.rand([n_samples, 1])*(self.range_[0] - self.range_[1]) + self.range_[1]).long().float()
        else:
            raise NotImplementedError

    def get_spk_embs(self, ids):
        embs = []
        for id_ in ids:
            if len(id_.split("-")) > 1:
                id_ = id_.split("-")[0]
            if self.training:
                if id_ in self.spk_emb_dict_train.keys():
                    embs.append(self.spk_emb_dict_train[id_])
                else:
                    emb = self.generate_random_logits(1)
                    embs.append(emb)
                    self.spk_emb_dict_train[id_] = emb
            else:
                if id_ in self.spk_emb_dict_val.keys():
                    embs.append(self.spk_emb_dict_val[id_])
                else:
                    emb = self.generate_random_logits(1)
                    embs.append(emb)
                    self.spk_emb_dict_val[id_] = emb

        return torch.cat(embs, dim=0)

    def on_train_epoch_start(self):
        if self.filter_data_train and self.random_mode == "remb_same":
            self.spk_emb_dict_train = {}

    def on_validation_epoch_start(self):
        if self.filter_data_valid and self.random_mode == "remb_same":
            self.spk_emb_dict_val = {}

    def on_test_epoch_start(self):
        if self.filter_data_valid and self.random_mode == "remb_same":
            self.spk_emb_dict_val = {}


    def training_step(self, batch, batch_idx):

        X, y, _ = batch

        dropout_p = self.hparams.ptraining["dropout_schedule"](batch_idx) # Not doing anything

        if self.filter_data_train:
            with torch.no_grad():
                if self.random_mode == "remb" or self.random_mode == "remb_same":
                    X, _ = self.apply_filter(batch)
                else:
                    X = self.apply_filter(batch)

        logits = self.classifier(X, dropout_p)
        
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
        if self.filter_data_valid and (self.random_mode == "remb" or self.random_mode == "remb_same"):
                loss, logits, labels, trg_labels, ids = self.evaluate(batch, "val")
                return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), 
                        "trg_labels": trg_labels.cpu(), "ids": ids}
        else:
            loss, logits, labels, ids = self.evaluate(batch, "val")
            return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), "ids": ids}

    def test_step(self, batch, batch_idx):
        if self.filter_data_valid and (self.random_mode == "remb" or self.random_mode == "remb_same"):
                loss, logits, labels, trg_labels, ids = self.evaluate(batch, "test")
                return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), 
                        "trg_labels": trg_labels.cpu(), "ids": ids}
        else:
            loss, logits, labels, ids = self.evaluate(batch, "test")
            return {"loss": loss.detach().item(), "logits": logits.detach().cpu(), "labels": labels.cpu(), "ids": ids}

    def evaluate(self, batch, stage=None):

        # Get batch
        X, y, ids = batch

        # Get predictions
        if self.filter_data_valid:
            if self.random_mode == "remb" or self.random_mode == "remb_same":
                F, trg_label = self.apply_filter(batch)
            else:
                F = self.apply_filter(batch)
            logits = self.classifier(F)
        else:
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

        if self.filter_data_valid:
            if self.random_mode == "remb" or self.random_mode == "remb_same":
                return loss, logits, y, trg_label, ids
            else:
                return loss, logits, y, ids
        else:
            return loss, logits, y, ids

    def validation_epoch_end(self, outputs):
        preds  = torch.cat([output["logits"] for output in outputs], dim=0)
        #labels = torch.cat([output["labels"] for output in outputs], dim=0)

        if self.filter_data_valid and self.evaluate_target and not self.regression:
                labels = torch.cat([output["trg_labels"] for output in outputs], dim=0)
                true_labels = torch.cat([output["labels"] for output in outputs], dim=0)
                true_uar = recall(true_labels.float(), labels, "multiclass", num_classes=self.n_classes)
                self.log(f"val_true_uar", true_uar)
        else:
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

        if self.filter_data_valid and self.random_mode == "remb_same":
            self.spk_emb_dict_val = {}

    def test_epoch_end(self, outputs):
        ids    = np.concatenate([list(output["ids"]) for output in outputs])
        ids_   = np.array([id_.split("-")[0] for id_ in ids])

        preds  = torch.cat([output["logits"] for output in outputs], dim=0)
        if self.filter_data_valid and self.evaluate_target and not self.regression:
            labels = torch.cat([output["trg_labels"] for output in outputs], dim=0)
            true_labels = torch.cat([output["labels"] for output in outputs], dim=0)
            true_uar = recall(true_labels.float(), labels, "multiclass", num_classes=self.n_classes)
            self.log(f"test_true_uar", true_uar)
        else:
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

            if "zebra" in self.ptraining.keys() and self.ptraining["zebra"]:
                scores = pd.DataFrame(data={0:ids, 1:ids, 2:torch.nn.functional.softmax(preds, dim=1)[:, 1].numpy()})
                keys   = pd.DataFrame(data={0:ids, 1:ids, 2:labels.numpy()})
                
                zebra_obj = PriorLogOddsPlots()
                scores = scores.pivot_table(index=0, columns=1, values=2)
                keys = keys.pivot_table(index=0, columns=1, values=2)

                classA_scores = scores.values[keys.values == True]
                classB_scores = scores.values[keys.values == False]
                
                zebra_obj.set_system(classA_scores, classB_scores)
                dece = zebra_obj.get_delta_ECE()
                max_abs_LLR = abs(np.hstack((zebra_obj.classA_llr_laplace, zebra_obj.classB_llr_laplace))).max() / np.log(10)
                actual_cllr = cllr(zebra_obj.classA_scores, zebra_obj.classB_scores)
                min_cllr = cllr(zebra_obj.classA_llr, zebra_obj.classB_llr)

                self.log(f"test_ece_sample", dece)
                self.log(f"test_indv_sample", max_abs_LLR)
                self.log(f"test_mincllr_sample", min_cllr)
                self.log(f"test_cllr_sample", actual_cllr)
            
                if not self.evaluate_target: 
                    scores = pd.DataFrame(data={0:ids_, 1:ids_, 2:torch.nn.functional.softmax(preds, dim=1)[:, 1].numpy()})
                    keys   = pd.DataFrame(data={0:ids_, 1:ids_, 2:labels.numpy()})

                    zebra_obj = PriorLogOddsPlots()
                    scores = scores.pivot_table(index=0, columns=1, values=2)
                    keys = keys.pivot_table(index=0, columns=1, values=2)

                    classA_scores = scores.values[keys.values == True]
                    classB_scores = scores.values[keys.values == False]
                    
                    zebra_obj.set_system(classA_scores, classB_scores)
                    dece = zebra_obj.get_delta_ECE()
                    max_abs_LLR = abs(np.hstack((zebra_obj.classA_llr_laplace, zebra_obj.classB_llr_laplace))).max() / np.log(10)
                    actual_cllr = cllr(zebra_obj.classA_scores, zebra_obj.classB_scores)
                    min_cllr = cllr(zebra_obj.classA_llr, zebra_obj.classB_llr)

                    self.log(f"test_ece_speaker", dece)
                    self.log(f"test_indv_speaker", max_abs_LLR)
                    self.log(f"test_mincllr_speaker", min_cllr)
                    self.log(f"test_cllr_speaker", actual_cllr)

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

        if "save_predictions" in self.ptraining.keys() and self.ptraining["save_predictions"]:
            if self.n_classes > 1:
                scores = pd.DataFrame(data={0:ids_, 1:ids_, 2:torch.nn.functional.softmax(preds, dim=1)[:, 1].numpy()})
            else:
                scores = pd.DataFrame(data={0:ids_, 1:ids_, 2:preds.numpy()})
            keys   = pd.DataFrame(data={0:ids_, 1:ids_, 2:labels.numpy()})

            os.makedirs("/".join(self.ptraining["save_predictions_params"]["score_save_path"].split("/")[:-1]), exist_ok=True)
            os.makedirs("/".join(self.ptraining["save_predictions_params"]["key_save_path"].split("/")[:-1]) , exist_ok=True)

            scores.to_csv(self.ptraining["save_predictions_params"]["score_save_path"], header=False, index=False, sep=" ")
            keys.to_csv(self.ptraining["save_predictions_params"]["key_save_path"], header=False, index=False, sep=" ")

        if self.filter_data_valid and self.random_mode == "remb_same":
            self.spk_emb_dict_val = {}

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

