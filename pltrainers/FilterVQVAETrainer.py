
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

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, recall, precision, f1_score, auroc, confusion_matrix, average_precision, \
                                    mean_squared_error, mean_absolute_error, pearson_corrcoef, concordance_corrcoef

from torch.utils.data         import Dataset, DataLoader

from speechbrain.utils.metric_stats  import ErrorRateStats, EER, minDCF
from speechbrain.utils.Accuracy      import AccuracyStats

from models  import EmbeddingClassifier, FilterVQVAE, AdversarialClassifier, SpeakerIDClassifier
from modules import DiffClusterMI, ReconstructionLoss
from utils import BalancedBatchSampler, HDF5VoxCelebDataset, HDF5VoxCelebVeriDataset

from speechbrain.dataio.encoder import CategoricalEncoder
from sklearn.feature_selection import mutual_info_regression

def age2class(y):
    return torch.LongTensor(pd.qcut(y.cpu().numpy(), 8,
           labels=torch.arange(0, 8, 1).numpy(), duplicates="drop").to_list())

class FilterVQVAETrainer(pl.LightningModule):

    def __init__(self, parameters_vqvae,     parameters_training,
                       parameters_embedding, parameters_data, 
                       parameters_loss,      parameters_adv_classifier={}):

        super(FilterVQVAETrainer, self).__init__()

        # Init external embedding classifier
        self.embedding_classifier = EmbeddingClassifier(parameters_embedding, load_checkpoint=True).eval()

        # Load MultivariateNormal logit generator
        self.mvn = torch.load(parameters_embedding["mvn_path"])

        # Init Filter VQ-VAE
        self.filter_vqvae = FilterVQVAE(parameters_vqvae)

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
        self.id_classifier = SpeakerIDClassifier.load_from_checkpoint(parameters_training["extractor"])
        self.label_encoder = torch.load(parameters_training["extractor"]["label_encoder_path"])
        self.n_spk_classes = self.id_classifier.weight.shape[0]

        #speaker_ids = pd.read_csv(parameters_data["train_csv"], dtype={"spkid": str})["spkid"]
        #self.label_encoder.update_from_iterable(speaker_ids)
        #self.label_encoder.add_unk()
        # Sanity Check
        #assert len(self.label_encoder) == self.n_spk_classes + 1

        self.parameters_data     = parameters_data
        self.parameters_training = parameters_training
        self.n_adv_classes = parameters_training["n_adv_classes"]

        # Weights for each of the terms in the loss
        self.alpha = parameters_training["alpha"] # Reconstruction loss
        self.beta  = parameters_training["beta"]  # Div loss
        self.gamma = parameters_training["gamma"] # AngularSoftmax loss

        # Init losses
        self.reconstruction_loss = parameters_loss["rec"]
        self.aam_loss            = parameters_loss["aam"]

        # Init MI loss
        self.mi_metric = parameters_loss["mi_metric"]
        if parameters_training["MI"]:
            self.mi_loss  = parameters_loss["mi"]
            self.delta    = parameters_training["delta"] # MI loss
            self.mi_start_epoch = parameters_training["mi_start"]

        # Init internal classifier
        if parameters_training["ADV"]:
            self.adversarial_classifier = AdversarialClassifier(parameters_adv_classifier)
            self.adversarial_loss       = parameters_loss["adv"]
            self.zeta                   = parameters_training["zeta"] # Adv loss

        self.regression  = parameters_training["regression"]

        self.random_mode = parameters_training["random_mode"]
        self.random_logits_mode = parameters_training["random_logits_mode"]
        if self.random_logits_mode == "int_unif":
            self.range_ = parameters_training["range"]

        self.test_mode = None
        if "verification" in parameters_data.keys():
            self.test_mode            = "verify"
            parameters                = parameters_data["verification"]
            self.save_verification_scores = parameters_data["verification"]["save_verification_scores"]
            self.save_score_path     = parameters_data["verification"]["save_score_path"] if self.save_verification_scores else ""
            self.verify_random        = parameters["random"]
            self.verification_dataset = HDF5VoxCelebVeriDataset(parameters["data_csv"],
                                                                parameters["data_folder"])
        elif "extraction" in parameters_data.keys():
            self.test_mode     = "extract"
            parameters         = parameters_data["extraction"]
            self.apply_random  = parameters["random"]
            self.save_path     = parameters["save_path"]
            self.return_trg_label = parameters["return_target_label"]
            self.override      = parameters["override"]

        else:
            self.test_mode = "evaluate"

        # Save hyper-parameters
        self.save_hyperparameters()

    def forward(self, X, dropout_p=None, embedding=None):
        return self.filter_vqvae(X, embedding, dropout_p)

    def generate_random_logits(self, n_samples):
        if self.random_logits_mode == "mvn":
            return self.mvn.rsample([n_samples])
        elif self.random_logits_mode == "int_unif":
            return (torch.rand([n_samples, 1])*(self.range_[0] - self.range_[1]) + self.range_[1]).long().float()
        else:
            raise NotImplementedError

    def classify_id(self, emb, ids):
        labels = self.label_encoder.encode_sequence_torch(ids)
        preds = self.id_classifier(emb)
        return preds, labels

    def training_step(self, batch, batch_idx):
        # Get data from batch
        X, y, ids = batch
        batch_size = X.shape[0]

        # Get embeddings through the pre-trained classifier
        with torch.no_grad():
            self.embedding_classifier.eval()
            true_logits = self.embedding_classifier(X)

        # Update dropout
        dropout_p = self.hparams.parameters_training["dropout_schedule"](batch_idx)

        # Run the VQ-VAE Filter - Get reconstructed signal and quantised vectors
        Y, e, q, r = self.filter_vqvae(X, true_logits, dropout_p, batch_idx)
        # Compute reconstruction loss
        rec_loss = self.reconstruction_loss(X, Y)

        # Compute diversity loss as in wav2vec 2.0
        div_loss = (r["num_vars"] - r["prob_perplexity"]) / r["num_vars"]

        # Compute AdditiveAngularMargin loss as in the training of the x-vector network
        spk_rec, y_spk_rec = self.classify_id(Y, ids)
        aam_loss = self.aam_loss(spk_rec, y_spk_rec.to(Y.device).unsqueeze(dim=1))

        # Total loss
        loss = self.alpha * rec_loss + self.beta * div_loss + self.gamma * aam_loss

        # Log everything
        self.log("train_rec_loss", rec_loss,             batch_size=batch_size)
        self.log("train_div_loss", div_loss,             batch_size=batch_size)
        self.log("train_code_ppl", r["code_perplexity"], batch_size=batch_size)
        self.log("train_aam_loss", aam_loss,             batch_size=batch_size)


        # For logging purposes only
        with torch.no_grad():
            #self.log("train_mi_metric", self.mi_metric(q, y), batch_size=batch_size)
            if spk_rec.ndim > 2:
                spk_rec = spk_rec.squeeze(dim=1)
            spk_acc = accuracy(spk_rec.argmax(dim=-1).detach().cpu(), y_spk_rec, "multiclass", num_classes=self.n_spk_classes)
            self.log("train_spk_acc", spk_acc, batch_size=batch_size)

        # Mutual Information loss
        if self.parameters_training["MI"]:
            if self.current_epoch >= self.mi_start_epoch:
                # Compute MI Loss
                #y = age2class(y).to(q.device)
                if self.parameters_training["mi"]["after_bottleneck"]:
                    mi_loss = self.mi_loss(q, y)
                else:
                    mi_loss = self.mi_loss(e, y)

                # Add to loss
                loss = loss + self.delta * mi_loss

                # Log
                self.log("train_mi_loss", mi_loss.item() if not type(mi_loss)==int else mi_loss, batch_size=batch_size)

        # Adversarial loss
        if self.parameters_training["ADV"]:
            # Run adversarial classifier
            if self.parameters_training["adversarial"]["after_bottleneck"]:
                y_adv = self.adversarial_classifier(q)
            else:
                y_adv = self.adversarial_classifier(e)

            # Compute adversarial loss
            if self.regression:
                adv_loss = self.adversarial_loss(y_adv.squeeze(dim=1), y.float())
            else:
                adv_loss = self.adversarial_loss(y_adv.squeeze(dim=1), y)

            # Compute adversarial accuracy
            with torch.no_grad():
                if not self.regression:
                    adv_acc = accuracy(y_adv.argmax(dim=-1).detach().cpu(), y.cpu(), "multiclass", num_classes=self.n_adv_classes)
                    self.log("train_adv_acc", adv_acc, batch_size=batch_size)
                else:
                    adv_mae = mean_absolute_error(y_adv.squeeze(dim=1).detach().cpu(), y.cpu())
                    self.log("train_adv_mae", adv_mae, batch_size=batch_size)

            # Add to loss
            loss = loss + self.zeta * adv_loss

            # Log
            self.log("train_adv_loss", adv_loss.item(), batch_size=batch_size)

        self.log("train_loss", loss.item(), batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "val", random=False), \
               self.evaluate(batch, "val", random=True)

    def on_validation_epoch_start(self):
        if self.random_mode == "remb_same":
            self.spk_emb_dict = {}

    def on_validation_epoch_end(self):
        if self.random_mode == "remb_same":
            self.spk_emb_dict = {}

    def evaluate(self, batch, stage=None, random=False):

        # Get batch
        X, y, ids = batch
        batch_size = X.shape[0]

        with torch.no_grad():
            # Get baseline id predictions

            # Apply filter
            if random:
                if self.random_mode == "remb_diff":
                    logits = self.generate_random_logits(X.shape[0]).to(X.device)
                elif self.random_mode == "mean":
                    logits = self.mvn.loc.unsqueeze(dim=0).repeat(X.shape[0], 1).to(X.device)
                elif self.random_mode == "zeros":
                    logits = torch.zeros(X.shape[0], self.n_adv_classes).to(X.device)
                elif self.random_mode == "remb_same":
                    logits = self.get_spk_embs(ids).to(X.device)
                else:
                    raise NotImplementedError("Random mode not implemented.")

                if not self.regression:
                    y_trg  = logits.argmax(dim=1).cpu()
                else:
                    y_trg = logits.cpu()

                Y, e, q, r = self.filter_vqvae(X, embedding=logits)
            else:
                Y, e, q, r = self.filter_vqvae(X, embedding=self.embedding_classifier(X))

            # Get predictions for the original and reconstructed signals
            if not random:
                out_org = self.embedding_classifier(X)

                if not self.regression:
                    predictions_org  = torch.argmax(out_org, dim=1).cpu()
                else:
                    predictions_org = out_org.cpu()

            out_rec = self.embedding_classifier(Y)

            if not self.regression:
                predictions_rec = torch.argmax(out_rec, dim=1).cpu()
            else:
                predictions_rec = out_rec.cpu()

            # Get losses
            rec_loss     = self.reconstruction_loss(X,  Y)
            rec_loss_cos = 1 - torch.nn.functional.cosine_similarity(X,  Y)

            # Compute total loss
            loss = self.alpha * rec_loss

            # Compute predictions
            if not random:
                if not self.regression:
                    org_acc = accuracy(predictions_org, y.cpu(), "multiclass", num_classes=self.n_adv_classes)
                else:
                    org_mae = mean_absolute_error(predictions_org.squeeze(dim=1), y.cpu())

            if not self.regression:
                rec_acc = accuracy(predictions_rec,  y.cpu(), "multiclass", num_classes=self.n_adv_classes)
            else:
                rec_mae = mean_absolute_error(predictions_rec.squeeze(dim=1), y.cpu())

            if random:
                if not self.regression:
                    targ_acc = accuracy(predictions_rec, y_trg, "multiclass", num_classes=self.n_adv_classes)
                    sim_acc  = accuracy(y_trg,           y.cpu(), "multiclass", num_classes=self.n_adv_classes)
                else:
                    targ_mae = mean_absolute_error(predictions_rec,      y_trg) #  num_classes=self.n_adv_classes)
                    sim_mae  = mean_absolute_error(y_trg.squeeze(dim=1), y.cpu())#, num_classes=self.n_adv_classes)

            # Log metrics
            if not random:
                if not self.regression:
                    self.log(f"{stage}_org_acc", org_acc, prog_bar=False, batch_size=batch_size)
                else:
                    self.log(f"{stage}_org_mae", org_mae, prog_bar=False, batch_size=batch_size)

            self.log(f"{stage}_rec_loss_r_{random}",     rec_loss,     prog_bar=False, batch_size=batch_size)
            self.log(f"{stage}_rec_cos_loss_r_{random}", rec_loss_cos, prog_bar=False, batch_size=batch_size)

            if not self.regression:
                self.log(f"{stage}_rec_acc_r_{random}", rec_acc, prog_bar=True, batch_size=batch_size)
            else:
                self.log(f"{stage}_rec_mae_r_{random}", rec_mae, prog_bar=True, batch_size=batch_size)

            if random:
                if not self.regression:
                    self.log(f"{stage}_targ_acc", targ_acc, prog_bar=False, batch_size=batch_size)
                    self.log(f"{stage}_sim_acc",  sim_acc,  prog_bar=False, batch_size=batch_size)
                else:
                    self.log(f"{stage}_targ_mae", targ_mae, prog_bar=False, batch_size=batch_size)
                    self.log(f"{stage}_sim_mae",  sim_mae,  prog_bar=False, batch_size=batch_size)


            if self.parameters_training["ADV"]:
                # Compute adversarial predictions
                if self.parameters_training["adversarial"]["after_bottleneck"]:
                    y_adv = self.adversarial_classifier(q)
                else:
                    y_adv = self.adversarial_classifier(e)

                # Compute adversarial loss
                adv_loss = self.adversarial_loss(y_adv.squeeze(dim=1), y)

                # Add to total loss
                loss = loss + self.zeta * adv_loss

                if not self.regression:
                    adv_acc = accuracy(y_adv.argmax(dim=-1).cpu(), y.cpu(), "multiclass", num_classes=self.n_adv_classes)
                else:
                    adv_mae = mean_absolute_error(y_adv.cpu().squeeze(dim=1), y.cpu())

                # Log
                if not random:
                    if not self.regression:
                        self.log(f"{stage}_adv_acc", adv_acc, 
                            prog_bar=True, batch_size=batch_size)
                    else:
                        self.log(f"{stage}_adv_mae", adv_mae, 
                            prog_bar=True, batch_size=batch_size)

                    self.log(f"{stage}_adv_loss", adv_loss, 
                        prog_bar=True, batch_size=batch_size)

        self.log(f"{stage}_loss_r_{random}", loss, prog_bar=False, batch_size=batch_size)

        return out_rec, y.cpu()

    def get_spk_embs(self, ids):
        if hasattr(self, 'spk_emb_dict'):
            embs = []
            for id_ in ids:
                if len(id_.split("-")) > 1:
                    id_ = id_.split("-")[0]
                if id_ in self.spk_emb_dict.keys():
                    embs.append(self.spk_emb_dict[id_])
                else:
                    emb = self.generate_random_logits(1)
                    embs.append(emb)
                    self.spk_emb_dict[id_] = emb
        else:
            raise ValueError("spk_emb_dict has not been initialized, something went wrong...")

        return torch.cat(embs, dim=0)

    def on_test_epoch_start(self):
        if self.test_mode == "verify":
            self.scores = {0: [], 1: []}
            self.scores_org = {0: [], 1: []}
            self.verification_info = []

        if self.random_mode == "remb_same":
            self.spk_emb_dict = {}

    def test_step(self, batch, batch_idx):
        if self.test_mode == "verify":
            return self.verify(batch)
        elif self.test_mode == "extract":
            return self.extract(batch)
        elif self.test_mode == "evaluate":
            _, _ = self.evaluate(batch, "test", random=False)
            _, _ = self.evaluate(batch, "test", random=True)
        else:
            raise NotImplementedError("Mode " + str(self.test_mode) + " is not implemented. Choose one of {verify, extract, evaluate}")

    def on_test_epoch_end(self, outputs=None):
        if self.test_mode == "verify":
            eer,     eer_th     = EER(torch.tensor(self.scores_org[1]),    torch.tensor(self.scores_org[0]))
            min_dcf, min_dcf_th = minDCF(torch.tensor(self.scores_org[1]), torch.tensor(self.scores_org[0]))
            self.log("verification_EER_org",       eer)
            self.log("verification_EER_th_org",    eer_th)
            self.log("verification_minDCF_org",    min_dcf)
            self.log("verification_minDCF_th_org", min_dcf_th)
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
                
        if self.random_mode == "remb_same":
            self.spk_emb_dict = {}


    def verify(self, batch):
        # Get batch
        enroll, test, label, id_0, id_1 = batch

        with torch.no_grad():
            # Compute original score, for sanity check
            score_org = F.cosine_similarity(enroll, test)

            # Apply filter
            if self.verify_random:
                if self.random_mode == "zeros":
                    Y_enroll, _, _, _ = self.filter_vqvae(enroll, embedding=torch.zeros(1, self.n_adv_classes).to(enroll.device))
                    Y_test,   _, _, _ = self.filter_vqvae(test,   embedding=torch.zeros(1, self.n_adv_classes).to(test.device))

                elif self.random_mode == "mean":
                    Y_enroll, _, _, _ = self.filter_vqvae(enroll, embedding=self.mvn.loc.unsqueeze(dim=0).to(enroll.device))
                    Y_test,   _, _, _ = self.filter_vqvae(test,   embedding=self.mvn.loc.unsqueeze(dim=0).to(test.device))

                elif self.random_mode == "remb_same":
                    emb_enroll = self.generate_random_logits(1).to(enroll.device)
                    if label == 0:
                        emb_test = self.generate_random_logits(1).to(enroll.device)
                    else:
                        emb_test = emb_enroll
                    Y_enroll, _, _, _ = self.filter_vqvae(enroll, embedding=emb_enroll)
                    Y_test,   _, _, _ = self.filter_vqvae(test,   embedding=emb_test)

                elif self.random_mode == "remb_diff":
                    emb_enroll = self.generate_random_logits(1).to(enroll.device)
                    emb_test   = self.generate_random_logits(1).to(test.device)
                    Y_enroll, _, _, _ = self.filter_vqvae(enroll, embedding=emb_enroll)
                    Y_test,   _, _, _ = self.filter_vqvae(test,   embedding=emb_test)

                else:
                    raise NotImplementedError("Verify random mode " + self.random_mode + " is not implemented. Choose one of {zeros, mean, remb_same, remb_diff}")
            else:
                Y_enroll, _, _, _ = self.filter_vqvae(enroll, embedding=self.embedding_classifier(enroll))
                Y_test,   _, _, _ = self.filter_vqvae(test,   embedding=self.embedding_classifier(test))
            
            # Compute score
            score = F.cosine_similarity(Y_enroll, Y_test)

        # Save score
        self.scores_org[label.item()].append(score_org.item())
        self.scores[label.item()].append(score.item())

        self.verification_info.append({"score_org": score_org.item(), "score": score.item(),
                                       "label": label.item(), "key_0": id_0, "key_1": id_1})

    def apply_filter(self, batch):
        # Get batch
        X, y, ids = batch
        with torch.no_grad():
            # Apply filter
            if self.apply_random:
                if self.random_mode == "zeros":
                    Y, _, _, _ = self.filter_vqvae(X, embedding=torch.zeros(X.shape[0], self.n_adv_classes).to(X.device))
                elif self.random_mode == "mean":
                    Y, _, _, _ = self.filter_vqvae(X, embedding=self.mvn.loc.unsqueeze(dim=0).repeat(X.shape[0], 1).to(X.device))
                elif self.random_mode == "remb_diff":
                    y_trg = self.generate_random_logits(X.shape[0])
                    Y, _, _, _ = self.filter_vqvae(X, embedding=y_trg.to(X.device))
                    if not self.regression:
                        y_trg = y_trg.argmax(dim=1)
                elif self.random_mode == "remb_same":
                    Y, _, _, _ = self.filter_vqvae(X, embedding=self.get_spk_embs(ids).to(X.device))
                    y_trg = self.get_spk_embs(ids)
                    if not self.regression:
                        y_trg = y_trg.argmax(dim=1)
                else:
                    raise NotImplemented("Apply random mode " + self.random_mode + " is not implemented. Choose one of {zeros, mean, remb}")
            else:
                Y, _, _, _ = self.filter_vqvae(X, embedding=self.embedding_classifier(X))

        if self.return_trg_label and (self.random_mode == "remb_same" or self.random_mode == "remb_diff"):
            return Y, torch.cat([y.cpu().reshape(-1, 1), y_trg.cpu().reshape(-1, 1)], dim=1), ids
        else:
            return Y, y, ids

    def save_to_file(self, X, y, ids):
        # Save to file
        for i, (x, y_, id_) in enumerate(zip(X, y, ids)):

            # Init file
            spk_id = id_.split("-")[0]
            new_filename = "-".join(id_.split("-")[1:-1]) + ".h5"
            path = os.path.join(self.save_path, spk_id)

            if not os.path.exists(path):
                os.makedirs(path)

            new_file = os.path.join(path, new_filename)
            if not os.path.exists(new_file):
                h5py.File(new_file, "w").close()

            with h5py.File(new_file, "a") as hf:
                try: 
                    ds = hf.create_dataset(id_, data=x)
                    if self.return_trg_label and (self.random_mode == "remb_diff" or self.random_mode == "remb_same"):
                        ds.attrs["label"] = y_[0].cpu()
                        ds.attrs["label_trg"] = y_[1].cpu()
                    else:
                        ds.attrs["label"] = y_.cpu()
                except:
                    warnings.warn("Dataset already exists. Moving on...")

    def extract(self, batch):
        # Apply filter
        X, y, ids = self.apply_filter(batch)
        
        # Save sample
        self.save_to_file(X.cpu(), y, ids)

    def configure_optimizers(self):
        trainable_parameters = list(self.filter_vqvae.parameters())

        if self.parameters_training["ADV"]:
            trainable_parameters = trainable_parameters + list(self.adversarial_classifier.parameters())
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

    def verification_dataloader(self):
        return DataLoader(self.verification_dataset,
                          batch_size=1, num_workers=1)

