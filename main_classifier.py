
import argparse, os, sys
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml

from pytorch_lightning                    import Trainer, seed_everything
from pytorch_lightning.callbacks          import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers            import TensorBoardLogger
from pytorch_lightning.strategies.ddp     import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from pltrainers import ClassifierTrainer
from utils      import create_dataloaders_classifier


def main():

    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyper-parameters
    with open(hparams_file, "r") as hpf:
        hparams = load_hyperpyyaml(hpf, overrides)

    # Init DataLoaders
    dataloaders = create_dataloaders_classifier(hparams["data"])

    # Init Trainer
    if hparams["train"] or hparams["evaluate"]:
        # Init Model
        model = ClassifierTrainer(hparams["classifier"], 
                                  hparams["training"],
                                  hparams["loss"],
                                  hparams["regression"],
                                  hparams["n_classes"],
                                  steps=len(dataloaders['train']) if 'train' in dataloaders.keys() else 0)

        # N-step checkpoint
        checkpoint_filename = hparams["checkpoint_name"]
        full_log_path  = os.path.join(hparams["lightning_path"], hparams["log_path"])

        train_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(full_log_path, checkpoint_filename),
                                                    filename=hparams["checkpoint_name"],
                                                    every_n_epochs=hparams["every_n_epochs"])

        checkpoint_path = os.path.join(hparams["lightning_path"], hparams["log_path"] + "_checkpoints/")

        if hparams["load_from_checkpoint"]:
            assert("checkpoint" in hparams.keys(), "Load checkpoint is set to true but no checkpoint is defined in the config file")

        if hparams["load_from_checkpoint"] and hparams["checkpoint"] == None:
            print("Warning: Loading from checkpoint, but checkpoint is set to None")

        trainer = Trainer(
            max_epochs=hparams["epochs"],
            devices=hparams["n_gpus"],
            accelerator=hparams["accelerator"],
            logger=TensorBoardLogger(hparams["lightning_path"],
                                     name=hparams["log_path"]),
            callbacks=[LearningRateMonitor(logging_interval="step"), 
                                           TQDMProgressBar(refresh_rate=1), 
                                           train_checkpoint_callback],
            strategy=DDPStrategy(find_unused_parameters=False),
            resume_from_checkpoint=hparams["checkpoint"] if hparams["load_from_checkpoint"] else None,
            detect_anomaly=False,
            track_grad_norm=2,
            gradient_clip_val=0.5,
            num_sanity_val_steps=2,
            log_every_n_steps=5,
            replace_sampler_ddp=False)

    # Train model
    if hparams["train"]:
        trainer.fit(model, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["dev"])

    # Test model against chosen partition
    if hparams["evaluate"]:
        assert (hparams["partition"] in ["train", "dev", "eval"], "No such partition:" + str(hparams["partition"]) + ". Choose one from {train, dev, eval}.")
        trainer.test(model, dataloaders=dataloaders[hparams["partition"]],
                     ckpt_path=hparams["checkpoint"] if hparams["load_from_checkpoint"] else None)

    return 0

if __name__ == '__main__':
    main()

