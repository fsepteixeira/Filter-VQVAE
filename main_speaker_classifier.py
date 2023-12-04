import sys, os, torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from pytorch_lightning                    import Trainer, seed_everything
from pytorch_lightning.callbacks          import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers            import TensorBoardLogger
from pytorch_lightning.strategies.ddp     import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from pltrainers import SpeakerIDTrainer

def main():

    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyper-parameters
    with open(hparams_file, "r") as hpf:
        hparams = load_hyperpyyaml(hpf, overrides)

    # Init Model
    model = SpeakerIDTrainer(hparams["speaker_classifier"],
                             hparams["training"],
                             hparams["data"],
                             hparams["loss"])

    # Init Trainer
    checkpoint_filename = hparams["checkpoint_name"]
    full_log_path  = os.path.join(hparams["lightning_path"], hparams["log_path"])

    train_checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(full_log_path, checkpoint_filename),
                                                filename=hparams["checkpoint_name"],
                                                every_n_epochs=hparams["every_n_epochs"])

    checkpoint_path = os.path.join(hparams["lightning_path"], hparams["log_path"] + "_checkpoints/")

    trainer = Trainer(
        max_epochs=hparams["epochs"],
        gpus=hparams["n_gpus"],
        accelerator=hparams["accelerator"],
        logger=TensorBoardLogger(hparams["lightning_path"],
                                 name=hparams["log_path"]),
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=1),
                   train_checkpoint_callback,
                   ],
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
        trainer.fit(model)
        if hparams["save_label_encoder"]:
            torch.save(model.label_encoder, hparams["save_label_encoder_path"])

    # Test model against chosen partition
    if hparams["evaluate"]:
        if hparams["partition"] == "train":
            dataloader = model.train_dataloader()
        elif hparams["partition"] == "dev":
            dataloader = model.val_dataloader()
        elif hparams["partition"] == "eval":
            dataloader = model.test_dataloader()
        else:
            raise NotImplementedError("No such partition: " + hparams["partition"] + ". Choose one from {train, dev, eval}")
        trainer.test(model, 
                     dataloaders=dataloader,
                     ckpt_path=hparams["checkpoint"] if "checkpoint" in hparams.keys() else None)

    return 0

if __name__ == '__main__':
    main()

