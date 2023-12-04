
import argparse, sys, os
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml
from pytorch_lightning                    import Trainer, seed_everything
from pytorch_lightning.loggers            import TensorBoardLogger

from pltrainers import VerificationTrainer


def main():

    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyper-parameters
    with open(hparams_file, "r") as hpf:
        hparams = load_hyperpyyaml(hpf, overrides)

    # Init Model
    model = VerificationTrainer(hparams["data"])

    trainer = Trainer(
        gpus=hparams["n_gpus"],
        accelerator=hparams["accelerator"],
        logger=TensorBoardLogger(hparams["lightning_folder"],
                name=hparams["log_path"]))

    trainer.test(model, dataloaders=model.verification_dataloader())

    return 0

if __name__ == '__main__':
    main()

