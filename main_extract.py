import torch
import argparse, h5py, os, tqdm, sys
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml
from utils       import create_dataloaders_voxceleb

from speechbrain.pretrained.interfaces import EncoderClassifier

def save_hdf5(ids, data, labels, target_folder):
    for i, (X, y, k) in enumerate(zip(data, labels, ids)):
        id_ = k.split("-")[0]
        new_filename = "-".join(k.split("-")[1:-1]) + ".h5"

        path = os.path.join(target_folder, id_)
        if not os.path.exists(path):
            os.makedirs(path)
        
        new_file = os.path.join(path, new_filename)
        if not os.path.exists(new_file):
            h5py.File(new_file, "w").close()

        with h5py.File(new_file, "a") as hf:
            ds = hf.create_dataset(k, data=X)
            ds.attrs["label"] = y

def extract(model, dataloader, target_folder, hparams, filter_vq=None):
    for batch in tqdm.tqdm(dataloader):
        X, ids, y, l = batch
        X = X.to(hparams["device"])
        with torch.no_grad():
            Y = model.encode_batch(X, normalize=hparams["normalize"])
        save_hdf5(ids, Y.cpu(), y, target_folder)

def main():

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyper-parameters
    with open(hparams_file, "r") as hpf:
        hparams = load_hyperpyyaml(hpf, overrides)

    # Init DataLoaders
    dataloaders = create_dataloaders_voxceleb(hparams["data"])

    # Create save dir
    if not os.path.exists(hparams["target_folder"]):
        os.makedirs(hparams["target_folder"])

    # Init Model
    model = EncoderClassifier.from_hparams(source=hparams["extractor"]["source"],
                                           savedir=hparams["extractor"]["savedir"],
                                           run_opts={"device": hparams["device"]})

    # Extract embeddings
    extract(model, dataloaders[hparams["partition"]], hparams["target_folder"], hparams["extractor"])

    return 0

if __name__ == '__main__':
    main()
