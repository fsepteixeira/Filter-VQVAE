
# Filter VQ-VAE
This is the official PyTorch implementation of [Privacy-oriented manipulation of speaker representations](https://arxiv.org/pdf/2310.06652.pdf).
This code heavily relies on [SpeechBrain](https://speechbrain.github.io) [1], using among other things SpeechBrain's [pre-trained ECAPA-TDNN model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), as well as [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/).

### Setup
1. Download source code from GitHub
  ``` 
  git clone https://github.com/fsepteixeira/FilterVQ-VAE.git
  ```
2. Create [miniconda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
  ```
  conda create --name FilterVQVAE python==3.9.12
  ```  
3. Activate conda environment 
  ```
  conda activate FilterVQVAE 
  ```
5. Install requirements
  ```
  pip install -r requirements.txt
  ```

### Paper experiments
To run the experiments described in the paper you need to go through the following instructions. Note that for each attribute (sex or age) there are different configuration files in the hparams/ folder. 
The following examples all correspond to the sex attribute but also apply to age by simply replacing the .yaml file.
For the paper's experiments you do not need to adjust the configuration files, just follow the commands below.
All experiments use cuda by default. If using cpu add "--device cpu" to any of the commands below.

For adjusts to these experiments and extra options, please check the contents of the .yaml files, or create a pull request!

1. Setup data
Before running any experiments you need to create .csv files for each partition {train, dev, eval}.
The csv files should have as a header:
```
spkid,sex,path,lengths
```
Where
- ```spkid``` corresponds to the unique speaker identifier (alpha-numerical)
- ```sex``` or  ```age``` corresponds to the attribute label {0, 1} in case of sex, integer value for age.
- ```path``` is the path to the data samples. Dataset classes are written for VoxCeleb which follows ```<any_path>/<spkid>/<utt_id>/<sample>.wav```.
- ```lengths``` corresponds to the length of each sample.

2. Extract speaker embeddings from your data
```
python main_extract.py hparams/run_x_vector_extraction.yaml \
                       --target_folder <path_to_save_embeddings>
                       --partition <partition_to_extract_embeddings_from>
                       --train_csv <path_to_your_train_csv_file.csv>
                       --dev_csv <path_to_your_dev_csv_file.csv>
                       --eval_csv <path_to_your_eval_csv_file.csv>
                       --data_folder <path_to_folder_with_audio_samples>
```

```data_folder``` replaces ```<any_path>``` with any desired path.
You need to run this command for the three partitions.

3. Train speaker id classifier using your training data (this will be used as a part of the training of the VQ-VAE). By default the model will be saved to ```lightning_logs/speaker_id_classifier/```. Use as many speakers and samples as possible, as a stronger speaker classifier will end-up providing better results in terms of speaker verification for the "filtered" samples.
```
python main_speaker_classifier.py hparams/train_speaker_classifier.yaml \
                                  --train True          # Whether to train the model \
                                  --evaluate True       # Whether to evaluate the trained model \
                                  --partition eval      # Which partition to evaluate on \
                                  --n_speakers <number_of_train_speakers>  # Number of speakers in the training set \
                                  --train_csv  <path_to_your_train_csv_file.csv> \
                                  --dev_csv    <path_to_your_dev_csv_file.csv> \
                                  --eval_csv   <path_to_your_eval_csv_file.csv> \
                                  --data_folder_train <path_to_folder_containing_train_speaker_embeddings> \
                                  --data_folder_dev   <path_to_folder_containing_dev_speaker_embeddings> \
                                  --data_folder_eval  <path_to_folder_containing_eval_speaker_embeddings>
```
To test a pre-trained model, use the following:

python main_speaker_classifier.py hparams/train_speaker_classifier.yaml \
                                  --train False         # Whether to train the model \
                                  --evaluate True       # Whether to evaluate the trained model \
                                  --partition eval      # Which partition to evaluate on \
                                  --load_from_checkpoint True \
                                  --checkpoint lightning_logs/speaker_id_classifier/<your_checkpoint>.ckpt \
                                  --n_speakers <number_of_train_speakers>  # Number of speakers in the training set \
                                  --train_csv  <path_to_your_train_csv_file.csv> \
                                  --dev_csv    <path_to_your_dev_csv_file.csv> \
                                  --eval_csv   <path_to_your_eval_csv_file.csv> \
                                  --data_folder_train <path_to_folder_containing_train_speaker_embeddings> \
                                  --data_folder_dev   <path_to_folder_containing_dev_speaker_embeddings> \
                                  --data_folder_eval  <path_to_folder_containing_eval_speaker_embeddings>

4. Train "external" sex/age classifier using the VQ-VAE training data (this will be used as a part of the training of the VQ-VAE). By default the model will be saved to ```lightning_logs/embedding_classifier_<attribute>/checkpoints/```.
```
python main_classifier.py hparams/train_sex_classifier.yaml \
                          --train True \
                          --evaluate True \
                          --partition train \ # Partition here corresponds to train because we want the logits model to be obtained from the training data
                          --save_mvn True \
                          --train_csv  <path_to_your_train_csv_file.csv> \
                          --dev_csv    <path_to_your_dev_csv_file.csv> \
                          --eval_csv   <path_to_your_eval_csv_file.csv> \
                          --data_folder_train <path_to_folder_containing_train_speaker_embeddings> \
                          --data_folder_dev   <path_to_folder_containing_dev_speaker_embeddings> \
                          --data_folder_eval  <path_to_folder_containing_eval_speaker_embeddings>
                
```

5. Train VQ-VAE for sex/age. By default the model will be saved to ```lightning_logs/vq_vae_adv_mi_<attribute>/checkpoints/```.
```
python main_filter_vqvae.py hparams/train_filter_vq_vae_sex.yaml
                            --train True \ 
                            --evaluate True \
                            --partition eval \
                            --mvn_path external/sex_logits_mvn.pth \
                            --speaker_id_classifier_checkpoint lightning_logs/speaker_id_classifier/checkpoint/<your_checkpoint.ckpt> \
                            --pre_trained_classifier_checkpoint lightning_logs/embedding_classifier_sex/checkpoint/<your_checkpoint.ckpt> \
                            --train_csv  <path_to_your_train_csv_file.csv> \
                            --dev_csv    <path_to_your_dev_csv_file.csv> \
                            --eval_csv   <path_to_your_eval_csv_file.csv> \
                            --data_folder_train <path_to_folder_containing_train_speaker_embeddings> \
                            --data_folder_dev   <path_to_folder_containing_dev_speaker_embeddings> \
                            --data_folder_eval  <path_to_folder_containing_eval_speaker_embeddings>
```

6. Obtain results for Automatic Speaker Verification Performance. Results for the original data, un-transformed will also appear as "_org".
```
python main_filter_vqvae.py hparams/verify_filter_vq_vae_sex.yaml \
                            --verify True \
                            --random True \
                            --data_folder <path_to_eval_speaker_embeddings> \
                            --data_csv /cfs/projects/funded/privadia/data/VoxCeleb/veri_test2.txt \
                            --load_from_checkpoint True \
                            --checkpoint lightning_logs/vq_vae_adv_mi_sex/checkpoint/<your_checkpoint.ckpt> \
                            --speaker_id_classifier_checkpoint lightning_logs/speaker_id_classifier/checkpoint/<your_checkpoint.ckpt> \
                            --pre_trained_classifier_checkpoint lightning_logs/embedding_classifier_sex/checkpoint/<your_checkpoint.ckpt> \
                            --mvn_path external/sex_logits_mvn.pth \
```

7. Train a new attribute classifier over the optionally "filtered" embeddings, and obtain results -- Corresponds to the "informed attacker" described in the paper
```
python main_classifier_w_filter.py hparams/train_test_sex_classifier_w_filter.yaml \
                                   --train True \
                                   --evaluate True \
                                   --partition eval \
                                   --random_mode mean # mean for results in the paper -- other options: {zeros, "true", "remb", "remb_same"}
                                   --filter_data_train True # True for informed attacker / False for ignorant attacker \
                                   --filter_data_valid True # If both this and the previous flag are set to false, the classifier will output results for the original data \
                                   --filter_vq_vae lightning_logs/vq_vae_adv_mi_sex/checkpoint/<your_checkpoint> \
                                   --mvn_path external/sex_logits_mvn.pth \
                                   --train_csv  <path_to_your_train_csv_file.csv> \
                                   --dev_csv    <path_to_your_dev_csv_file.csv> \
                                   --eval_csv   <path_to_your_eval_csv_file.csv> \
                                   --data_folder_train <path_to_folder_containing_train_speaker_embeddings> \
                                   --data_folder_dev   <path_to_folder_containing_dev_speaker_embeddings> \
                                   --data_folder_eval  <path_to_folder_containing_eval_speaker_embeddings>
```

To evaluate attribute manipulation performance, select ```--random_mode remb_same``` for added variability of labels and add ```--return_target_label True``` to the previous command.

To evaluate any classifier with other partitions, or other pre-trained models, or to re-start training from the previous epoch, add the following to any previous training command:
```
  --train False # if just testing
  --load_from_checkpoint True \
  --checkpoint lightning_logs/<classifier_folder>/<your_checkpoint>.ckpt
```

- Note that we are using [pytorch-lightning](https://lightning.ai/docs/pytorch/stable/), and that repeating experiments will create new checkpoints within the same checkpoint folder for each experiment. Monitor new checkpoints to ensure you are using the correct one in downstream experiments.
- This repository uses [Zebra](https://gitlab.eurecom.fr/nautsch/zebra/-/tree/master) to compute the Zebra metrics proposed in [2].

[1] [Ravanelli, Mirco, et al. "SpeechBrain: A general-purpose speech toolkit." arXiv preprint arXiv:2106.04624 (2021)](https://arxiv.org/abs/2106.04624).
[2] [Nautsch, A., Patino, J., Tomashenko, N., Yamagishi, J., Noé, P.-G., Bonastre, J.-F., Todisco, M., Evans, N. (2020) The Privacy ZEBRA: Zero Evidence Biometric Recognition Assessment. Proc. Interspeech 2020, 1698-1702, doi: 10.21437/Interspeech.2020-1815](https://www.isca-speech.org/archive/interspeech_2020/nautsch20_interspeech.html).
