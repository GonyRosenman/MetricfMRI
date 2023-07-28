# MetricfMRI (MIDL 2023)
This repo is the implementation for [**Pre-Training Transformers for Fingerprinting to Improve Stress Prediction in fMRI **](https://openreview.net/forum?id=W9qI8DwoUFF)



## Contents
- [Datasets](#datasets)
- [Train with our datasets](#training)
- [Cite](#cite)


## Datasets
We currently support the following datasets
* HCP - human connectome project S1200
  * Register at (https://db.humanconnectome.org/)
  * Download: WU-Minn HCP Data - 1200 Subjects -> Subjects with 3T MR session data -> Resting State fMRI 1 Preprocessed
  * Preprocess the data by configuring the folders and run 'data_preprocess_and_load/preprocessing.main()'
  * [Link to a single subject scan from HCP](https://drive.google.com/file/d/1zT9n1QL7GYTUAb8HOlqjEujzg40WAi6U/view?usp=share_link) (insert it in PathHandler -> exemplar)]
    
    
* ucla (Consortium for Neuropsychiatric Phenomics LA5c Study) 
  * Original version available at (https://openneuro.org/datasets/ds000030/versions/00016)
  * can be preprocessed according to similar hyperparameters available at the preprocess directory.

* stress-no stress (the CPS dataset mentioned in our paper, cannot be published) 
  * due to medical privacy of subjects - cannot be published.

## Training
* For the full pipeline described in the paper (that includes -
  1. volume reconstruction phase prediction run 'python main.py --fine_tune_task binary_classification'
  2. fingerprinting phase (subject triplet)
  3. fine-tuning on stress prediction.
  - run main.py


## Tensorboard support
All metrics are being logged automatically and stored in
```
MetricfMRI/runs
```
Run `tesnroboard --logdir=<path>` to see the the logs.


## Citing & Authors
If you find this repository helpful, feel free to cite our publication -

@inproceedings{
rosenman2023pretraining,
title={Pre-Training Transformers for Fingerprinting to Improve Stress Prediction in f{MRI}},
author={Gony Rosenman and Itzik Malkiel and Ayam Greental and Talma Hendler and Lior Wolf},
booktitle={Medical Imaging with Deep Learning},
year={2023},
url={https://openreview.net/forum?id=W9qI8DwoUFF}
}

Contact: [Gony Rosenman](mailto:gonyrosenman@mail.tau.ac.il), [Itzik Malkiel](mailto:itzik.malkiel@microsoft.com).
