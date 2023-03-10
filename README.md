# MetricfMRI
This repo is the implementation for [**MetricfMRI**](insert arxiv link here). 



## Contents
- [Datasets](#datasets)
- [Train with our datasets](#training)
- [Explainability]
- [Cite](#cite)


## Datasets
We currently support the following datasets
* HCP - human connectome project S1200
  * Register at (https://db.humanconnectome.org/)
  * Download: WU-Minn HCP Data - 1200 Subjects -> Subjects with 3T MR session data -> Resting State fMRI 1 Preprocessed
  * Preprocess the data by configuring the folders and run 'data_preprocess_and_load/preprocessing.main()'
    
* ucla (Consortium for Neuropsychiatric Phenomics LA5c Study) 
  * Original version available at (https://openneuro.org/datasets/ds000030/versions/00016)
  * can be preprocessed according to similar hyperparameters available at the preprocess directory.

* ayam (the CPS dataset mentioned in our paper, cannot be published) 
  * Original version available at (https://openneuro.org/datasets/ds000030/versions/00016)
  * Data after proprocessing will be added soon, for now can download original and preprocess indiependently.

## Training
* For the full pipeline described in the paper (that includes -
  1. volume reconstruction phase prediction run 'python main.py --dataset_name S1200 --fine_tune_task binary_classification'
  2. fingerprinting phase (subject triplet)
  3. fine-tuning on stress prediction.
  - run main.py

## Explainability
in the explainability directory there are example scripts to run in order to produce explainability figures that are based on the saliency map method described in the paper. notice that in order to produce explainability figures one should insert a path to a trained model. 

## Tensorboard support
All metrics are being logged automatically and stored in
```
MetricfMRI/runs
```
Run `tesnroboard --logdir=<path>` to see the the logs.


## Citing & Authors
If you find this repository helpful, feel free to cite our publication -
insert Cite here

Contact: [Gony Rosenman](mailto:gonyrosenman@mail.tau.ac.il), [Itzik Malkiel](mailto:itzik.malkiel@microsoft.com).
