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


@InProceedings{pmlr-v227-rosenman24a,
  title = 	 {Pre-Training Transformers for Fingerprinting to Improve Stress Prediction in fMRI},
  author =       {Rosenman, Gony and Malkiel, Itzik and Greental, Ayam and Hendler, Talma and Wolf, Lior},
  booktitle = 	 {Medical Imaging with Deep Learning},
  pages = 	 {212--234},
  year = 	 {2024},
  editor = 	 {Oguz, Ipek and Noble, Jack and Li, Xiaoxiao and Styner, Martin and Baumgartner, Christian and Rusu, Mirabela and Heinmann, Tobias and Kontos, Despina and Landman, Bennett and Dawant, Benoit},
  volume = 	 {227},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {10--12 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v227/rosenman24a/rosenman24a.pdf},
  url = 	 {https://proceedings.mlr.press/v227/rosenman24a.html},
  abstract = 	 {We harness a Transformer-based model and a pre-training procedure for fingerprinting on fMRI data, to enhance the accuracy of stress predictions. Our model, called MetricFMRI, first optimizes a pixel-based reconstruction loss. In a second unsupervised training phase, a triplet loss is used to encourage fMRI sequences of the same subject to have closer representations, while sequences from different subjects are pushed away from each other. Finally, supervised learning is used for the target task, based on the learned representation. We evaluate the performance of our model and other alternatives and conclude that the triplet training for the fingerprinting task is key to the improved accuracy of our method for the task of stress prediction. To obtain insights regarding the learned model, gradient-based explainability techniques are used, indicating that sub-cortical brain regions that are known to play a central role in stress-related processes are highlighted by the model.}
}


Contact: [Gony Rosenman](mailto:gonyrosenman@mail.tau.ac.il), [Itzik Malkiel](mailto:itzik.malkiel@microsoft.com).
