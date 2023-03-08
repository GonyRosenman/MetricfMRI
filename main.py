from general_utils import *
from trainer import Trainer
import os
from pathlib import Path
import argparse

def get_arguments(base_path):
    """
    handle arguments from commandline.
    some other hyper parameters can only be changed manually (such as model architecture,dropout,etc)
    notice some arguments are global and take effect for the entire three phase training process, while others are determined per phase
    """
    delete_stale_experiments()
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default=base_path)
    parser.add_argument('--seed', type=int, default=3)#55555555)
    parser.add_argument('--dataset_name', type=str, default="ayam")
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--log_dir', type=str, default=os.path.join(base_path, 'runs'))
    parser.add_argument('--random_TR', default=True)
    parser.add_argument('--transformer_hidden_layers', default=4)
    parser.add_argument('--load_cls_embedding', default=False)
    parser.add_argument('--cls_pooling', default=True)
    parser.add_argument('--perceptual_layers', default=[True,True,False])
    parser.add_argument('--stride_factor', default=0.25)
    parser.add_argument('--only_per_voxel', default=True)
    parser.add_argument('--margin', default=0.15)
    parser.add_argument('--fine_tune_task',
                        default='classification',
                        choices=['regression','classification','binary_classification','Triplet'],
                        help='fine tune model objective. choose binary_classification in case of a binary classification task')
    parser.add_argument('--running_mean_size', default=2500)
    parser.add_argument('--early_stop', default=True)


    ##phase 1
    parser.add_argument('--task_phase1', type=str, default='autoencoder_reconstruction')
    parser.add_argument('--batch_size_phase1', type=int, default=8)
    parser.add_argument('--validation_frequency_phase1', type=int, default=2000)
    parser.add_argument('--nEpochs_phase1', type=int, default=10)
    parser.add_argument('--augment_prob_phase1', default=0)
    parser.add_argument('--weight_decay_phase1', default=1e-7)
    parser.add_argument('--lr_init_phase1', default=1e-3)
    parser.add_argument('--lr_gamma_phase1', default=0.97)
    parser.add_argument('--lr_step_phase1', default=1000)
    parser.add_argument('--sequence_length_phase1', default=1)
    parser.add_argument('--workers_phase1', default=0)
    parser.add_argument('--parallel_phase1', default=False)



    ##phase 2
    parser.add_argument('--task_phase2', type=str, default='subject_triplet')
    parser.add_argument('--batch_size_phase2', type=int, default=1)
    parser.add_argument('--validation_frequency_phase2', type=int, default=500)
    parser.add_argument('--nEpochs_phase2', type=int, default=8)
    parser.add_argument('--augment_prob_phase2', default=0)
    parser.add_argument('--weight_decay_phase2', default=1e-2)
    parser.add_argument('--lr_init_phase2', default=1e-5)
    parser.add_argument('--final_lr_phase2', default=1e-6)
    parser.add_argument('--lr_gamma_phase2', default=0.97)
    parser.add_argument('--lr_step_phase2', default=1500)
    parser.add_argument('--sequence_length_phase2', default=30)
    parser.add_argument('--workers_phase2', default=0)
    parser.add_argument('--parallel_phase2', default=False)
    parser.add_argument('--save_all_checkpoints_phase2', default=False)


    ##phase 3
    parser.add_argument('--task_phase3', type=str, default='fine_tune')
    parser.add_argument('--batch_size_phase3', type=int, default=1)
    parser.add_argument('--validation_frequency_phase3', type=int, default=1000)
    parser.add_argument('--nEpochs_phase3', type=int, default=35)
    parser.add_argument('--augment_prob_phase3', default=0.25)
    parser.add_argument('--weight_decay_phase3', default=1e-2)
    parser.add_argument('--lr_init_phase3', default=2e-5)
    parser.add_argument('--final_lr_phase3', default=1e-6)
    parser.add_argument('--lr_gamma_phase3', default=0.9)
    parser.add_argument('--lr_step_phase3', default=1000)
    parser.add_argument('--sequence_length_phase3', default=30)
    parser.add_argument('--workers_phase3', default=0)
    parser.add_argument('--parallel_phase3', default=False)
    parser.add_argument('--balance_samples_phase3', default=True)
    args = parser.parse_args()
    return args

def setup(cuda_num):
    cuda_num = str(cuda_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    base_path = os.getcwd()
    os.makedirs(os.path.join(base_path,'experiments'),exist_ok=True)
    os.makedirs(os.path.join(base_path,'runs'),exist_ok=True)
    os.makedirs(os.path.join(base_path, 'splits'), exist_ok=True)
    return base_path

def test(args,model_weights_path):
    experiment_folder = '{}_{}_{}'.format(args.dataset_name, 'test_{}'.format(args.fine_tune_task), datestamp())
    experiment_folder = os.path.join(args.base_path,'tests', experiment_folder)
    os.makedirs(experiment_folder)
    trainer = Trainer(sets=['test'],**args)
    trainer.testing()

def main(base_path):
    args = get_arguments(base_path)
    args.seed = 5555555
    model_weights_path_phase2 = run_phase(args, None, '1', 'AR_pervoxel_', Trainer)
    model_weights_path_phase3 = run_phase(args,model_weights_path_phase2,'2','subj_trip_per_voxel_',Trainer)
    #run_phase(args,model_weights_path_phase3,'3','FT_',Trainer)




if __name__ == '__main__':
    base_path = setup(cuda_num='0')
    main(base_path)
