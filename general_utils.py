import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import os
import dill
import shutil
from pathlib import Path
import sys
import random
import time

def save_code_to_disk(experiment_dir):
    code_dir = experiment_dir.joinpath('code_at_execution_time')
    code_dir.mkdir(exist_ok=True)
    #for code_file in PathHandler().work_dir.rglob('*.py'):
    #alternative version that goes through 2 subdirectories, instead of all subdirectories, more efficient.
    for code_file in list(PathHandler().work_dir.glob('*.py')) + list(PathHandler().work_dir.glob('*/*.py')) + list(PathHandler().work_dir.glob('*/*/*.py')):
        #if 'experiments' in str(code_file):
        #    continue
        new_path = ''
        parent_dir = code_file.parent
        while parent_dir.name != PathHandler().work_dir.name:
            new_path = Path(parent_dir.name).joinpath(new_path)
            parent_dir = parent_dir.parent
        new_path = Path(new_path).joinpath(code_file.name)
        new_path = code_dir.joinpath(new_path)
        new_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(code_file,new_path)

def datestamp():
    time_x = datetime.now().strftime("%d_%m___%H_%M_%S")
    for _ in PathHandler().experiments.glob('*' + time_x + '*'):
        #this time stamp exists already, wait a second to create a new time stamp
        time.sleep(1)
        time_x = datetime.now().strftime("%d_%m___%H_%M_%S")
    return time_x

def reproducibility(**kwargs):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print('notice: calling reproduce function!!!')
    seed = kwargs.get('seed')
    cuda = kwargs.get('cuda')
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    #random.seed(seed)
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = True

def sort_args(phase, args):
    phase_specific_args = {}
    for name, value in args.items():
        if not 'phase' in name:
            phase_specific_args[name] = value
        elif 'phase' + phase in name:
            phase_specific_args[name.replace('_phase' + phase, '')] = value
    return phase_specific_args

def args_logger(args):
    args_to_pkl(args)
    args_to_text(args)

def load_args_as_dict(path_to_pkl):
    with open(path_to_pkl, 'rb') as f:
        args = dill.load(f)
    return args

def args_to_pkl(args):
    with open(os.path.join(args.experiment_folder,'arguments_as_is.pkl'),'wb') as f:
        dill.dump(vars(args),f)

def args_to_text(args):
    with open(os.path.join(args.experiment_folder,'argument_documentation.txt'),'w+') as f:
        for name,arg in vars(args).items():
            f.write('{}: {}\n'.format(name,arg))

def delete_stale_experiments():
    q = False
    base_path = os.getcwd()
    experiments_folder = Path(os.path.join(base_path,'experiments'))
    for folder in experiments_folder.iterdir():
        time_then = folder.stat().st_mtime
        time_now = datetime.now().timestamp()
        if abs(time_now - time_then)/3600 < 4:
            continue
        if not any(['.pt' in str(file) for file in folder.iterdir()]):
            shutil.rmtree(folder)
            q = True
    if q:
        print('warning: deleted some empty files')


def run_phase(args,loaded_model_weights_path,phase_num,phase_name,trainer_class):
    """
    main process that runs each training phase
    :return path to model weights (pytorch file .pth) aquried by the current training phase
    """
    if isinstance(loaded_model_weights_path,list):
        for loaded_model_weights_path_ in loaded_model_weights_path:
            model_weights_path = run_phase(args,loaded_model_weights_path_,phase_num,phase_name,trainer_class)
    else:
        reproducibility(**vars(args))
        experiment_folder = '{}_{}_{}'.format(args.dataset_name,phase_name,datestamp())
        experiment_folder = Path(os.path.join(args.base_path,'experiments',experiment_folder))
        os.makedirs(experiment_folder)
        save_code_to_disk(experiment_folder)
        setattr(args,'loaded_model_weights_path_phase' + phase_num,loaded_model_weights_path)
        args.experiment_folder = experiment_folder
        args.experiment_title = experiment_folder.name
        fine_tune_task = args.fine_tune_task
        args_logger(args)
        args = sort_args(phase_num, vars(args))
        S = ['train','val']
        trainer = trainer_class(sets=S,**args)
        trainer.training()
        if phase_num == '3' and not fine_tune_task == 'regression':
            critical_metric = 'accuracy'
        else:
            critical_metric = 'loss'
        if 'save_all_checkpoints' in args and args['save_all_checkpoints']:
            model_weights_path = list(Path(trainer.writer.experiment_folder).glob('*_mid_BEST_val_{}.pth'.format(critical_metric)))
            model_weights_path = [str(x) for x in model_weights_path]
        else:
            model_weights_path = os.path.join(trainer.writer.experiment_folder,trainer.writer.experiment_title + '_BEST_val_{}.pth'.format(critical_metric))
    return model_weights_path



class PathHandler():
    def __init__(self):
        if sys.platform == 'linux':
            self.work_dir = Path('/media/data2/itzik/dev/MetricfMRI')
            self.experiments = Path('/media/data2/itzik/dev/MetricfMRI/experiments')
            self.ucla = Path('/media/data2/itzik/fmri_data/ucla_final_version')
            self.ziv = None
            self.exemplar = Path('/media/data2/itzik/fmri_data/hcp_exemplar_data/subj1/rfMRI_REST1_LR.nii.gz')
            self.baselines = Path('/media/data2/itzik/MetricfMRI/baselines')
        else:
            self.work_dir = Path(r'K:\MetricfMRI')
            self.experiments = Path(r'K:\MetricfMRI\experiments')
            self.ucla = Path(r'K:\ucla\ucla\ucla\output')
            self.ziv = Path(r'K\ptsd\ziv\caps.csv')
            self.exemplar = Path(r'K:\HCP-1200\one_patient_original_version\rfMRI_REST1_LR.nii.gz')
            self.baselines = Path(r'K:\MetricfMRI\baselines')

