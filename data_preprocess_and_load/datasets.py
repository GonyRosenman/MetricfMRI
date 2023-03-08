import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import augmentations
import pandas as pd
from pathlib import Path
import numpy as np
from general_utils import PathHandler
import nibabel as nib


class BaseDataset(Dataset):
    def __init__(self,**kwargs):
        super(BaseDataset,self).__init__()
        self.device = None#torch.device('cuda') if kwargs.get('cuda') else torch.device('cpu')
        self.experiment_folder = kwargs.get('experiment_folder')
        self.task = kwargs.get('task')
        self.index_l = []
        self.norm = 'global_normalize'
        self.complementary = 'per_voxel_normalize'
        self.subj_identifier = 0
        if kwargs.get('only_per_voxel'):
            self.norm = 'per_voxel_normalize'
            self.complementary = None
            print('notice: loading only per voxel input')
        self.random_TR = kwargs.get('random_TR')
        self.set_augmentations(**kwargs)
        self.stride_factor = kwargs.get('stride_factor')
        self.stride_factor = 1 if self.stride_factor is None else self.stride_factor
        self.sequence_stride = kwargs.get('sequence_stride') if kwargs.get('sequence_stride') is not None else 1
        self.sequence_length = kwargs.get('sequence_length')
        self.sample_duration = self.sequence_length * self.sequence_stride
        self.stride = max(round(self.stride_factor * self.sample_duration),1)
        self.TR_skips = range(0,self.sample_duration,self.sequence_stride)
        self.label_dict = {}

    def verify_preprocessing(self,folder):
        root = Path(self.root)
        assert Path(folder).exists(), '{} does not exist\nmust run preprocessing_{}.py before running experiment!'.format(str(folder),root.name)


    def convert_class(self,class_name):
        if len(self.label_dict) == 0:
            self.label_dict[class_name] = torch.tensor([0])
        elif class_name not in self.label_dict:
            self.label_dict[class_name] = max(self.label_dict.values()) + 1
        return self.label_dict[class_name]

    def get_label_dict(self):
        if len(self.label_dict) == 0:
            return None
        else:
            return self.label_dict

    def get_input_shape(self):
        #TODO: fix this patch-up
        if 'oasis' in self.__class__.__name__.lower():
            shape = (91,109,91)
        else:
            shape = torch.load(os.path.join(self.index_l[0][2],self.index_l[0][3] + '.pt')).squeeze().shape
        return shape

    def set_augmentations(self,**kwargs):
        if kwargs.get('augment_prob') > 0:
            self.augment = transforms.Compose([augmentations.brain_gaussian(**kwargs)])
        else:
            self.augment = None

    def TR_int(self,TR_string):
        #todo: make use of TR_int everywhere
        return int(TR_string.replace('TR_',''))

    def TR_string(self,filename_TR,x):
        #all datasets should have the TR mentioned in the format of 'some prefix _ number.pt'
        TR_num = [xx for xx in filename_TR.split('_') if xx.isdigit()][0]
        assert len(filename_TR.split('_')) == 2
        filename = filename_TR.replace(TR_num,str(int(TR_num) + x)) + '.pt'
        return filename

    def determine_TR(self,TRs_path,TR):
        if self.random_TR:
            possible_TRs = len(os.listdir(TRs_path)) - self.sample_duration
            TR = 'TR_' + str(torch.randint(0,possible_TRs,(1,)).item())
        return TR

    def load_sequence(self, TRs_path, TR):
        # the logic of this function is that always the first channel corresponds to global norm and if there is a second channel it belongs to per voxel.
        TR = self.determine_TR(TRs_path,TR)
        y = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)),map_location=self.device).unsqueeze(0) for x in self.TR_skips], dim=4)
        if self.complementary is not None:
            y1 = torch.cat([torch.load(os.path.join(TRs_path, self.TR_string(TR, x)).replace(self.norm, self.complementary),map_location=self.device).unsqueeze(0)
                            for x in self.TR_skips], dim=4)
            y = torch.cat([y, y1], dim=0)
            del y1
        if y.isnan().any():
            #todo: make it better
            y = y.nan_to_num(np.nanmin(y))
        if self.augment is not None:
            y = self.augment(y)
        return y, TR

    def load_parcellated_sequence(self, TRs_path, TR):
        # the logic of this function is that always the first channel corresponds to global norm and if there is a second channel it belongs to per voxel.
        TR = self.determine_TR(TRs_path,TR)
        parcellated_TR_path = Path(TRs_path.replace(self.norm,self.parcellation.mask_path.stem))
        y = None
        for t in self.TR_skips:
            y_t_path = parcellated_TR_path.joinpath(self.TR_string(TR,t))
            if y_t_path.exists():
                y_t = torch.load(y_t_path)
            else:
                y_t_path.parent.mkdir(exist_ok=True)
                y_t = torch.load(Path(TRs_path.replace(self.norm,self.complementary)).joinpath(self.TR_string(TR,t)))
                y_t = self.parcellation.get_parcellated_volume(y_t.squeeze()).unsqueeze(1)
                torch.save(y_t,y_t_path)
            if y is None:
                y = y_t
            else:
                y = torch.cat([y,y_t],dim=1)
        return y

class Rest_1200_3D(BaseDataset):
    def __init__(self, **kwargs):
        super(Rest_1200_3D,self).__init__(**kwargs)
        self.root = r'D:\users\Gony\HCP-1200'
        self.meta_data = pd.read_csv(os.path.join(self.root, 'subject_data.csv'))
        self.meta_data_residual = pd.read_csv(os.path.join(self.root,'HCP_1200_precise_age.csv'))
        self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.verify_preprocessing(self.data_dir)
        self.subject_names = os.listdir(self.data_dir)
        self.subject_folders = []
        for i,subject in enumerate(os.listdir(self.data_dir)):
            try:
                age = torch.tensor(self.meta_data_residual[self.meta_data_residual['subject']==int(subject)]['age'].values[0])
            except Exception:
                age = self.meta_data[self.meta_data['Subject'] == int(subject)]['Age'].values[0]
                age = torch.tensor([float(x) for x in age.replace('+','-').split('-')]).mean()
            gender = self.meta_data[self.meta_data['Subject']==int(subject)]['Gender'].values[0]
            path_to_TRs = os.path.join(self.data_dir,subject,self.norm)
            subject_duration = len(os.listdir(path_to_TRs))#121
            session_duration = subject_duration - self.sample_duration
            filename = os.listdir(path_to_TRs)[0]
            filename = filename[:filename.find('TR')+3]

            for k in range(0,session_duration,self.stride):
                self.index_l.append((i, subject, path_to_TRs,filename + str(k),session_duration, age , self.convert_class(gender)))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj, subj_name, path_to_TRs, TR , session_duration, age, gender = self.index_l[index]
        age = age.float()
        y, TR = self.load_sequence(path_to_TRs,TR)
        return {'fmri_sequence':y,'subject':subj,'subject_classification':gender,'subject_regression':age,'TR':int(TR.split('_')[1])}


class Ucla(BaseDataset):
    def __init__(self, **kwargs):
        super(Ucla,self).__init__(**kwargs)
        self.root = str(PathHandler().ucla)
        self.meta_data = pd.read_csv(os.path.join(self.root, 'participants.tsv'), sep='\t')
        self.data_dir = os.path.join(self.root, 'rest','MNI_to_TRs')
        self.verify_preprocessing(self.data_dir)
        self.subjects = len(os.listdir(self.data_dir))
        self.subjects_names = os.listdir(self.data_dir)
        self.labels_used = kwargs.get('labels_used') if kwargs.get('labels_used') is not None else 'schz_bipolar_adhd_control'
        print('notice: only performing classification of the following classes - \n{}'.format(self.labels_used))
        for i, subject in enumerate(self.subjects_names):
            diagnosis = self.meta_data.loc[self.meta_data['participant_id'] == subject, ['diagnosis']].values[0][0]
            if diagnosis.lower() not in self.labels_used and self.task == 'fine_tune':
                continue
            TRs_path = os.path.join(self.data_dir, subject,'rest',self.norm)
            session_duration = len(os.listdir(TRs_path)) - self.sample_duration
            for k in range(0, session_duration, self.stride):
                self.index_l.append((i, subject, TRs_path, 'TR_' + str(k), session_duration,self.convert_class(diagnosis)))


    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name ,TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        y, TR = self.load_sequence(TRs_path,TR)
        input_dict = {'fmri_sequence':y,'subject':subj_num ,'subject_num':subj_num,'subject_classification':diagnosis , 'TR':int(TR.split('_')[1])}
        return input_dict

class OasisMRI(BaseDataset):
    def __init__(self, **kwargs):
        super(OasisMRI, self).__init__(**kwargs)
        self.root = '/media/data2/itzik/fmri_data'
        self.data_dir = os.path.join(self.root, 'oasis1')
        self.meta_data = pd.read_csv(os.path.join(self.data_dir, 'oasis_cross-sectional.csv'))
        self.data_dict = {}
        for file in Path(self.data_dir).rglob('*.nii.gz'):
            subject = file.name[file.name.find('OAS1_')+5:file.name.find('_MR1')]
            if subject in self.data_dict:
                self.data_dict[subject].append(str(file))
            else:
                self.data_dict[subject] = [str(file)]
        self.index_l = [[i,x] for i,x in enumerate(self.data_dict.keys())]

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subject_num = self.index_l[index][0]
        subject = self.index_l[index][1]
        AGE = self.meta_data.Age.iloc[[subject in x for x in self.meta_data.ID]].values[0]
        MRI1 = torch.from_numpy(nib.load(self.data_dict[subject][0]).get_fdata()).unsqueeze(0)
        MRI2 = torch.from_numpy(nib.load(self.data_dict[subject][1]).get_fdata()).unsqueeze(0)
        y = torch.cat([MRI1,MRI2],dim=0).unsqueeze(-1).to(dtype=torch.float32)
        return {'fmri_sequence':y,'subject_regression':torch.tensor(AGE).float(),'subject':subject_num}


class Tom(BaseDataset):
    def __init__(self, **kwargs):
        super(Tom,self).__init__(**kwargs)
        self.sessions = ['ses-1','ses-2','ses-3']
        self.root = r'D:\users\Gony\ptsd\tom'
        #self.meta_data = pd.read_csv(os.path.join(self.root, 'caps.csv'))
        self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.subject_names = os.listdir(self.data_dir)
        self.subjects = len(os.listdir(self.data_dir))
        self.index_l = []
        for j,subj in enumerate(os.listdir(self.data_dir)):
            for time in os.listdir(os.path.join(self.data_dir,subj)):
                for session in os.listdir(os.path.join(self.data_dir,subj,time)):
                    for task in os.listdir(os.path.join(self.data_dir,subj,time,session)):
                        path_to_TRs = os.path.join(self.data_dir,subj,time,session,task,self.norm)
                        session_duration = len(os.listdir(path_to_TRs)) - self.sample_duration
                        for k in range(0,session_duration,self.stride):
                            self.index_l.append((j,subj,path_to_TRs, 'TR_' + str(k), session_duration, self.convert_class(task)))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name, TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        diagnosis = np.nan
        y, TR = self.load_sequence(TRs_path, TR)
        input_dict = {'fmri_sequence': y, 'subject': subj_num, 'subject_classification': diagnosis,
                      'TR': int(TR.split('_')[1])}
        return input_dict

class Ayam(BaseDataset):
    def __init__(self, **kwargs):
        super(Ayam,self).__init__(**kwargs)
        self.sessions = ['ses-1', 'ses-2', 'ses-3']
        print('temporary debug root folder')
        self.norm = 'per_voxel_normalize'
        self.root = r'C:\Users\gonyr\ayam'
        self.meta_data = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        self.data_dir = os.path.join(self.root, 'MNI_to_TRs')
        self.verify_preprocessing(self.data_dir)
        self.subject_names = os.listdir(self.data_dir)
        self.subjects = len(os.listdir(self.data_dir))
        self.index_l = []
        if kwargs.get('ayam_low_stress_prediction') is not None:
            self.stress_categories = 3
        else:
            self.stress_categories = 2
        for k, subject in zip(range(0,self.subjects*self.stress_categories,self.stress_categories),self.subject_names):
            if '012' in subject:
                print('currently skipping subject 12 (debug)')
                continue
            ID = float(subject.replace('sub-','').lstrip('0'))
            try:
                meta = [self.meta_data.loc[self.meta_data.subId == ID].Pilot.values[0],
                        self.meta_data.loc[self.meta_data.subId == ID].Performance_Group.values[0]]
            except IndexError:
                meta = ['unknown','unknown']
            for task in os.listdir(os.path.join(self.data_dir, subject)):
                for run in os.listdir(os.path.join(self.data_dir,subject,task)):
                    TRs_path = os.path.join(self.data_dir, subject, task,run,self.norm)
                    session_duration = len(os.listdir(TRs_path)) - self.sample_duration
                    if self.stress_categories == 3:
                        if '2' in run:
                            if 'ns' in task:
                                task_ = 'rest_ls'
                                subj_num = k + 1
                            else:
                                task_ = 'rest_st'
                                subj_num = k + 2
                        else:
                            task_ = 'rest_ns'
                            subj_num = k
                    else:
                        task_ = task if '2' in run else 'rest_ns'
                        subj_num = k if 'ns' in task_ else k + 1
                    for kk in range(0, session_duration, self.stride):
                        if self.task == 'tsne':
                            diagnosis = [[self.convert_class(task_)] + meta]
                        else:
                            diagnosis = self.convert_class(task_)
                        self.index_l.append((subj_num, subject, TRs_path, 'TR_' + str(kk), session_duration,diagnosis))



    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name ,TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        try:
            y, TR = self.load_sequence(TRs_path,TR)
        except Exception as e:
            with open(r'D:\users\Gony\ayam\faulty_files.txt','a+') as f:
                f.write('couldnt load {} because of {}\n\n'.format(TRs_path+str(TR),e))
        input_dict = {'fmri_sequence':y,'subject':subj_num ,'subject_classification':diagnosis , 'TR':int(TR.split('_')[1])}
        return input_dict

class Ziv(BaseDataset):
    def __init__(self, **kwargs):
        super(Ziv,self).__init__(**kwargs)
        self.sessions = ['ses-1','ses-2','ses-3']
        self.root = Path(r'D:\users\Gony\ptsd\ziv')
        self.meta_data = pd.read_csv(self.root.joinpath('caps.csv'))
        self.data_dir = self.root.joinpath('MNI_to_TRs')
        self.subject_names = list(self.data_dir.iterdir())
        self.index_l = []
        for i,subject_dir in enumerate(self.data_dir.rglob('sub*')):
            for task_dir in subject_dir.iterdir():
                for session_dir in task_dir.iterdir():
                    session = session_dir.name
                    task = task_dir.name
                    if not task == 'fcmri':
                        continue
                    subject = subject_dir.name
                    TRs_path = session_dir.joinpath(self.norm)
                    session_duration = len(list(TRs_path.iterdir())) - self.sample_duration
                    for k in range(0,session_duration, self.stride):
                        self.index_l.append((i, subject[-4:], str(TRs_path), 'TR_' + str(k), session_duration, (task, session)))

    def __len__(self):
        N = len(self.index_l)
        return N

    def __getitem__(self, index):
        subj_num, subj_name ,TRs_path, TR, session_duration, diagnosis = self.index_l[index]
        diagnosis = torch.tensor([float('nan')])
        y, TR = self.load_sequence(TRs_path,TR)
        input_dict = {'fmri_sequence':y,'subject':subj_num ,'subject_classification':diagnosis , 'TR':int(TR.split('_')[1])}
        return input_dict

