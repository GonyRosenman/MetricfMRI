import numpy as np
from torch.utils.data import DataLoader,Subset
from data_preprocess_and_load.dataset_wrappers import *
import copy
import random

class DataHandler():
    def __init__(self,test=False,**kwargs):
        self.test = test
        self.kwargs = kwargs
        self.dataset_name = kwargs.get('dataset_name')
        self.splits_folder = Path(kwargs.get('base_path')).joinpath('splits',self.dataset_name)
        if kwargs.get('kfold') is not None:
            self.K = kwargs.get('kfold')
            self.K_count = kwargs.get('kfold_count')
            assert self.K_count is not None,'when using folds, fold num must be specfiied every time'
            self.splits_folder = self.splits_folder.joinpath('{}_fold'.format(self.K))
            self.current_split = self.splits_folder.joinpath('fold_num_{}.txt'.format(self.K_count))
        else:
            self.seed = kwargs.get('seed')
            self.current_split = self.splits_folder.joinpath('seed_{}.txt'.format(self.seed))
        self.splits_folder.mkdir(exist_ok=True, parents=True)
        if kwargs.get('debug_split'):
            #D:\gonyr\ETFF\experiments\ayam_FT__13_08___08_19_56
            print('setting split to be equal to split in golden experiment')
            self.current_split = self.splits_folder.joinpath('seed_{}.txt'.format('11'))



    def log_label_dict(self,dataset):
        label_dict = dataset.get_label_dict()
        text = ''
        for name,val in label_dict.items():
            text += '{}:{}\n'.format(name,val)
        with open(self.kwargs.get('experiment_folder').joinpath('label_dictionary.txt'),'w+') as f:
            f.write(text)

    def get_dataset(self,**kwargs):
        task = kwargs.get('task').lower()
        if self.dataset_name == 'S1200':
            dataset =  Rest_1200_3D
        elif self.dataset_name == 'ucla':
            dataset = Ucla
        elif self.dataset_name == 'ayam':
            dataset = Ayam
        elif self.dataset_name == 'ptsd':
            dataset = (Ziv,Ayam,Tom)
        elif self.dataset_name == 'ziv':
            dataset = Ziv
        elif 'ziv' in self.dataset_name and 'ayam' in self.dataset_name:
            dataset = (Ziv,Ayam)
        else:
            raise NotImplementedError
        if kwargs.get('time_series') == 'only':
            if 'triplet' in kwargs.get('fine_tune_task').lower() or 'fingerprint' in task.lower():
                return concat_dynamic_loaders(dataset,dynamic_timeseries_triplet_loader,**self.kwargs)
            else:
                return concat_dynamic_loaders(dataset,dynamic_timeseries_loader,**self.kwargs)
        elif task == 'noisy_labels':
            return concat_dynamic_loaders(dataset,dynamic_noisy_label_loader,**self.kwargs)
        elif task == 'subject_triplet' or task == 'fingerprint_finetune':
            return concat_dynamic_loaders(dataset,dynamic_subject_triplet_loader,**self.kwargs)
        elif 'cosine' in task:
            return concat_dynamic_loaders(dataset,dynamic_triplet_loader,**self.kwargs)
        else:
            return dataset

    def merge_datasets(self,datasets_tuple):
        datasets_list = []
        merged_id = -1
        current_id = None
        for dataset in datasets_tuple:
            dataset = dataset(**self.kwargs)
            for sample_idx in range(len(dataset.index_l)):
                sample = dataset.index_l[sample_idx]
                new_id = sample[0]
                if current_id != new_id:
                    current_id = new_id
                    merged_id += 1
                dataset.index_l[sample_idx] = (merged_id,) + sample[1:]
            datasets_list.append(dataset)
        train_loader = ConcatDataset(datasets_list)
        val_loader = ConcatDataset(datasets_list)
        return train_loader,val_loader

    def current_split_exists(self):
        return self.current_split.exists()

    def create_dataloaders(self):
        dataset = self.get_dataset(**self.kwargs)
        print('grabbed dataloader of type {}...'.format(dataset.__name__))
        if isinstance(dataset,tuple):
            self.subject_list = []
            train_loader, val_loader = self.merge_datasets(dataset)
            for dataset in val_loader.datasets:
                dataset.augment = None
                self.subject_list += dataset.index_l
            test_loader = copy.copy(val_loader)
        else:
            train_loader = dataset(**self.kwargs)
            val_loader = copy.copy(train_loader)
            val_loader.augment = None
            test_loader = copy.copy(val_loader)
            self.subject_list = train_loader.index_l
        assert len(self.subject_list) > 0
        if hasattr(train_loader,'labels_used'):
            self.splits_folder = self.splits_folder.joinpath(train_loader.labels_used)
            self.splits_folder.mkdir(exist_ok=True,parents=True)
            self.current_split = self.splits_folder.joinpath('seed_{}.txt'.format(self.seed))
        if self.current_split_exists():
            train_names, val_names, test_names = self.load_split()
            #print('mega debug!!!!!!!!!!!!!!!!!!!!!\n\n!!!\n!\n!\n!!!!')
            #train_names = train_names[:5]
            #val_names = val_names[:5]
            #test_names = test_names[:5]
        else:
            if hasattr(self,'K'):
                self.determine_kfolds_randomly(**self.kwargs)
            else:
                train_names,val_names,test_names = self.determine_split_randomly(**self.kwargs)
        train_names, val_names, test_names = [str(x) for x in train_names],[str(x) for x in val_names],[str(x) for x in test_names]
        split = {'train_subjects': train_names, 'val_subjects': val_names, 'test_subjects': test_names}
        self.save_split(split)
        train_loader.split = split['train_subjects']
        val_loader.split = split['val_subjects']
        test_loader.split = split['test_subjects']

        train_idx, val_idx, test_idx = self.convert_subject_list_to_idx_list(train_names,val_names,test_names)
        #train_idx = [train_idx[x] for x in torch.randperm(len(train_idx))[:10]]
        #val_idx = [val_idx[x] for x in torch.randperm(len(val_idx))[:10]]
        #test_idx = [test_idx[x] for x in torch.randperm(len(test_idx))[:10]]
        #print('debug!!!')
        train_loader = Subset(train_loader, train_idx)
        val_loader = Subset(val_loader, val_idx)
        test_loader = Subset(test_loader, test_idx)

        training_generator = DataLoader(train_loader, **self.get_params(**self.kwargs))
        val_generator = DataLoader(val_loader, **self.get_params(eval=True,**self.kwargs))
        test_generator = DataLoader(test_loader, **self.get_params(eval=True,**self.kwargs)) if self.test else None
        return training_generator, val_generator, test_generator


    def get_params(self,eval=False,**kwargs):
        batch_size = kwargs.get('batch_size')
        workers = kwargs.get('workers')
        cuda = kwargs.get('cuda')
        if eval:
            workers = 0
        shuffle = kwargs.get('explain_task') is None

        #preserve reproducability accross seeds
        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'num_workers': workers,
                  'drop_last': True,
                  'pin_memory': True if cuda else False,
                  'persistent_workers': True if workers > 0 and cuda else False}
        if workers > 0:
            g = torch.Generator()
            g.manual_seed(kwargs.get('seed'))
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            params['worker_init_fn'] = seed_worker
            params['generator'] = g
        return params

    def save_split(self,sets_dict):
        if self.current_split_exists():
            return
        total_num_current = 0
        for l in sets_dict.values():
            total_num_current += len(l)
        if self.dataset_name == 'ayam':
            total_num = len(np.unique([x[1] for x in self.subject_list]))
        else:
            total_num = len(np.unique([x[0] for x in self.subject_list]))
        assert total_num == total_num_current, 'trying to save split but the number of subjects is not equal to the nubmer originally, probably a debugging issue\ncomment out the save method while debugging'
        with open(self.current_split,'w+') as f:
            for name,subj_list in sets_dict.items():
                f.write(name + '\n')
                for subj_name in subj_list:
                    f.write(str(subj_name) + '\n')

    def convert_subject_list_to_idx_list(self,train_names,val_names,test_names):
        if self.dataset_name == 'ayam':
            subj_idx = [x[1] for x in self.subject_list]
        else:
            subj_idx = np.array([str(x[0]) for x in self.subject_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx,val_idx,test_idx

    def determine_kfolds_randomly(self,**kwargs):
        if self.dataset_name == 'ayam':
            sub_ids = np.unique([x[1] for x in self.subject_list])
            S = len(sub_ids)
            sub_ids = sub_ids[torch.randperm(S)]
            fold_size = S // self.K
            val_size = (fold_size // 2) + 1
            for i in range(self.K):
                self.current_split = self.splits_folder.joinpath('fold_num_{}.txt'.format(i))
                start = i*fold_size
                test_names = sub_ids[start:start+fold_size]
                remaining = np.setdiff1d(sub_ids, test_names)
                remaining = remaining[torch.randperm(len(remaining))]
                train_names = remaining[:-val_size]
                val_names = remaining[-val_size:]
                split = {'train_subjects': train_names, 'val_subjects': val_names, 'test_subjects': test_names}
                self.save_split(split)
        else:
            raise NotImplementedError
        assert False,'for convenience, after creating kfold splits and saveing them to disk the program must be ran again to start the training.'

    def determine_split_randomly(self,**kwargs):
        train_percent = kwargs.get('train_split') if kwargs.get('train_split') is not None else 0.7
        val_percent = kwargs.get('val_split') if kwargs.get('val_split') is not None else 0.15
        if self.dataset_name == 'ayam':
            sub_ids = np.unique([x[1] for x in self.subject_list])
            S = len(sub_ids)
            sub_ids = sub_ids[torch.randperm(S)]
            S_train = int(S * train_percent)
            S_val = int(S * val_percent)
            train_names = sub_ids[:S_train]
            val_names = sub_ids[S_train:S_train+S_val]
            test_names = sub_ids[S_train+S_val:]
        else:
            S = len(np.unique([x[0] for x in self.subject_list]))
            S_train = int(S * train_percent)
            S_val = int(S * val_percent)
            train_names = np.random.choice(S, S_train, replace=False)
            remaining = np.setdiff1d(np.arange(S), train_names)
            val_names = np.random.choice(remaining,S_val, replace=False)
            test_names = np.setdiff1d(np.arange(S), np.concatenate([train_names,val_names]))
        return train_names,val_names,test_names

    def load_split(self):
        subject_order = open(self.current_split, 'r').readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(['train' in line for line in subject_order])
        val_index = np.argmax(['val' in line for line in subject_order])
        test_index = np.argmax(['test' in line for line in subject_order])
        train_names = subject_order[train_index + 1:val_index]
        val_names = subject_order[val_index+1:test_index]
        test_names = subject_order[test_index + 1:]
        return train_names,val_names,test_names