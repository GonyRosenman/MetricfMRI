from parcellations.make_parcellations import RawDataParcellation
from data_preprocess_and_load.datasets import *

def concat_dynamic_loaders(BaseClass,dynamic_func,**kwargs):
    if isinstance(BaseClass,tuple):
        OutClass = ()
        for BaseClass_x in BaseClass:
            OutClass_x = dynamic_func(BaseClass_x,**kwargs)
            OutClass += (OutClass_x,)
    else:
        OutClass = dynamic_func(BaseClass,**kwargs)
    return OutClass

def dynamic_timeseries_triplet_loader(BaseClass,**kwargs):
    T = kwargs.get('sequence_length')
    assert kwargs.get('atlases') is not None
    assert T>1
    if kwargs.get('random_TR'):
        print('in triplet loss, a random TR could mean overlap between positive samples, changing to fixed TR...')
        kwargs['random_TR'] = False
    class FC_Triplet_Class(BaseClass):
        def __init__(self,**kwargs):
            super(FC_Triplet_Class,self).__init__(**kwargs)
            self.parcellation = RawDataParcellation(**kwargs)
            self.data_dir = Path(self.root).joinpath('functional_connectivity_{}'.format(self.parcellation.specific_parcellation_dir.name))
            self.verify_preprocessing(self.data_dir)
            self.load_time_series_data()
            self.subj_identifier = 1 if 'ayam' in BaseClass.__name__.lower() else 0
            for sample in self.index_l:
                TR = self.TR_int(sample[3])
                if TR <= self.sample_duration and (TR + self.sample_duration) > sample[4]:
                    assert False, 'size of sub sequence too large for the dataset, no positive triplet without overlap -> change sequence length attribute'

        def load_time_series_data(self):
            self.data_dict = {}
            for file_path in self.data_dir.rglob('*.pt'):
                run = file_path.parent.parent.name
                task = file_path.parent.parent.parent.name
                subject = file_path.parent.parent.parent.parent.name
                data = torch.load(file_path,map_location=self.device)
                name = subject + '_' + task + '_' + run
                self.data_dict[name] = data

        def __len__(self):
            N = len(self.index_l)
            return N

        def __getitem__(self,index):
            subj_num_anchor, subj_name_anchor, TRs_path_anchor, TR_anchor, _, diagnosis_vec_anchor = self.index_l[index]
            subj_id_anchor = self.index_l[index][self.subj_identifier]
            try:
                positive_samples = [x for x in self.index_l if x[self.subj_identifier] == subj_id_anchor and abs(self.TR_int(x[3]) - self.TR_int(TR_anchor)) >= self.sample_duration]
                negative_samples = [x for x in self.index_l if x[self.subj_identifier] != subj_id_anchor and str(x[self.subj_identifier]) in self.split]
            except RuntimeError:
                with open(str(PathHandler().work_dir.joinpath('debug_message.txt')),'a+') as f:
                    f.write(TRs_path_anchor)
                    f.write(self.TR_int(TR_anchor))
                    f.write('___')
            subj_num_positive, subj_name_positive, TRs_path_positive, TR_positive, _, diagnosis_vec_positive = positive_samples[torch.randint(0, len(positive_samples), (1,))]
            subj_num_negative, subj_name_negative, TRs_path_negative, TR_negative, _, diagnosis_vec_negative = negative_samples[torch.randint(0, len(negative_samples), (1,))]
            TRs_path_anchor = Path(TRs_path_anchor)
            subject_anchor = TRs_path_anchor.parent.parent.parent.name
            task_anchor = TRs_path_anchor.parent.parent.name
            run_anchor = TRs_path_anchor.parent.name
            name_anchor = subject_anchor + '_' + task_anchor + '_' + run_anchor
            t_anchor = self.TR_int(TR_anchor)
            TRs_path_positive = Path(TRs_path_positive)
            subject_positive = TRs_path_positive.parent.parent.parent.name
            task_positive = TRs_path_positive.parent.parent.name
            run_positive = TRs_path_positive.parent.name
            name_positive = subject_positive + '_' + task_positive + '_' + run_positive
            t_positive = self.TR_int(TR_positive)
            TRs_path_negative = Path(TRs_path_negative)
            subject_negative = TRs_path_negative.parent.parent.parent.name
            task_negative = TRs_path_negative.parent.parent.name
            run_negative = TRs_path_negative.parent.name
            name_negative = subject_negative + '_' + task_negative + '_' + run_negative
            t_negative = self.TR_int(TR_negative)
            y = torch.cat([self.data_dict[name_anchor][t_anchor:t_anchor+self.sequence_length,:].T.to(dtype=torch.float32).unsqueeze(0),
                           self.data_dict[name_positive][t_positive:t_positive + self.sequence_length, :].T.to(dtype=torch.float32).unsqueeze(0),
                           self.data_dict[name_negative][t_negative:t_negative + self.sequence_length, :].T.to(dtype=torch.float32).unsqueeze(0)],dim=0)
            diagnosis = torch.tensor([diagnosis_vec_anchor.to(dtype=torch.int64),
                                      diagnosis_vec_positive.to(dtype=torch.int64),
                                      diagnosis_vec_negative.to(dtype=torch.int64)])
            t = torch.tensor([t_anchor,t_positive,t_negative])
            subj_num = torch.tensor([subj_num_anchor,subj_num_positive,subj_num_negative])
            if self.task == 'fingerprint_finetune':
                diagnosis = torch.tensor([1.0,0.0])
            input_dict = {'fmri_sequence': y, 'subject_num': subj_num,'subject_classification': diagnosis,'TR': t}
            return input_dict
    return FC_Triplet_Class

def dynamic_timeseries_loader(BaseClass,**kwargs):
    T = kwargs.get('sequence_length')
    assert kwargs.get('atlases') is not None
    assert T>1
    class FC_Class(BaseClass):
        def __init__(self,**kwargs):
            super(FC_Class,self).__init__(**kwargs)
            self.parcellation = RawDataParcellation(**kwargs)
            self.data_dir = Path(self.root).joinpath('functional_connectivity_{}'.format(self.parcellation.specific_parcellation_dir.name))
            self.verify_preprocessing(self.data_dir)
            self.load_time_series_data()

        def load_time_series_data(self):
            self.data_dict = {}
            for file_path in self.data_dir.rglob('*.pt'):
                run = file_path.parent.parent.name
                task = file_path.parent.parent.parent.name
                subject = file_path.parent.parent.parent.parent.name
                data = torch.load(file_path,map_location=self.device)
                name = subject + '_' + task + '_' + run
                self.data_dict[name] = data

        def __len__(self):
            N = len(self.index_l)
            return N

        def __getitem__(self,index):
            subj_num, subj_name, TRs_path, TR, session_duration, diagnosis = self.index_l[index]
            TRs_path = Path(TRs_path)
            subject = TRs_path.parent.parent.parent.name
            task = TRs_path.parent.parent.name
            run = TRs_path.parent.name
            name = subject + '_' + task + '_' + run
            t = self.TR_int(TR)
            y = self.data_dict[name][t:t+self.sequence_length,:].T.to(dtype=torch.float32)
            diagnosis = diagnosis.to(dtype=torch.int64)
            if self.baseline_name == 'FC_MLP':
                y = torch.cov(y)
            input_dict = {'fmri_sequence': y, 'subject_num': subj_num,'subject_classification': diagnosis,'TR': t}
            return input_dict
    return FC_Class

def dynamic_subject_triplet_loader(BaseClass,**kwargs):
    T = kwargs.get('sequence_length')
    assert T > 1, 'triplet loss not working for sequence length 1'
    if kwargs.get('random_TR'):
        print('in triplet loss, a random TR could mean overlap between positive samples, changing to fixed TR...')
        kwargs['random_TR'] = False
    class SubjectTriplet_Class(BaseClass):
        def __init__(self,**kwargs):
            super(SubjectTriplet_Class,self).__init__(**kwargs)
            self.debug = kwargs.get('debug')
            self.subj_identifier = 1 if 'ayam' in BaseClass.__name__.lower() else 0
            for sample in self.index_l:
                TR = self.TR_int(sample[3])
                if TR <= self.sample_duration and (TR + self.sample_duration) > sample[4]:
                    assert False, 'size of sub sequence too large for the dataset, no positive triplet without overlap -> change sequence length attribute'

        def __len__(self):
            N = len(self.index_l)
            return N

        def __getitem__(self, index):
            subj_num_anchor, subj_name_anchor, TRs_path_anchor, TR_anchor, _, diagnosis_vec_anchor = self.index_l[index]
            subj_id_anchor = self.index_l[index][self.subj_identifier]
            positive_samples = [x for x in self.index_l if x[self.subj_identifier] == subj_id_anchor and abs(self.TR_int(x[3]) - self.TR_int(TR_anchor)) >= self.sample_duration]
            if self.debug:
                negative_samples = [x for x in self.index_l if x[self.subj_identifier] != subj_id_anchor]
            else:
                negative_samples = [x for x in self.index_l if x[self.subj_identifier] != subj_id_anchor and str(x[self.subj_identifier]) in self.split]
            subj_num_positive, subj_name_positive, TRs_path_positive, TR_positive, _, diagnosis_vec_positive = \
                positive_samples[torch.randint(0, len(positive_samples), (1,))]
            subj_num_negative, subj_name_negative, TRs_path_negative, TR_negative, _, diagnosis_vec_negative = \
                negative_samples[torch.randint(0, len(negative_samples), (1,))]

            y_anchor, TR_anchor = self.load_sequence(TRs_path_anchor, TR_anchor)
            y_positive, TR_positive = self.load_sequence(TRs_path_positive, TR_positive)
            y_negative, TR_negative = self.load_sequence(TRs_path_negative, TR_negative)

            y = torch.cat([y_anchor.unsqueeze(0), y_positive.unsqueeze(0), y_negative.unsqueeze(0)], dim=0)
            subj_num = torch.tensor([subj_num_anchor, subj_num_positive, subj_num_negative])
            subj_diag = torch.tensor([diagnosis_vec_anchor, diagnosis_vec_positive, diagnosis_vec_negative])
            TR = torch.tensor([self.TR_int(TR_anchor),self.TR_int(TR_positive),self.TR_int(TR_negative)])
            if self.task == 'fingerprint_finetune':
                subj_diag = torch.tensor([1.0,0.0])
            input_dict = {'fmri_sequence': y, 'subject_num': subj_num, 'subject_diagnosis': subj_diag, 'TR': TR}
            return input_dict
    return SubjectTriplet_Class

def dynamic_noisy_label_loader(BaseClass, **kwargs):
    T = kwargs.get('sequence_length')
    batch = kwargs.get('batch_size')
    assert T > 1
    assert batch == 1
    class TripletLabelClass(BaseClass):
        def __init__(self, **kwargs):
            super(TripletLabelClass, self).__init__(**kwargs)

        def __len__(self):
            N = len(self.index_l)
            return N

        def __getitem__(self, index):
            subj_num_anchor, subj_name_anchor, TRs_path_anchor, TR_anchor, _, diagnosis_vec_anchor = self.index_l[index]
            positive_samples = [x for x in self.index_l if
                                x[0] != subj_num_anchor and (x[-1] == diagnosis_vec_anchor).all()]
            negative_samples = [x for x in self.index_l if (x[-1] != diagnosis_vec_anchor).any()]
            subj_num_positive, subj_name_positive, TRs_path_positive, TR_positive, _, diagnosis_vec_positive = \
            positive_samples[torch.randint(0, len(positive_samples), (1,))]
            subj_num_negative, subj_name_negative, TRs_path_negative, TR_negative, _, diagnosis_vec_negative = \
            negative_samples[torch.randint(0, len(negative_samples), (1,))]

            y_anchor, TR_anchor = self.load_sequence(TRs_path_anchor, TR_anchor)
            y_positive, TR_positive = self.load_sequence(TRs_path_positive, TR_positive)
            y_negative, TR_negative = self.load_sequence(TRs_path_negative, TR_negative)

            y = torch.cat([y_anchor.unsqueeze(0), y_positive.unsqueeze(0), y_negative.unsqueeze(0)], dim=0)
            diagnosis_vec = torch.cat([x.unsqueeze(0) for x in [diagnosis_vec_anchor,diagnosis_vec_positive,diagnosis_vec_negative]],dim=0)
            subject_num = torch.tensor([subj_num_anchor,subj_num_positive,subj_num_negative])
            t = torch.tensor([self.TR_int(TR_anchor),self.TR_int(TR_positive),self.TR_int(TR_negative)])
            input_dict = {'fmri_sequence': y,'diagnosis':diagnosis_vec,'TR':t,'subject_num':subject_num}
            return input_dict
    return TripletLabelClass

def dynamic_triplet_loader(BaseClass, **kwargs):
    T = kwargs.get('sequence_length')
    assert T > 1
    batch = kwargs.get('batch_size')
    assert batch == 1
    class TripletClass(BaseClass):
        def __init__(self, **kwargs):
            super(TripletClass, self).__init__(**kwargs)
            self.positive_condition ,self.negative_condition = kwargs.get('cosine_conditions')
            self.subj_identifier = 1 if 'ayam' in BaseClass.__name__.lower() else 0
        def __len__(self):
            N = len(self.index_l)
            return N

        def __getitem__(self, index):
            subj_num_anchor, subj_name_anchor, TRs_path_anchor, TR_anchor, _, diagnosis_vec_anchor = self.index_l[index]
            positive_samples = [x for x in self.index_l if self.positive_condition(base=self.index_l[index],sample=x) and str(x[self.subj_identifier]) in self.split]
            negative_samples = [x for x in self.index_l if self.negative_condition(base=self.index_l[index],sample=x) and str(x[self.subj_identifier]) in self.split]
            subj_num_positive, subj_name_positive, TRs_path_positive, TR_positive, _, diagnosis_vec_positive = positive_samples[torch.randint(0, len(positive_samples), (1,))]
            subj_num_negative, subj_name_negative, TRs_path_negative, TR_negative, _, diagnosis_vec_negative = negative_samples[torch.randint(0, len(negative_samples), (1,))]

            y_anchor, TR_anchor = self.load_sequence(TRs_path_anchor, TR_anchor)
            y_positive, TR_positive = self.load_sequence(TRs_path_positive, TR_positive)
            y_negative, TR_negative = self.load_sequence(TRs_path_negative, TR_negative)

            y = torch.cat([y_anchor.unsqueeze(0), y_positive.unsqueeze(0), y_negative.unsqueeze(0)], dim=0)
            subj_num = torch.tensor([subj_num_anchor,subj_num_positive,subj_num_negative])
            subj_diag = torch.tensor([diagnosis_vec_anchor,diagnosis_vec_positive,diagnosis_vec_negative])
            TR = torch.tensor([self.TR_int(TR_anchor),self.TR_int(TR_positive),self.TR_int(TR_negative)])
            input_dict = {'fmri_sequence': y,'subject_num':subj_num,'subject_diagnosis':subj_diag,'TR':TR}
            return input_dict

    return TripletClass

