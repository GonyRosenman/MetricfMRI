from torch.nn import MSELoss,L1Loss,BCELoss,CrossEntropyLoss
from losses import Percept_Loss,make_custome_triplet_loss
import csv
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from itertools import zip_longest
from metrics import Metrics
import torch
from parcellations.make_parcellations import Parcellation
from explainability.explainer_class import ExplainerTSNE

class Writer():
    """
    main class to handle logging the results, both to tensorboard and to a local csv file
    """
    def __init__(self,sets,**kwargs):
        self.device = torch.device('cuda:0') if kwargs.get('cuda') else torch.device('cpu')
        self.register_args(**kwargs)
        self.register_losses(**kwargs)
        self.create_score_folders()
        self.metrics = Metrics()
        self.sets = sets
        self.total_train_steps = 0
        self.eval_iter = 0
        self.subject_accuracy = {}
        self.tensorboard = SummaryWriter(log_dir=self.tensorboard_dir, comment=self.experiment_title)
        self.experiment_hyperparameters_to_tensorboard()
        self.initialize_losses(sets)

    def initialize_losses(self,sets):
        for set in sets:
            setattr(self,'total_{}_loss_values'.format(set),[])
            history = 'total_{}_loss_history'.format(set)
            if not hasattr(self,history):
                setattr(self,history,[])
        for name, loss_dict in self.losses.items():
            if loss_dict['is_active']:
                for set in sets:
                    setattr(self, '{}_{}_loss_values'.format(name,set),[])
                    history = '{}_{}_loss_history'.format(name,set)
                    if not hasattr(self,history):
                        setattr(self,history,[])

    def get_partial_epoch(self):
        return self.total_train_steps / self.train_steps_per_epoch

    def adjust_dataset(self,train_loader):
        self.set_labels_dict(train_loader)
        self.train_steps_per_epoch = len(train_loader)
        if hasattr(self,'balance_samples') and self.balance_samples:
            print('balancing samples with weighted loss')
            self.balance_samples_(train_loader)

    def balance_samples_(self,train_loader):
        index_l = train_loader.dataset.dataset.index_l
        if len(index_l[0][-1]) == 1:
            nSamples = torch.cat([x[-1].unsqueeze(0) for x in train_loader.dataset.dataset.index_l], dim=0).to(dtype=torch.int16)
            nSamples = torch.bincount(nSamples.squeeze())
        else:
            nSamples = torch.cat([x[-1].unsqueeze(0) for x in train_loader.dataset.dataset.index_l],dim=0).sum(0)
        sampleWeights = nSamples / nSamples.sum()
        sampleWeights = (1/sampleWeights) / (1/sampleWeights).sum()
        sampleWeights *= 1/sampleWeights.max()
        sampleWeights = sampleWeights.to(device=self.device)
        self.losses[self.fine_tune_task]['criterion'].__init__(weight=sampleWeights)

    def create_score_folders(self):
        self.tensorboard_dir = Path(os.path.join(self.log_dir, self.experiment_title))
        self.csv_path = os.path.join(self.experiment_folder, 'history')
        os.makedirs(self.csv_path,exist_ok=True)
        if self.task == 'fine_tune':
            self.per_subject_predictions = os.path.join(self.experiment_folder, 'per_subject_predictions')
            os.makedirs(self.per_subject_predictions,exist_ok=True)

    def save_history_to_csv(self):
        rows = [getattr(self, x) for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)]
        column_names = tuple([x for x in dir(self) if 'history' in x and isinstance(getattr(self, x), list)])
        export_data = zip_longest(*rows, fillvalue='')
        with open(os.path.join(self.csv_path, 'full_scores.csv'), 'w', encoding="ISO-8859-1", newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(column_names)
            wr.writerows(export_data)

    def loss_summary(self,lr):
        self.scalar_to_tensorboard('learning_rate',lr)
        loss_d = self.append_total_to_losses()
        for name, loss_dict in loss_d.items():
            if loss_dict['is_active']:
                for set in self.sets:
                    title = name + '_' + set
                    values = getattr(self,title + '_loss_values')
                    if len(values) == 0:
                        print('warning: zero length for values at {}'.format(title))
                        continue
                    score = np.mean(values)
                    history = getattr(self,title + '_loss_history')
                    history.append(score)
                    print('{}: {}'.format(title,score))
                    setattr(self,title + '_loss_history',history)
                    self.scalar_to_tensorboard(title,score)

    def accuracy_summary(self,mid_epoch):
        torch.set_printoptions(precision=7)
        pred_all_sets = {x:[] for x in self.sets}
        truth_all_sets = {x:[] for x in self.sets}
        intra_subject_variance_all_sets = {x:[] for x in self.sets}
        metrics = {}
        for subj_name,subj_dict in self.subject_accuracy.items():
            subj_pred = subj_dict['score'].mean(dim=0)
            subj_error = subj_dict['score'].std(dim=0)
            subj_truth = subj_dict['truth']
            subj_mode = subj_dict['mode']
            with open(os.path.join(self.per_subject_predictions,'epoch_{0:.2f}.txt'.format(self.total_train_steps)),'a+') as f:
                f.write(self.write_prediction(subj_name,subj_mode,subj_pred,subj_error,subj_truth))
            pred_all_sets[subj_mode].append(subj_pred)
            truth_all_sets[subj_mode].append(subj_truth)
            intra_subject_variance_all_sets[subj_mode].append(subj_error.mean().item())
        for (name,pred),(_,truth),(_,std) in zip(pred_all_sets.items(),truth_all_sets.items(),intra_subject_variance_all_sets.items()):
            if len(pred) == 0:
                continue
            metrics[name + '_intra_subject_variability'] = np.mean(std)
            if self.fine_tune_task == 'regression':
                metrics[name + '_MAE_history'] = self.metrics.MAE(truth,pred)
                metrics[name + '_MSE_history'] = self.metrics.MSE(truth,pred)
                metrics[name +'_NMSE_history'] = self.metrics.NMSE(truth,pred)
            elif self.fine_tune_task == 'classification':
                if pred[0].shape[0] == 2:
                    is_binary = True
                    auroc_input = [x.div(x.sum()).tolist()[1] for x in pred]
                else:
                    is_binary = False
                    auroc_input = np.stack([x.div(x.sum()).tolist() for x in pred])
                truth_ = [x.item() for x in truth]
                metrics[name + '_Balanced_Accuracy_history'] = self.metrics.BAC(truth_,[x.argmax(dim=0).item() for x in pred])
                metrics[name + '_Regular_Accuracy_history'] = self.metrics.RAC(truth_,[x.argmax(dim=0).item() for x in pred])
                metrics[name + '_AUROC_history'] = self.metrics.AUROC(np.stack(truth_),auroc_input,binary=is_binary,mode=name)
                metrics[name + '_Confusion_Matrix_summary'] = self.metrics.CONFUSION_MAT(truth_,[x.argmax(dim=0).item() for x in pred])
        if hasattr(self.metrics,'optimial_validation_threshold'):
            t = self.metrics.optimial_validation_threshold
            print('already found optimal threshold for validation set\nrecalculating accuracy with better threshold...')
            for (name, pred), (_, truth) in zip(pred_all_sets.items(), truth_all_sets.items()):
                if len(pred) == 0:
                    continue
                pred = [x.div(x.sum()).tolist()[1] for x in pred]
                truth = [int(x) for x in truth]
                metrics[name + '_Optimal_Threshold_Balanced_Accuracy_history'] = self.metrics.BAC(truth,[int(x >= t) for x in pred])
                metrics[name + '_Optimal_Threshold_Confusion_Matrix_summary'] = self.metrics.CONFUSION_MAT(truth,[int(x >= t) for x in pred])
        for name,value in metrics.items():
            self.scalar_to_tensorboard(name,value)
            if hasattr(self,name):
                l = getattr(self,name)
                l.append(value)
                setattr(self,name,l)
            else:
                setattr(self, name, [value])
            print('{}: {}'.format(name,value))
        self.eval_iter += 1
        if mid_epoch and len(self.subject_accuracy) > 0:
            self.subject_accuracy = {k: v for k, v in self.subject_accuracy.items() if v['mode'] == 'train'}
        else:
            self.subject_accuracy = {}

    def write_losses(self,final_loss_dict,set):
        for loss_name,loss_value in final_loss_dict.items():
            title = loss_name + '_' + set
            loss_values_list = getattr(self,title + '_loss_values')
            loss_values_list.append(loss_value)
            if set == 'train':
                loss_values_list = loss_values_list[-self.running_mean_size:]
            setattr(self,title + '_loss_values',loss_values_list)

    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def register_losses(self,**kwargs):
        self.losses = {'intensity':
                           {'is_active':False,'criterion':L1Loss(),'thresholds':[0.9, 0.99],'factor':1},
                       'perceptual':
                           {'is_active':False,'criterion': Percept_Loss(**kwargs),'factor':1,'functional':False},
                       'reconstruction':
                           {'is_active':False,'criterion':L1Loss(),'factor':1,'functional':False},
                       'classification':
                           {'is_active': False, 'criterion': CrossEntropyLoss(), 'factor': 1},
                       'regression':
                           {'is_active':False,'criterion':L1Loss(),'factor':1},
                       'triplet':
                           {'is_active':False,'criterion':make_custome_triplet_loss(**kwargs),'factor':1},
                       'shuffle':
                           {'is_active': False, 'criterion': BCELoss(), 'factor': 1},
                       'dfc':
                           {'is_active': False, 'criterion': L1Loss(), 'factor': 1},
                       'parcel_penalty':
                           {'is_active':False,'criterion':L1Loss(),'factor':1,'parcel_names':[]},
                       'temporal_regularization':
                           {'is_active':False,'criterion':make_custome_triplet_loss(temporal=True,**kwargs),'factor':1}
                       }
        if 'reconstruction' in kwargs.get('task').lower():
            self.losses['intensity']['is_active'] = True
            self.losses['perceptual']['is_active'] = True
            self.losses['reconstruction']['is_active'] = True
            if kwargs.get('parcel_penalty') is not None:
                self.losses['parcel_penalty']['is_active'] = True
                self.losses['parcel_penalty']['parcel_names'] = kwargs.get('parcel_penalty')
                self.parcellation = Parcellation(**kwargs)
            if 'functional' in kwargs.get('task').lower():
                print('using functional')
                self.losses['perceptual']['functional'] = True
                self.losses['reconstruction']['functional'] = True
                self.losses['intensity']['is_active'] = False
            if kwargs.get('only_per_voxel'):
                self.losses['intensity']['is_active'] = False
        elif 'triplet' in kwargs.get('task').lower() or 'noisy_label' in kwargs.get('task').lower():
            assert kwargs.get('margin') is not None
            self.losses['triplet']['is_active'] = True
        elif 'shuffle' in kwargs.get('task').lower():
            self.losses['shuffle']['is_active'] = True
        elif 'dfc' in kwargs.get('task').lower():
            self.losses['dfc']['is_active'] = True
        else:
            self.losses[kwargs.get('fine_tune_task')]['is_active'] = True
        if kwargs.get('temporal_regularization'):
            assert kwargs.get('margin') is not None
            self.losses['temporal_regularization']['is_active'] = True
        for name, loss_D in self.losses.copy().items():
            factor = kwargs.get(name+'_factor')
            if factor is not None:
                self.losses[name]['factor'] = factor
                print('setting custom factor...\n{} loss:{}'.format(name,factor))
            if not loss_D['is_active']:
                del self.losses[name]

    def append_total_to_losses(self):
        loss_d = self.losses.copy()
        loss_d.update({'total': {'is_active': True}})
        return loss_d

    def scalar_to_tensorboard(self,tag,scalar):
        if 'summary' in tag:
            return
        tag = tag.replace('_history','')
        if self.tensorboard is not None:
            self.tensorboard.add_scalar(tag,scalar,self.total_train_steps)

    def write_prediction(self,subj_name,subj_mode,subj_pred,subj_error,subj_truth):
        if self.fine_tune_task == 'classification':
            pred_with_error = ''
            for pred,error in zip(subj_pred,subj_error):
                pred_with_error += '{:.10f}\u00B1{:.10f},'.format(pred,error)
            pred_with_error = pred_with_error[:-1]
            text = 'subject:{} ({})\noutputs: {}  -  truth: {}\n'.format(subj_name,subj_mode,pred_with_error,subj_truth)
        else:
            subj_pred = subj_pred.item()
            subj_error = subj_error.item()
            subj_truth = subj_truth.item()
            text = 'subject:{} ({})\noutputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_pred,subj_error,subj_truth)
        return text

    def set_labels_dict(self,train_loader):
        if hasattr(train_loader.dataset.dataset,'label_dict'):
            self.label_dict = train_loader.dataset.dataset.label_dict
            self.metrics.set_labels(self.label_dict)

    def final_summary_(self):
        total_step = None
        final_summary_text = ''
        if self.task == 'fine_tune' and self.fine_tune_task != 'regression':
            path = os.path.join(self.experiment_folder,self.experiment_title + '_BEST_val_auroc.pth')
        else:
            path = os.path.join(self.experiment_folder, self.experiment_title + '_BEST_val_loss.pth')
        best_ckpt = torch.load(path)
        for name,value in best_ckpt.items():
            if 'history' in name:
                final_summary_text += '{}: {}\n\n'.format(name,value)
            elif 'confusion' in name.lower():
                self.tensorboard.add_figure(name,value.figure_)
            elif 'total_train_step' in name:
                total_step = value
        self.tensorboard.add_text('final summary',final_summary_text,global_step=total_step)
        if hasattr(self,'visualize_with_TSNE') and self.visualize_with_TSNE:
            self.kwargs['loaded_model_weights_path'] = path
            self.kwargs['load_cls_embedding'] = True
            print('finished training...\nvisualizing middle vectors with TSNE')
            self.kwargs['experiment_folder'] = self.kwargs['experiment_folder'].joinpath('explainability','TSNE_vusialization_auto_gen')
            try:
                e = ExplainerTSNE(**self.kwargs)
                e.get_cls_visualization(self.sets)
                e.plot_tsne_cls_activations()
            except Exception as e:
                print(e)

    def experiment_hyperparameters_to_tensorboard(self):
        text_summary = ''
        relevant_param_names = ['batch_size','random_TR','transformer_hidden_layers','load_cls_embedding',
                                'weight_decay','augment','lr_init','lr_step','lr_gamma','sequence_length',
                                'loaded_model_weights_path','dropout']
        if self.tensorboard is not None:
            for name,value in self.kwargs.items():
                if any([param in name for param in relevant_param_names]):
                    text_summary += '{}:{}\n\n'.format(name,value)
            self.tensorboard.add_text('experiment_summary',text_summary)