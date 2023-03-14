from loss_writer import Writer
from train_utils import EarlyStop
import warnings
from tqdm import tqdm
from model import *
from losses import get_intense_voxels
from base_experiment_module import BaseModule

class Trainer(BaseModule):
    """
    main class to handle training, validation and testing.
    note: the order of commands in the constructor is necessary
    """
    def __init__(self,sets,**kwargs):
        super(Trainer, self).__init__(sets,**kwargs)
        self.create_optimizer()
        self.lr_handler.set_schedule(self.optimizer)
        self.writer = Writer(sets,**kwargs)
        self.writer.adjust_dataset(self.train_loader)
        self.early_stop = EarlyStop(**kwargs)
        self.sets = sets

        for name, loss_dict in self.writer.losses.items():
            if loss_dict['is_active']:
                print('using {} loss'.format(name))
                setattr(self, name + '_loss_func', loss_dict['criterion'])

    def create_optimizer(self):
        lr = self.lr_handler.base_lr
        params = self.model.parameters()
        weight_decay = self.kwargs.get('weight_decay')
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def eval_epoch(self,set):
        loader = self.val_loader if set == 'val' else self.test_loader
        self.eval(set)
        with torch.no_grad():
            for input_dict in tqdm(loader, position=0, leave=True):
                loss_dict, _ = self.forward_pass(input_dict)
                self.writer.write_losses(loss_dict, set=set)

    def testing(self):
        self.initialize_weights(load_cls_embedding=True)
        self.eval_epoch('test')
        self.writer.loss_summary(lr=0)
        self.writer.accuracy_summary(mid_epoch=False)
        for metric_name in dir(self.writer):
            if 'history' not in metric_name:
                continue
            metric_score = getattr(self.writer,metric_name)[-1]
            print('final test score - {} = {}'.format(metric_name,metric_score))


    def training(self):
        self.eval_epoch('val')
        self.writer.loss_summary(lr=float('nan'))
        self.writer.accuracy_summary(mid_epoch=False)
        self.writer.save_history_to_csv()
        for epoch in range(self.nEpochs):
            self.train_epoch(epoch)
            if self.early_stop.status:
                break
            self.eval_epoch('val')
            if any(['test' in set_ for set_ in self.sets]):
                self.eval_epoch('test')
            print('______epoch summary {}/{}_____\n'.format(epoch+1,self.nEpochs))
            self.writer.loss_summary(lr=self.lr_handler.schedule.get_last_lr()[0])
            self.writer.accuracy_summary(mid_epoch=False)
            self.writer.save_history_to_csv()
            self.save_checkpoint_(epoch)
            self.early_stop.check(self.writer,epoch)
            if self.early_stop.status:
                break
        self.writer.final_summary_()


    def train_epoch(self,epoch):
        self.train()
        self.writer.initialize_losses(self.sets)
        for batch_idx,input_dict in enumerate(tqdm(self.train_loader,position=0,leave=True)):
            self.writer.total_train_steps += 1
            self.optimizer.zero_grad()
            loss_dict, loss = self.forward_pass(input_dict)
            loss.backward()
            self.optimizer.step()
            self.lr_handler.schedule_check_and_update()
            self.writer.write_losses(loss_dict, set='train')
            if (batch_idx + 1) % self.validation_frequency == 0:
                partial_epoch = epoch + (batch_idx / len(self.train_loader))
                self.eval_epoch('val')
                if any(['test' in set_ for set_ in self.sets]):
                    self.eval_epoch('test')
                print('______mid-epoch summary {0:.2f}/{1:.0f}______\n'.format(partial_epoch,self.nEpochs))
                self.writer.loss_summary(lr=self.lr_handler.schedule.get_last_lr()[0])
                self.writer.accuracy_summary(mid_epoch=True)
                self.writer.save_history_to_csv()
                self.save_checkpoint_(partial_epoch)
                self.train()
                self.early_stop.check(self.writer,partial_epoch)
                if self.early_stop.status:
                    break

    def save_checkpoint_(self,epoch):
        metrics = {'total_train_step':self.writer.total_train_steps}
        for name in ['total_val_loss_history',
                     'val_MAE_history',
                     'val_AUROC_history',
                     'val_Balanced_Accuracy_history',
                     'val_Confusion_Matrix_summary',
                     'val_Optimal_Threshold_Ballanced_Accuracy_history',
                     'val_Optimal_Threshold_Confusion_Matrix_summary']:
            if hasattr(self.writer,name):
                metrics[name] = getattr(self.writer,name)[-1]
                train_set = name.replace('val','train')
                if hasattr(self.writer,train_set):
                    metrics[train_set] = getattr(self.writer, train_set)[-1]
                test_set = name.replace('val','test')
                if hasattr(self.writer,test_set):
                    metrics[test_set] = getattr(self.writer, test_set)[-1]
            else:
                metrics[name] = None
        self.model.save_checkpoint(
            self.writer.experiment_folder, self.writer.experiment_title, epoch, self.optimizer ,schedule=self.lr_handler.schedule,**metrics)


    def forward_pass(self,input_dict):
        input_dict = {k:(v.cuda() if self.cuda and torch.is_tensor(v) else v) for k,v in input_dict.items()}
        if 'triplet' in self.task or 'noisy' in self.task:
            input_dict['fmri_sequence'] = input_dict['fmri_sequence'].squeeze(0)
        output_dict = self._pytorch_model(input_dict['fmri_sequence'])
        loss_dict, loss = self.aggregate_losses(input_dict, output_dict)
        if self.task == 'fine_tune':
            self.compute_accuracy(input_dict, output_dict)
        return loss_dict, loss


    def aggregate_losses(self,input_dict,output_dict):
        final_loss_dict = {}
        final_loss_value = 0
        for loss_name, current_loss_dict in self.writer.losses.items():
            loss_func = getattr(self, 'compute_' + loss_name)
            current_loss_value = loss_func(input_dict,output_dict)
            if current_loss_value.isnan().sum() > 0:
                warnings.warn('found nans in computation')
                print('at {} loss'.format(loss_name))
            lamda = current_loss_dict['factor']
            factored_loss = current_loss_value * lamda
            final_loss_dict[loss_name] = factored_loss.item()
            final_loss_value += factored_loss
        final_loss_dict['total'] = final_loss_value.item()
        return final_loss_dict, final_loss_value

    def compute_temporal_regularization(self,input_dict,output_dict):
        init_vectors = output_dict['init_vector_sequence']
        batch,seq = init_vectors.shape[:2]
        temporal_loss = 0
        for b in range(batch):
            for t in range(seq-2):
                anchor,positive,negative = init_vectors[b,t],init_vectors[b,t+1],init_vectors[b,t+2]
                temporal_loss += self.temporal_regularization_loss_func(anchor.unsqueeze(0),positive.unsqueeze(0),negative.unsqueeze(0))
        return temporal_loss

    def compute_parcel_penalty(self,input_dict,output_dict):
        ground_truth_tensor = input_dict['fmri_sequence']
        output_tensor = output_dict['reconstructed_fmri_sequence']
        parcel_loss = 0
        for parcel_name in self.writer.losses['parcel_penalty']['parcel_name_for_increased_peenalty']:
            for norm_index in [0,1]:
                ground_truth_signal = self.writer.parcellation.get_one_parcel_time_series(ground_truth_tensor[:,norm_index],parcel_name)
                output_signal = self.writer.parcellation.get_one_parcel_time_series(output_tensor[:,norm_index],parcel_name)
                parcel_loss += self.parcel_penalty_loss_func(ground_truth_signal,output_signal)

    def compute_reconstruction(self,input_dict,output_dict):
        if self.writer.losses['reconstruction']['functional']:
            fmri_sequence = input_dict['fmri_sequence'].mean(dim=1,keepdim=True)
        else:
            fmri_sequence = input_dict['fmri_sequence'][:,0].unsqueeze(1)
        reconstruction_loss = self.reconstruction_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return reconstruction_loss

    def compute_intensity(self,input_dict,output_dict):
        per_voxel = input_dict['fmri_sequence'][:,1,:,:,:,:]
        voxels = get_intense_voxels(per_voxel, output_dict['reconstructed_fmri_sequence'].shape)
        output_intense = output_dict['reconstructed_fmri_sequence'][voxels]
        truth_intense = input_dict['fmri_sequence'][:,0][voxels.squeeze(1)]
        intensity_loss = self.intensity_loss_func(output_intense.squeeze(), truth_intense)
        return intensity_loss

    def compute_perceptual(self,input_dict,output_dict):
        if self.writer.losses['perceptual']['functional']:
            fmri_sequence = input_dict['fmri_sequence'].mean(dim=1,keepdim=True)
        else:
            fmri_sequence = input_dict['fmri_sequence'][:, 0].unsqueeze(1)
        perceptual_loss = self.perceptual_loss_func(output_dict['reconstructed_fmri_sequence'],fmri_sequence)
        return perceptual_loss

    def compute_shuffle(self,input_dict,output_dict):
        shuffle_loss = self.shuffle_loss_func(output_dict['is_shuffled'].squeeze(), output_dict['ground_truth'].squeeze())
        return shuffle_loss

    def compute_classification(self,input_dict,output_dict):
        multiclass_loss = self.classification_loss_func(output_dict['classification'].squeeze(-1), input_dict['subject_classification'].squeeze(-1))
        return multiclass_loss

    def compute_regression(self,input_dict,output_dict):
        regression_loss = self.regression_loss_func(output_dict['regression'].squeeze(-1),input_dict['subject_regression'].squeeze(-1))
        return regression_loss

    def compute_triplet(self,input_dict,output_dict):
        anchor,positive,negative = output_dict['cls']
        triplet_loss = self.triplet_loss_func(anchor.unsqueeze(0),positive.unsqueeze(0),negative.unsqueeze(0))
        return triplet_loss

    def compute_dfc(self,input_dict,output_dict):
        dfc_loss = self.dfc_loss_func(input_dict['dfc'],output_dict['dfc'])
        return dfc_loss

    def compute_accuracy(self,input_dict,output_dict):
        task = self.model.task
        out = output_dict[task].detach().clone().cpu().squeeze(-1)
        if task == 'classification':
            out = torch.nn.functional.softmax(out,dim=1)
        labels = input_dict['subject_' + task].clone().cpu()
        subjects = input_dict['subject'].clone().cpu()
        for i, subj in enumerate(subjects):
            subject = str(subj.item())
            if subject not in self.writer.subject_accuracy:
                self.writer.subject_accuracy[subject] = {'score': out[i].unsqueeze(0), 'mode': self.mode,'truth': labels[i], 'count': 1}
            else:
                self.writer.subject_accuracy[subject]['score'] = torch.cat([self.writer.subject_accuracy[subject]['score'], out[i].unsqueeze(0)], dim=0)
                self.writer.subject_accuracy[subject]['count'] += 1

